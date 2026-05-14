/**
 * @file control_policy.hpp
 * @brief OpenVINO-accelerated control-policy engine (observations → actions).
 *
 * PolicyEngine loads an ONNX policy model and runs inference via OpenVINO
 * on Intel GPU, NPU, or CPU.
 *
 * ## I/O Contract
 *
 *   - **Input**: single tensor `obs_dict` (float32, dimension = policy obs size).
 *   - **Output**: single tensor `action` (float32, dimension = G1_NUM_MOTOR = 29).
 *
 * ## Typical Usage
 *
 *   1. `Initialize(model_path)` – load and compile the model.
 *   2. Fill `GetInputBuffer()` with observation data.
 *   3. `Infer()` – runs inference and populates `GetActionBuffer()`.
 */

#ifndef POLICY_ENGINE_HPP
#define POLICY_ENGINE_HPP

#include <memory>
#include <string>
#include <array>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include "inference_backend.hpp"
#include "robot_parameters.hpp"

/**
 * @class PolicyEngine
 * @brief Runs the main RL control policy via OpenVINO on Intel GPU/NPU/CPU.
 *
 * Non-copyable; destroyed via `Destroy()` or the destructor.
 */
class PolicyEngine {
public:
  PolicyEngine() = default;
  ~PolicyEngine() { Destroy(); }

  /**
   * @brief Initialize the control policy from a model path
   * @param model_path Path to ONNX model file
   * @param use_fp16 Whether to use FP16 precision
   * @param device OpenVINO device string ("GPU", "CPU", "NPU", "AUTO:GPU,CPU")
   * @return true if initialization successful, false otherwise
   */
  bool Initialize(const std::string& model_path, bool use_fp16 = false,
                  const std::string& device = "NPU") {
    if (model_path.empty()) {
      std::cerr << "✗ PolicyEngine::Initialize - Empty model path" << std::endl;
      return false;
    }

    config_.model_path = model_path;
    config_.use_fp16 = use_fp16;
    config_.device = device;

    try {
      std::cout << "Loading policy model..." << std::endl;
      std::cout << "[Policy] Device: " << device
                << " | Precision: " << (use_fp16 ? "FP16" : "FP32") << std::endl;

      inference_engine_ = std::make_unique<TRTInferenceEngine>();

      Options options;
      options.deviceID = config_.device_id;
      std::string prefix("policy_");
      if (use_fp16) {
        options.precision = Precision::FP16;
        prefix += "fp16_";
      }

      std::string model_file;
      if (!ConvertONNXToTRT(options, model_path, model_file, prefix, false)) {
        std::cerr << "✗ Failed to prepare policy model: " << model_path << std::endl;
        inference_engine_.reset();
        return false;
      }

      // Initialize with device selection
      auto init_start = std::chrono::steady_clock::now();
      if (!inference_engine_->Initialize(model_file, device,
              use_fp16 ? Precision::FP16 : Precision::FP32)) {
        std::cerr << "✗ Failed to initialize policy on " << device << ": " << model_file << std::endl;
        inference_engine_.reset();
        return false;
      }
      auto init_end = std::chrono::steady_clock::now();
      auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();
      std::cout << "[Policy] ✓ OpenVINO initialization took " << init_ms << "ms" << std::endl;

      if (!inference_engine_->InitInputs({})) {
        std::cerr << "✗ Failed to initialize policy model inputs" << std::endl;
        inference_engine_.reset();
        return false;
      }

      std::cout << "✓ Successfully loaded policy model: " << model_file << std::endl;

      // Validate required inputs
      auto input_names = inference_engine_->GetInputTensorNames();
      if (input_names.size() != 1) {
        std::cerr << "✗ Policy must have exactly 1 input, found " << input_names.size() << ": ";
        for (const auto& n : input_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (std::find(input_names.begin(), input_names.end(), std::string("obs_dict")) == input_names.end()) {
        std::cerr << "✗ Policy input tensor 'obs_dict' not found. Available inputs: ";
        for (const auto& n : input_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      input_tensor_name_ = "obs_dict";
      if (inference_engine_->GetTensorDataType(input_tensor_name_) != DataType::FLOAT) {
        std::cerr << "✗ Policy input 'obs_dict' must be float32" << std::endl;
        inference_engine_.reset();
        return false;
      }

      // Initialize input buffer
      std::vector<int64_t> input_dims;
      inference_engine_->GetTensorShape(input_tensor_name_, input_dims);

      config_.input_dimension = std::accumulate(
        input_dims.begin(), input_dims.end(), static_cast<size_t>(1), std::multiplies<size_t>()
      );
      policy_input_buffer_.resize(config_.input_dimension, 0.0f);

      // Set initial input data (zeros)
      inference_engine_->SetInputData(input_tensor_name_, policy_input_buffer_);

      // Validate required outputs
      auto output_names = inference_engine_->GetOutputTensorNames();
      if (output_names.empty()) {
        std::cerr << "✗ Policy model has no outputs" << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (output_names.size() != 1) {
        std::cerr << "✗ Policy must have exactly 1 output, found " << output_names.size() << ": ";
        for (const auto& n : output_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (std::find(output_names.begin(), output_names.end(), std::string("action")) == output_names.end()) {
        std::cerr << "✗ Policy output tensor 'action' not found. Available outputs: ";
        for (const auto& n : output_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      output_tensor_name_ = "action";

      // Get output dimensions
      std::vector<int64_t> output_dims;
      if (inference_engine_->GetTensorShape(output_tensor_name_, output_dims)) {
        config_.action_dimension = std::accumulate(
          output_dims.begin(), output_dims.end(), 1, std::multiplies<size_t>()
        );
      }

      // Validate action dimension matches robot configuration
      if (config_.action_dimension != G1_NUM_MOTOR) {
        std::cerr << "✗ Policy action dimension (" << config_.action_dimension
                  << ") doesn't match G1 robot motors (" << G1_NUM_MOTOR << ")" << std::endl;
        inference_engine_.reset();
        return false;
      }

      action_buffer_.resize(config_.action_dimension, 0.0f);

      // Run warmup inference
      std::cout << "[Policy] Running warmup inference..." << std::endl;
      auto warmup_start = std::chrono::steady_clock::now();
      if (!inference_engine_->Enqueue(nullptr)) {
        std::cout << "[Policy] ⚠ Warmup inference failed (non-fatal)" << std::endl;
      }
      auto warmup_end = std::chrono::steady_clock::now();
      auto warmup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(warmup_end - warmup_start).count();
      std::cout << "[Policy] ✓ Warmup inference took " << warmup_ms << "ms" << std::endl;

      initialized_ = true;
      std::cout << "✓ Policy engine initialized successfully!" << std::endl;
      std::cout << "  Model: " << model_path << std::endl;
      std::cout << "  Device: " << device << std::endl;
      std::cout << "  Input dimension: " << config_.input_dimension << std::endl;
      std::cout << "  Action dimension: " << config_.action_dimension << std::endl;
      std::cout << "  Input tensor: " << input_tensor_name_ << std::endl;
      std::cout << "  Output tensor: " << output_tensor_name_ << std::endl;
      std::cout << "  Precision: " << (use_fp16 ? "FP16" : "FP32") << std::endl;
      return true;

    } catch (const std::exception& e) {
      std::cerr << "✗ PolicyEngine::Initialize - Exception: " << e.what() << std::endl;
      inference_engine_.reset();
      return false;
    }
  }

  /**
   * @brief Set input data for the control policy
   * @param data Input data buffer
   * @param element_count Number of elements
   */
  template<typename T>
  void SetInputData(const T* data, size_t element_count) {
    if (!initialized_ || !inference_engine_) { return; }
    inference_engine_->SetInputData(input_tensor_name_, data, element_count);
  }

  /**
   * @brief Set input data using std::vector
   * @param data Input data vector
   */
  template<typename T>
  void SetInputData(const std::vector<T>& data) {
    if (!initialized_ || !inference_engine_) { return; }
    inference_engine_->SetInputData(input_tensor_name_, data);
  }

  /**
   * @brief Run control policy inference and populate internal action buffer
   * @return true if inference successful, false otherwise
   */
  bool Infer() {
    if (!initialized_) {
      std::cerr << "✗ PolicyEngine::Infer - Not initialized" << std::endl;
      return false;
    }
    if (!inference_engine_) {
      std::cerr << "✗ PolicyEngine::Infer - Engine not initialized" << std::endl;
      return false;
    }

    // Transfer input data
    inference_engine_->SetInputData(input_tensor_name_, policy_input_buffer_);

    // Run inference
    if (!inference_engine_->Enqueue(nullptr)) {
      std::cerr << "✗ PolicyEngine::Infer - Inference failed" << std::endl;
      return false;
    }

    // Read output
    inference_engine_->GetOutputData(output_tensor_name_, action_buffer_);

    return true;
  }

  /**
   * @brief Get input dimension
   * @return Input dimension size
   */
  size_t GetInputDimension() const { return config_.input_dimension; }

  /**
   * @brief Get action dimension
   * @return Action dimension size
   */
  size_t GetActionDimension() const { return config_.action_dimension; }

  /**
   * @brief Check if control policy is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const { return initialized_; }

  /**
   * @brief Get input tensor names
   * @return Vector of input tensor names
   */
  std::vector<std::string> GetInputTensorNames() const {
    if (!initialized_ || !inference_engine_) { return {}; }
    return inference_engine_->GetInputTensorNames();
  }

  /**
   * @brief Get output tensor names
   * @return Vector of output tensor names
   */
  std::vector<std::string> GetOutputTensorNames() const {
    if (!initialized_ || !inference_engine_) { return {}; }
    return inference_engine_->GetOutputTensorNames();
  }

  /**
   * @brief Get reference to internal input buffer
   * @return Reference to input buffer
   */
  std::vector<float>& GetInputBuffer() { return policy_input_buffer_; }

  /**
   * @brief Get reference to internal action buffer
   * @return Reference to action buffer
   */
  std::vector<float>& GetActionBuffer() { return action_buffer_; }

  /**
   * @brief Get input tensor name
   * @return Input tensor name
   */
  const std::string& GetInputTensorName() const { return input_tensor_name_; }

  /**
   * @brief Get output tensor name
   * @return Output tensor name
   */
  const std::string& GetOutputTensorName() const { return output_tensor_name_; }

  /**
   * @brief Destroy and clean up control policy resources
   */
  void Destroy() {
    if (!initialized_) { return; }
    if (inference_engine_) {
      inference_engine_->Destroy();
      inference_engine_.reset();
    }
    initialized_ = false;
  }

private:
  struct Config {
    std::string model_path;
    std::string device = "GPU";
    int device_id = 0;
    size_t input_dimension = 0;
    size_t action_dimension = 0;
    bool use_fp16 = false;
  };
  Config config_;

  std::unique_ptr<TRTInferenceEngine> inference_engine_;

  std::string input_tensor_name_;
  std::string output_tensor_name_;

  std::vector<float> policy_input_buffer_;
  std::vector<float> action_buffer_;

  bool initialized_ = false;
};

#endif // POLICY_ENGINE_HPP
