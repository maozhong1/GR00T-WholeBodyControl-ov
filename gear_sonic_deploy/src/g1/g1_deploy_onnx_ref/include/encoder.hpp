/**
 * @file encoder.hpp
 * @brief OpenVINO-accelerated encoder engine (observations → token state).
 *
 * EncoderEngine loads an ONNX encoder model and runs inference via OpenVINO
 * on Intel GPU, NPU, or CPU.
 *
 * The encoder compresses high-dimensional robot observations into a compact
 * latent / token representation that is then consumed by the control policy
 * as the `token_state` observation.
 *
 * ## I/O Contract
 *
 *   - **Input**: single tensor `obs_dict` (float32, encoder observation dim).
 *   - **Output**: single tensor `encoded_tokens` (float32, token dimension).
 *
 * ## Configuration
 *
 * Driven by the `encoder:` section of `observation_config.yaml` (parsed by
 * ObservationConfigParser).  The encoder is created only when `token_state`
 * is enabled in the observation config.  The model path is provided via the
 * `--encoder-model` command-line argument.
 *
 * ## Typical Usage
 *
 *   1. `Initialize(model_path)` – load and compile the model with OpenVINO.
 *   2. Fill `GetInputBuffer()` with encoder observations.
 *   3. `Encode()` – runs inference and populates `GetTokenBuffer()`.
 */

#ifndef ENCODER_HPP
#define ENCODER_HPP

#include <memory>
#include <string>
#include <array>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include "inference_backend.hpp"

/**
 * @class EncoderEngine
 * @brief Runs the observation encoder via OpenVINO on Intel GPU/NPU/CPU.
 *
 * Non-copyable. Owns the inference engine and I/O buffers.
 */
class EncoderEngine {
public:
  EncoderEngine() = default;
  ~EncoderEngine() { Destroy(); }

  /**
   * @brief Initialize the encoder engine from a model path
   * @param model_path Path to ONNX model file
   * @param use_fp16 Whether to use FP16 precision
   * @param device OpenVINO device string ("GPU", "CPU", "NPU", "AUTO:GPU,CPU")
   * @param npu_tiles Number of NPU tiles to use (0 = auto, 1-N = specific count)
   * @param model_priority OpenVINO model priority ("HIGH", "NORMAL", "LOW")
   * @return true if initialization successful, false otherwise
   */
  bool Initialize(const std::string& model_path, bool use_fp16 = false,
                  const std::string& device = "NPU", int npu_tiles = 0,
                  const std::string& model_priority = "NORMAL") {
    if (model_path.empty()) {
      std::cerr << "✗ EncoderEngine::Initialize - Empty model path" << std::endl;
      return false;
    }

    config_.model_path = model_path;
    config_.use_fp16 = use_fp16;
    config_.device = device;

    try {
      std::cout << "Loading encoder model..." << std::endl;
      std::cout << "[Encoder] Device: " << device
                << " | Precision: " << (use_fp16 ? "FP16" : "FP32") << std::endl;

      inference_engine_ = std::make_unique<TRTInferenceEngine>();

      Options options;
      options.deviceID = config_.device_id;
      std::string prefix("encoder_");
      if (use_fp16) { options.precision = Precision::FP16; prefix += "fp16_"; }

      std::string model_file;
      if (!ConvertONNXToTRT(options, model_path, model_file, prefix, false)) {
        std::cerr << "✗ Failed to prepare encoder model: " << model_path << std::endl;
        inference_engine_.reset();
        return false;
      }

      // Initialize with device selection and tile configuration
      auto init_start = std::chrono::steady_clock::now();
      if (!inference_engine_->Initialize(model_file, device,
              use_fp16 ? Precision::FP16 : Precision::FP32, npu_tiles, model_priority)) {
        std::cerr << "✗ Failed to initialize encoder on " << device << ": " << model_file << std::endl;
        inference_engine_.reset();
        return false;
      }
      auto init_end = std::chrono::steady_clock::now();
      auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();
      std::cout << "[Encoder] ✓ OpenVINO initialization took " << init_ms << "ms" << std::endl;

      if (!inference_engine_->InitInputs({})) {
        std::cerr << "✗ Failed to initialize encoder model inputs" << std::endl;
        inference_engine_.reset();
        return false;
      }

      auto output_names = inference_engine_->GetOutputTensorNames();
      if (output_names.empty()) {
        std::cerr << "✗ Encoder model has no outputs" << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (output_names.size() != 1) {
        std::cerr << "✗ Encoder must have exactly 1 output, found " << output_names.size() << ": ";
        for (const auto& n : output_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (std::find(output_names.begin(), output_names.end(), std::string("encoded_tokens")) == output_names.end()) {
        std::cerr << "✗ Encoder output tensor 'encoded_tokens' not found. Available outputs: ";
        for (const auto& n : output_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      output_tensor_name_ = "encoded_tokens";

      std::vector<int64_t> output_dims;
      if (inference_engine_->GetTensorShape(output_tensor_name_, output_dims)) {
        config_.token_dimension = std::accumulate(output_dims.begin(), output_dims.end(), static_cast<size_t>(1), std::multiplies<size_t>());
      }

      token_buffer_.resize(config_.token_dimension, 0.0f);

      auto input_names = inference_engine_->GetInputTensorNames();
      if (input_names.empty()) {
        std::cerr << "✗ Encoder model has no inputs" << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (input_names.size() != 1) {
        std::cerr << "✗ Encoder must have exactly 1 input, found " << input_names.size() << ": ";
        for (const auto& n : input_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      if (std::find(input_names.begin(), input_names.end(), std::string("obs_dict")) == input_names.end()) {
        std::cerr << "✗ Encoder input tensor 'obs_dict' not found. Available inputs: ";
        for (const auto& n : input_names) std::cerr << n << ' ';
        std::cerr << std::endl;
        inference_engine_.reset();
        return false;
      }
      input_tensor_name_ = "obs_dict";
      if (inference_engine_->GetTensorDataType(input_tensor_name_) != DataType::FLOAT) {
        std::cerr << "✗ Encoder input 'obs_dict' must be float32" << std::endl;
        inference_engine_.reset();
        return false;
      }

      // Initialize input buffer
      std::vector<int64_t> input_dims;
      if (!inference_engine_->GetTensorShape(input_tensor_name_, input_dims)) {
        std::cerr << "✗ Failed to get encoder input shape" << std::endl;
        inference_engine_.reset();
        return false;
      }

      config_.input_dimension = std::accumulate(
        input_dims.begin(), input_dims.end(), static_cast<size_t>(1), std::multiplies<size_t>()
      );
      encoder_input_buffer_.resize(config_.input_dimension, 0.0f);

      // Set initial input data (zeros)
      inference_engine_->SetInputData(input_tensor_name_, encoder_input_buffer_);

      // Run warmup inference
      std::cout << "[Encoder] Running warmup inference..." << std::endl;
      auto warmup_start = std::chrono::steady_clock::now();
      if (!inference_engine_->Enqueue(nullptr)) {
        std::cout << "[Encoder] ⚠ Warmup inference failed (non-fatal)" << std::endl;
      }
      auto warmup_end = std::chrono::steady_clock::now();
      auto warmup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(warmup_end - warmup_start).count();
      std::cout << "[Encoder] ✓ Warmup inference took " << warmup_ms << "ms" << std::endl;

      initialized_ = true;
      std::cout << "✓ Encoder initialized successfully!" << std::endl;
      std::cout << "  Model: " << model_path << std::endl;
      std::cout << "  Device: " << device << std::endl;
      std::cout << "  Input dimension: " << config_.input_dimension << std::endl;
      std::cout << "  Token dimension: " << config_.token_dimension << std::endl;
      std::cout << "  Input tensor: " << input_tensor_name_ << std::endl;
      std::cout << "  Output tensor: " << output_tensor_name_ << std::endl;
      return true;

    } catch (const std::exception& e) {
      std::cerr << "✗ EncoderEngine::Initialize - Exception: " << e.what() << std::endl;
      inference_engine_.reset();
      return false;
    }
  }


  /**
   * @brief Set input data for the encoder
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
   * @brief Run encoder inference and populate internal token buffer
   * @return true if inference successful, false otherwise
   */
  bool Encode() {
    if (!initialized_) { std::cerr << "✗ EncoderEngine::Encode - Not initialized" << std::endl; return false; }
    if (!inference_engine_) { std::cerr << "✗ EncoderEngine::Encode - Engine not initialized" << std::endl; return false; }

    // Transfer input data
    inference_engine_->SetInputData(input_tensor_name_, encoder_input_buffer_);

    // Run inference
    if (!inference_engine_->Enqueue(nullptr)) {
      std::cerr << "✗ EncoderEngine::Encode - Inference failed" << std::endl;
      return false;
    }

    // Read output
    inference_engine_->GetOutputData(output_tensor_name_, token_buffer_);

    return true;
  }

  /**
   * @brief Get input dimension
   * @return Input dimension size
   */
  size_t GetInputDimension() const { return config_.input_dimension; }

  /**
   * @brief Get token dimension
   * @return Token dimension size
   */
  size_t GetTokenDimension() const { return config_.token_dimension; }

  /**
   * @brief Check if encoder is initialized
   * @return true if initialized, false otherwise
   */
  bool IsInitialized() const { return initialized_; }

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
   * @brief Get reference to internal input buffer
   * @return Reference to input buffer
   */
  std::vector<float>& GetInputBuffer() { return encoder_input_buffer_; }

  /**
   * @brief Get reference to internal token buffer
   * @return Reference to token buffer
   */
  std::vector<float>& GetTokenBuffer() { return token_buffer_; }

  /**
   * @brief Destroy and clean up encoder resources
   */
  void Destroy() {
    if (!initialized_) { return; }
    if (inference_engine_) { inference_engine_->Destroy(); inference_engine_.reset(); }
    initialized_ = false;
  }

private:
  struct Config {
    std::string model_path;
    std::string device = "GPU";
    int device_id = 0;
    size_t input_dimension = 0;
    size_t token_dimension = 0;
    bool use_fp16 = false;
  };
  Config config_;

  std::unique_ptr<TRTInferenceEngine> inference_engine_;

  std::string input_tensor_name_;
  std::string output_tensor_name_;

  std::vector<float> encoder_input_buffer_;
  std::vector<float> token_buffer_;

  bool initialized_ = false;
};

#endif // ENCODER_HPP
