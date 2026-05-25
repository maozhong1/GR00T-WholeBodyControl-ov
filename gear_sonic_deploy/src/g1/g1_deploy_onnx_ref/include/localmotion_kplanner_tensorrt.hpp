/**
 * @file localmotion_kplanner_tensorrt.hpp
 * @brief OpenVINO (CPU) backend for the locomotion planner.
 *
 * LocalMotionPlannerTensorRT is a concrete implementation of
 * LocalMotionPlannerBase that runs planner inference on the CPU via OpenVINO.
 *
 * ## Pipeline
 *
 *   1. Constructor loads the ONNX model via OpenVINO and runs a warmup.
 *   2. `UpdateInputTensors()` writes new locomotion commands into buffers.
 *   3. `RunInference()` sets inputs, runs Enqueue(), reads outputs.
 *   4. Output buffers (`mujoco_qpos_values_`, `num_pred_frames_values_`)
 *      are read by the base class to resample the trajectory at 50 Hz.
 *
 * ## Model Versions
 *
 *   Version | Inputs | Notes
 *   --------|--------|------
 *   0       | 6      | Basic: context, mode, target_vel, movement/facing direction, random_seed.
 *   1–2     | 11     | Adds: height, has_specific_target, specific_target_positions/headings, allowed_pred_num_tokens.
 */

#ifndef LOCALMOTION_KPLANNER_TENSORRT_HPP
#define LOCALMOTION_KPLANNER_TENSORRT_HPP

#include "inference_backend.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "localmotion_kplanner.hpp"

/**
 * @class LocalMotionPlannerTensorRT
 * @brief OpenVINO (CPU) backend for the locomotion planner.
 */
class LocalMotionPlannerTensorRT : public LocalMotionPlannerBase {
public:
    /**
     * @brief Constructor for LocalMotionPlannerTensorRT
     * @param use_fp16 Whether to use FP16 precision
     * @param device_id Unused (kept for API compatibility)
     * @param config Planner configuration parameters
     */
    LocalMotionPlannerTensorRT(bool use_fp16,
                               int device_id = 0,
                               const PlannerConfig& config = PlannerConfig())
        : LocalMotionPlannerBase(config), use_fp16_(use_fp16),
          device_id_(device_id) {

        inference_engine_ = std::make_unique<TRTInferenceEngine>();

        // Initialize input buffers
        mode_values_.resize(1);
        target_vel_values_.resize(1);
        movement_direction_values_.resize(3, 0.0f);
        facing_direction_values_.resize(3, 0.0f);
        random_seed_values_.resize(1);
        context_qpos_values_.resize(4 * (G1_NUM_MOTOR + 7));

        // Version 1+ inputs
        target_height_values_.resize(1, -1.0f);
        has_specific_target_.resize(1, 0);
        specific_target_positions_.resize(12, 0.0f);
        specific_target_headings_.resize(4, 0.0f);
        allowed_pred_num_tokens_.resize(11, 0);

        // Default allowed tokens
        allowed_pred_num_tokens_[0] = 0; // 6 tokens
        allowed_pred_num_tokens_[1] = 0; // 7
        allowed_pred_num_tokens_[2] = 0; // 8
        allowed_pred_num_tokens_[3] = 1; // 9
        allowed_pred_num_tokens_[4] = 1; // 10
        allowed_pred_num_tokens_[5] = 1;

        // Initialize output buffers
        mujoco_qpos_values_.resize(64 * 36);
        num_pred_frames_values_.resize(1);

        bool success = InitializeEngine();
        if (!success) {
            std::cout << "✗ Failed to initialize planner engine" << std::endl;
            throw std::runtime_error("Failed to initialize planner engine");
        }
    }

    ~LocalMotionPlannerTensorRT() = default;

    bool InitializeSpecific() override {
        return inference_engine_ != nullptr;
    }

private:
    // ------------------------------------------------------------------
    // Engine components
    // ------------------------------------------------------------------
    std::unique_ptr<TRTInferenceEngine> inference_engine_;
    bool use_fp16_;
    int device_id_;

    // ------------------------------------------------------------------
    // Input data buffers
    // ------------------------------------------------------------------
    std::vector<float> context_qpos_values_;          ///< [4 × 36] Context frames.
    std::vector<int64_t> mode_values_;                ///< [1] Locomotion mode.
    std::vector<float> target_vel_values_;            ///< [1] Target speed.
    std::vector<float> movement_direction_values_;    ///< [3] Movement direction.
    std::vector<float> facing_direction_values_;      ///< [3] Facing direction.
    std::vector<int64_t> random_seed_values_;         ///< [1] Random seed.

    // Version 1–2 additional inputs
    std::vector<float> target_height_values_;         ///< [1] Target body height.
    std::vector<int64_t> has_specific_target_;        ///< [1] Waypoint target flag.
    std::vector<float> specific_target_positions_;    ///< [12] 4 waypoint positions × xyz.
    std::vector<float> specific_target_headings_;     ///< [4] 4 waypoint heading angles.
    std::vector<int64_t> allowed_pred_num_tokens_;    ///< [11] Prediction token mask.

    // ------------------------------------------------------------------
    // Output data buffers
    // ------------------------------------------------------------------
    std::vector<float> mujoco_qpos_values_;           ///< [frames × 36] Predicted qpos.
    std::vector<int32_t> num_pred_frames_values_;     ///< [1] Number of predicted frames.

    // ------------------------------------------------------------------
    // Inference logging
    // ------------------------------------------------------------------
    uint64_t planner_infer_count_ = 0;
    uint64_t planner_latency_sum_us_ = 0;

    // ------------------------------------------------------------------
    // CSV dump for accuracy validation
    // ------------------------------------------------------------------
    bool csv_header_written_ = false;
    uint64_t csv_dump_count_ = 0;

    /**
     * @brief Dump all planner input tensors to CSV for NPU accuracy validation.
     *
     * Controlled by config_.dump_input_csv (enable/disable) and
     * config_.max_input_dump_cnt (max rows, 0 = unlimited).
     *
     * Each row = one inference call. Columns:
     *   timestamp_us, mode[1], target_vel[1], target_height[1],
     *   movement_direction[3], facing_direction[3], random_seed[1],
     *   has_specific_target[1], specific_target_positions[12],
     *   specific_target_headings[4], allowed_pred_num_tokens[11],
     *   context_mujoco_qpos[144]
     */
    void DumpInputsToCSV() {
        if (!config_.dump_input_csv) return;
        if (config_.max_input_dump_cnt > 0 &&
            csv_dump_count_ >= static_cast<uint64_t>(config_.max_input_dump_cnt)) return;

        std::ofstream csv;
        if (!csv_header_written_) {
            csv.open(config_.dump_csv_path, std::ios::out | std::ios::trunc);
            if (!csv.is_open()) {
                std::cerr << "[Planner] ⚠ Cannot open CSV dump file: " << config_.dump_csv_path << std::endl;
                return;
            }
            // Write header
            csv << "infer_count,timestamp_us";
            csv << ",mode(shape=1;dtype=int64)";
            csv << ",target_vel(shape=1;dtype=float)";
            csv << ",target_height(shape=1;dtype=float)";
            for (int i = 0; i < 3; ++i) csv << ",movement_direction_" << i << "(shape=3;dtype=float)";
            for (int i = 0; i < 3; ++i) csv << ",facing_direction_" << i << "(shape=3;dtype=float)";
            csv << ",random_seed(shape=1;dtype=int64)";
            csv << ",has_specific_target(shape=1;dtype=int64)";
            for (int i = 0; i < 12; ++i) csv << ",specific_target_positions_" << i << "(shape=1x4x3;dtype=float)";
            for (int i = 0; i < 4; ++i) csv << ",specific_target_headings_" << i << "(shape=1x4;dtype=float)";
            for (int i = 0; i < 11; ++i) csv << ",allowed_pred_num_tokens_" << i << "(shape=1x11;dtype=int64)";
            for (int i = 0; i < 4 * (G1_NUM_MOTOR + 7); ++i)
                csv << ",context_mujoco_qpos_" << i << "(shape=1x4x36;dtype=float)";
            csv << "\n";
            csv_header_written_ = true;
        } else {
            csv.open(config_.dump_csv_path, std::ios::out | std::ios::app);
            if (!csv.is_open()) return;
        }

        auto now = std::chrono::steady_clock::now().time_since_epoch();
        auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(now).count();

        csv << std::setprecision(8);
        csv << planner_infer_count_ << "," << ts_us;
        csv << "," << mode_values_[0];
        csv << "," << target_vel_values_[0];
        csv << "," << target_height_values_[0];
        for (int i = 0; i < 3; ++i) csv << "," << movement_direction_values_[i];
        for (int i = 0; i < 3; ++i) csv << "," << facing_direction_values_[i];
        csv << "," << random_seed_values_[0];
        csv << "," << has_specific_target_[0];
        for (int i = 0; i < 12; ++i) csv << "," << specific_target_positions_[i];
        for (int i = 0; i < 4; ++i) csv << "," << specific_target_headings_[i];
        for (int i = 0; i < 11; ++i) csv << "," << allowed_pred_num_tokens_[i];
        for (size_t i = 0; i < context_qpos_values_.size(); ++i) csv << "," << context_qpos_values_[i];
        csv << "\n";
        csv.close();
        csv_dump_count_++;

        if (config_.max_input_dump_cnt > 0 &&
            csv_dump_count_ >= static_cast<uint64_t>(config_.max_input_dump_cnt)) {
            std::cout << "[Planner] CSV dump reached max count (" << config_.max_input_dump_cnt
                      << "), stopping dump." << std::endl;
        }
    }

    /// Named tensor identifiers.
    struct TensorNames {
        std::string context_qpos = "context_mujoco_qpos";
        std::string mode = "mode";
        std::string target_vel = "target_vel";
        std::string target_height = "height";
        std::string has_specific_target = "has_specific_target";
        std::string specific_target_positions = "specific_target_positions";
        std::string specific_target_headings = "specific_target_headings";
        std::string allowed_pred_num_tokens = "allowed_pred_num_tokens";
        std::string movement_direction = "movement_direction";
        std::string facing_direction = "facing_direction";
        std::string random_seed = "random_seed";
        std::string mujoco_qpos_output = "mujoco_qpos";
        std::string num_pred_frames_output = "num_pred_frames";
    } tensor_names_;

    /**
     * @brief Load and compile the ONNX model with OpenVINO, run warmup.
     */
    bool InitializeEngine() {
        std::cout << "Initialize Engine..." << std::endl;

        Options options;
        options.deviceID = device_id_;
        std::string prefix("planner_");
        if (use_fp16_) {
            options.precision = Precision::FP16;
            prefix += "fp16_";
        }

        std::string model_file;
        std::string onnxModelPath = config_.model_path;

        if (!ConvertONNXToTRT(options, onnxModelPath, model_file, prefix, false)) {
            std::cout << "✗ Failed to prepare planner model: " << onnxModelPath << std::endl;
            return false;
        }

        // Device-aware precision:
        //   GPU/NPU → FP16 (native, handled by OVInferenceEngine)
        //   CPU → FP32 (AVX/AVX-512 accelerated)
        Precision planner_precision = use_fp16_ ? Precision::FP16 : Precision::FP32;
        std::string precision_str = (config_.device == "GPU" || config_.device == "NPU" || use_fp16_) ? "FP16" : "FP32";
        std::cout << "[Planner] Initializing with OpenVINO backend (" << config_.device << ", " << precision_str << ")..." << std::endl;
        std::cout << "[Planner] Model: " << model_file << std::endl;

        auto init_start = std::chrono::steady_clock::now();
        if (!inference_engine_->Initialize(model_file, config_.device, planner_precision, config_.npu_tiles)) {
            std::cout << "✗ Failed to initialize planner on " << config_.device << ": " << model_file << std::endl;
            return false;
        }
        auto init_end = std::chrono::steady_clock::now();
        auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();
        std::cout << "[Planner] ✓ OpenVINO initialization took " << init_ms << "ms" << std::endl;

        if (!inference_engine_->InitInputs({})) {
            std::cout << "✗ Failed to initialize planner model inputs: " << model_file << std::endl;
            return false;
        }

        std::cout << "✓ Successfully loaded planner model: " << model_file << std::endl;

        // Run warmup inference
        std::cout << "[Planner] Running warmup inference..." << std::endl;
        auto warmup_start = std::chrono::steady_clock::now();
        if (!inference_engine_->Enqueue(nullptr)) {
            std::cout << "[Planner] ⚠ Warmup inference failed (non-fatal)" << std::endl;
        }
        auto warmup_end = std::chrono::steady_clock::now();
        auto warmup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(warmup_end - warmup_start).count();
        std::cout << "[Planner] ✓ Warmup inference took " << warmup_ms << "ms" << std::endl;

        // Validate tensor names
        std::vector<std::string> inputNames = inference_engine_->GetInputTensorNames();
        std::vector<std::string> outputNames = inference_engine_->GetOutputTensorNames();

        size_t expectedInputs = (config_.version == 1 || config_.version == 2) ? 11 : 6;
        if (inputNames.size() != expectedInputs) {
            std::cout << "Model version: " << config_.version << std::endl;
            std::cout << "Model has " << inputNames.size() << " inputs, expected " << expectedInputs << std::endl;
            std::cout << "✗ Failed to initialize planner engine (input count mismatch)" << std::endl;
            return false;
        }

        {
            std::vector<std::string> requiredNames = {
                tensor_names_.context_qpos,
                tensor_names_.target_vel,
                tensor_names_.mode,
                tensor_names_.movement_direction,
                tensor_names_.facing_direction,
                tensor_names_.random_seed
            };
            if (config_.version == 1 || config_.version == 2) {
                requiredNames.push_back(tensor_names_.target_height);
                requiredNames.push_back(tensor_names_.has_specific_target);
                requiredNames.push_back(tensor_names_.specific_target_positions);
                requiredNames.push_back(tensor_names_.specific_target_headings);
                requiredNames.push_back(tensor_names_.allowed_pred_num_tokens);
            }
            for (const auto &req : requiredNames) {
                if (std::find(inputNames.begin(), inputNames.end(), req) == inputNames.end()) {
                    std::cout << "✗ Missing required input tensor: " << req << std::endl;
                    return false;
                }
            }
        }

        std::cout << "Input tensors:" << std::endl;
        for (const auto& name : inputNames) {
            std::vector<int64_t> shape;
            inference_engine_->GetTensorShape(name, shape);
            std::cout << "  " << name << " ";
            for (auto& dim : shape) {
                std::cout << dim << " ";
            }

            auto dataType = inference_engine_->GetTensorDataType(name);
            if (dataType == DataType::FLOAT) {
                std::cout << "float";
            } else if (dataType == DataType::INT64) {
                std::cout << "int64";
            } else if (dataType == DataType::INT32) {
                std::cout << "int32";
            } else {
                std::cout << "unknown";
            }
            std::cout << std::endl;
        }

        std::cout << "Output tensors:" << std::endl;
        for (const auto& name : outputNames) {
            std::vector<int64_t> shape;
            inference_engine_->GetTensorShape(name, shape);
            std::cout << "  " << name << " ";
            for (auto& dim : shape) {
                std::cout << dim << " ";
            }

            auto dataType = inference_engine_->GetTensorDataType(name);
            if (dataType == DataType::FLOAT) {
                std::cout << "float";
            } else if (dataType == DataType::INT32) {
                std::cout << "int32";
            } else {
                std::cout << "unknown";
            }
            std::cout << std::endl;
        }

        std::cout << "✓ Planner model loaded successfully!" << std::endl;
        return true;
    }

    /// Run planner inference: set inputs → infer → read outputs.
    void RunInference() override {
        auto infer_start = std::chrono::steady_clock::now();

        inference_engine_->SetInputData(tensor_names_.context_qpos, context_qpos_values_);
        inference_engine_->SetInputData(tensor_names_.facing_direction, facing_direction_values_);
        inference_engine_->SetInputData(tensor_names_.mode, mode_values_);
        inference_engine_->SetInputData(tensor_names_.target_vel, target_vel_values_);
        inference_engine_->SetInputData(tensor_names_.movement_direction, movement_direction_values_);
        inference_engine_->SetInputData(tensor_names_.random_seed, random_seed_values_);

        if (config_.version == 1 || config_.version == 2) {
            inference_engine_->SetInputData(tensor_names_.target_height, target_height_values_);
            inference_engine_->SetInputData(tensor_names_.has_specific_target, has_specific_target_);
            inference_engine_->SetInputData(tensor_names_.specific_target_positions, specific_target_positions_);
            inference_engine_->SetInputData(tensor_names_.specific_target_headings, specific_target_headings_);
            inference_engine_->SetInputData(tensor_names_.allowed_pred_num_tokens, allowed_pred_num_tokens_);
        }

        // Dump inputs to CSV for NPU accuracy validation
        DumpInputsToCSV();

        if (!inference_engine_->Enqueue(nullptr)) {
            std::cerr << "[Planner] ✗ Inference failed!" << std::endl;
            return;
        }

        inference_engine_->GetOutputData(tensor_names_.mujoco_qpos_output, mujoco_qpos_values_);
        inference_engine_->GetOutputData(tensor_names_.num_pred_frames_output, num_pred_frames_values_);

        auto infer_end = std::chrono::steady_clock::now();
        auto latency_us = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start).count();

        // Log planner inference latency periodically
        planner_infer_count_++;
        planner_latency_sum_us_ += latency_us;
        if (planner_infer_count_ % 10 == 1) {
            bool has_nan = false, has_inf = false;
            float max_abs_val = 0.0f;
            for (size_t i = 0; i < mujoco_qpos_values_.size(); ++i) {
                float v = mujoco_qpos_values_[i];
                if (std::isnan(v)) { has_nan = true; break; }
                if (std::isinf(v)) { has_inf = true; break; }
                max_abs_val = std::max(max_abs_val, std::abs(v));
            }
            int num_pred = num_pred_frames_values_.empty() ? -1 : static_cast<int>(num_pred_frames_values_[0]);
            double avg_latency = static_cast<double>(planner_latency_sum_us_) / planner_infer_count_;

            std::cout << "[Planner] Inference #" << planner_infer_count_
                      << " | latency: " << latency_us << "us"
                      << " (avg: " << static_cast<int>(avg_latency) << "us)"
                      << " | num_pred_frames: " << num_pred
                      << " | max|qpos|: " << max_abs_val;
            if (has_nan) std::cout << " | ⚠ NaN DETECTED";
            if (has_inf) std::cout << " | ⚠ Inf DETECTED";
            std::cout << " | mode: " << mode_values_[0]
                      << " vel: " << target_vel_values_[0]
                      << " dir: [" << movement_direction_values_[0] << ","
                      << movement_direction_values_[1] << ","
                      << movement_direction_values_[2] << "]"
                      << std::endl;
        }
    }

    /// Write new locomotion commands into input buffers.
    void UpdateInputTensors(int mode_value,
                           float target_vel,
                           float target_height,
                           const std::array<float, 3>& movement_direction,
                           const std::array<float, 3>& facing_direction,
                           int random_seed) override {
        // Update mode
        if (mode_value < 0 || mode_value >= GetValidModeValueRange()) {
            std::cout << "✗ Invalid mode value: " << mode_value << ". Please verify the planner model version and the mode value range" << std::endl;
        }
        mode_values_[0] = mode_value < GetValidModeValueRange() ? mode_value : 0;

        // Update target velocity
        target_vel_values_[0] = target_vel;

        if (config_.version == 1 || config_.version == 2) {
            target_height_values_[0] = target_height;
        }

        // Update movement direction
        movement_direction_values_[0] = movement_direction[0];
        movement_direction_values_[1] = movement_direction[1];
        movement_direction_values_[2] = movement_direction[2];

        // Update facing direction
        facing_direction_values_[0] = facing_direction[0];
        facing_direction_values_[1] = facing_direction[1];
        facing_direction_values_[2] = facing_direction[2];

        // Update random seed if provided
        if (random_seed != -1) {
            current_random_seed_ = random_seed;
            random_seed_values_[0] = random_seed;
        }

        // Log replanning values
        std::cout << "Replanning with mode: ";
        switch(mode_values_[0])
        {
            case 0:
                std::cout << "IDLE";
                break;
            case 1:
                std::cout << "SLOW_WALK";
                break;
            case 2:
                std::cout << "WALK";
                break;
            case 3:
                std::cout << "RUN";
                break;
            case 4:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_SQUAT";
                }
                else
                {
                    std::cout << "BOXING";
                }
                break;
            case 5:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_KNEEL_TWO_LEGS";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 6:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_KNEEL";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 7:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_LYING_FACE_DOWN";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 8:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_CRAWLING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 9:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "IDLE_BOXING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 10:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "WALK_BOXING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 11:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "LEFT_PUNCH";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 12:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "RIGHT_PUNCH";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 13:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "RANDOM_PUNCH";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 14:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "ELBOW_CRAWLING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 15:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "LEFT_HOOK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 16:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "RIGHT_HOOK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 17:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "FORWARD_JUMP";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 18:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "STEALTH_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 19:
                if (config_.version == 1 || config_.version == 2)
                {
                    std::cout << "INJURED_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 20:
                if (config_.version == 2)
                {
                    std::cout << "LEDGE_WALKING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 21:
                if (config_.version == 2)
                {
                    std::cout << "OBJECT_CARRYING";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 22:
                if (config_.version == 2)
                {
                    std::cout << "STEALTH_WALK_2";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 23:
                if (config_.version == 2)
                {
                    std::cout << "HAPPY_DANCE_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 24:
                if (config_.version == 2)
                {
                    std::cout << "ZOMBIE_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 25:
                if (config_.version == 2)
                {
                    std::cout << "GUN_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            case 26:
                if (config_.version == 2)
                {
                    std::cout << "SCARE_WALK";
                }
                else
                {
                    std::cout << "UNKNOWN";
                }
                break;
            default:
                std::cout << "UNKNOWN";
                break;
        }
        if (config_.version == 1 || config_.version == 2)
        {
            std::cout << ", target_height: " << target_height_values_[0];
        }
        std::cout << ", target_vel: " << target_vel_values_[0]
                  << ", movement: [" << movement_direction_values_[0] << ", " << movement_direction_values_[1] << ", " << movement_direction_values_[2] << "]"
                  << ", facing: [" << facing_direction_values_[0] << ", " << facing_direction_values_[1] << ", " << facing_direction_values_[2] << "]" << std::endl;
    }

    virtual float *GetContextBuffer() override {
        return context_qpos_values_.data();
    }
    virtual int32_t GetNumPredFrames() override {
        return num_pred_frames_values_[0];
    }
    virtual const float *GetMujocoQposBuffer() override {
        return mujoco_qpos_values_.data();
    }
    virtual float *GetMovementDirectionValues() override {
        return movement_direction_values_.data();
    }
    virtual float *GetFacingDirectionValues() override {
        return facing_direction_values_.data();
    }
    virtual float GetTargetVelValue() override {
        return target_vel_values_[0];
    }
    virtual float GetHeightValue() override {
        return target_height_values_[0];
    }
    virtual int32_t GetRandomSeedValue() override {
        return random_seed_values_[0];
    }
    virtual int32_t GetModeValue() override {
        return mode_values_[0];
    }

};

#endif // LOCALMOTION_KPLANNER_TENSORRT_HPP
