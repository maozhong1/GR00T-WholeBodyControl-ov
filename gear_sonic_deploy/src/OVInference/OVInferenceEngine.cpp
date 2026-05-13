/**
 * @file OVInferenceEngine.cpp
 * @brief OpenVINO inference engine implementation.
 *
 * Loads ONNX models directly via OpenVINO 2026 Runtime API.
 * Default device: NPU (Intel integrated/discrete), fallback: CPU.
 */

#include "OVInferenceEngine.h"

#include <openvino/openvino.hpp>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <filesystem>
#include <cstdlib>

// ============================================================================
// Helper: Determine model cache directory
// ============================================================================
// Priority: $OV_CACHE_DIR > $HOME/.cache/openvino_model_cache > /tmp/openvino_model_cache
static std::string GetCacheDir() {
    // Check environment variable override
    const char* env_cache = std::getenv("OV_CACHE_DIR");
    if (env_cache && env_cache[0] != '\0') {
        std::filesystem::create_directories(env_cache);
        return env_cache;
    }

    // Default: ~/.cache/openvino_model_cache
    const char* home = std::getenv("HOME");
    std::string cache_dir;
    if (home && home[0] != '\0') {
        cache_dir = std::string(home) + "/.cache/openvino_model_cache";
    } else {
        cache_dir = "/tmp/openvino_model_cache";
    }

    std::filesystem::create_directories(cache_dir);
    return cache_dir;
}

// ============================================================================
// Implementation (pimpl)
// ============================================================================
class OVInferenceEngine::Impl {
public:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    std::string device_used;
    bool initialized = false;

    // Cached tensor metadata
    struct TensorMeta {
        ov::element::Type element_type;
        ov::Shape shape;
        size_t byte_size;
    };
    std::unordered_map<std::string, TensorMeta> input_meta;
    std::unordered_map<std::string, TensorMeta> output_meta;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

// ============================================================================
// Helper: Convert ov::element::Type to our DataType enum
// ============================================================================
static DataType OVTypeToDataType(ov::element::Type t) {
    if (t == ov::element::f32)   return DataType::FLOAT;
    if (t == ov::element::f16)   return DataType::HALF;
    if (t == ov::element::i8)    return DataType::INT8;
    if (t == ov::element::i32)   return DataType::INT32;
    if (t == ov::element::boolean) return DataType::BOOL;
    if (t == ov::element::u8)    return DataType::UINT8;
    if (t == ov::element::i64)   return DataType::INT64;
    return DataType::UNKNOWN;
}

// ============================================================================
// Construction / Destruction
// ============================================================================
OVInferenceEngine::OVInferenceEngine() : m_impl(std::make_shared<Impl>()) {}

OVInferenceEngine::~OVInferenceEngine() { Destroy(); }

// ============================================================================
// Initialize (TRT-compatible signature)
// ============================================================================
bool OVInferenceEngine::Initialize(const std::string& modelPath, int /*deviceID*/,
                                   const Options::AxisNames& /*axisNames*/) {
    // Use AUTO:NPU,CPU as default — OpenVINO will pick best available
    return Initialize(modelPath, "AUTO:GPU,CPU", Precision::FP32);
}

// ============================================================================
// Initialize (extended, with device selection)
// ============================================================================
bool OVInferenceEngine::Initialize(const std::string& modelPath, const std::string& device,
                                   Precision precision) {
    try {
        std::cout << "[OVInference] Loading model: " << modelPath << std::endl;
        std::cout << "[OVInference] Requested device: " << device << std::endl;

        // Print available devices
        auto devices = m_impl->core.get_available_devices();
        std::cout << "[OVInference] Available devices:";
        for (const auto& d : devices) std::cout << " " << d;
        std::cout << std::endl;

        // Enable model caching for faster subsequent loads (GPU/NPU compilation is cached on disk)
        std::string cache_dir = GetCacheDir();
        if (!cache_dir.empty()) {
            m_impl->core.set_property(ov::cache_dir(cache_dir));
            std::cout << "[OVInference] Model cache dir: " << cache_dir << std::endl;
        }

        // Read model (supports ONNX natively)
        m_impl->model = m_impl->core.read_model(modelPath);

        // Set precision hint
        ov::AnyMap config;
        if (precision == Precision::FP16) {
            config[ov::hint::inference_precision.name()] = ov::element::f16;
        } else {
            config[ov::hint::inference_precision.name()] = ov::element::f32;
        }
        // Latency-optimized for real-time control
        config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;

        // Compile model (uses cache if available, otherwise compiles and stores to cache)
        m_impl->compiled_model = m_impl->core.compile_model(m_impl->model, device, config);
        m_impl->device_used = device;

        // Create inference request
        m_impl->infer_request = m_impl->compiled_model.create_infer_request();

        // Cache input metadata
        for (const auto& input : m_impl->model->inputs()) {
            std::string name = input.get_any_name();
            auto shape = input.get_shape();
            auto type = input.get_element_type();
            size_t byte_size = ov::shape_size(shape) * type.size();

            m_impl->input_meta[name] = {type, shape, byte_size};
            m_impl->input_names.push_back(name);
        }

        // Cache output metadata
        for (const auto& output : m_impl->model->outputs()) {
            std::string name = output.get_any_name();
            auto shape = output.get_shape();
            auto type = output.get_element_type();
            size_t byte_size = ov::shape_size(shape) * type.size();

            m_impl->output_meta[name] = {type, shape, byte_size};
            m_impl->output_names.push_back(name);
        }

        m_impl->initialized = true;

        std::cout << "[OVInference] ✓ Model compiled on device: " << device << std::endl;
        std::cout << "[OVInference]   Inputs: " << m_impl->input_names.size()
                  << ", Outputs: " << m_impl->output_names.size() << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[OVInference] ✗ Failed to initialize: " << e.what() << std::endl;
        m_impl->initialized = false;
        return false;
    }
}

// ============================================================================
// InitInputs — for dynamic shapes (reshaping). Static shapes: no-op.
// ============================================================================
bool OVInferenceEngine::InitInputs(const AxisSizes& /*axisSizes*/) {
    // For static ONNX models used in this pipeline, no reshape is needed.
    // If dynamic shapes are required in the future, implement reshape here.
    return m_impl->initialized;
}

// ============================================================================
// Destroy
// ============================================================================
void OVInferenceEngine::Destroy() {
    if (m_impl) {
        m_impl->initialized = false;
        m_impl->input_meta.clear();
        m_impl->output_meta.clear();
        m_impl->input_names.clear();
        m_impl->output_names.clear();
    }
}

// ============================================================================
// SetInputData (sync)
// ============================================================================
void OVInferenceEngine::SetInputData(const std::string& name, const void* data, size_t byteCount) {
    if (!m_impl->initialized) return;

    auto it = m_impl->input_meta.find(name);
    if (it == m_impl->input_meta.end()) {
        std::cerr << "[OVInference] ✗ Unknown input tensor: " << name << std::endl;
        return;
    }

    // Get the input tensor from the infer request and copy data into it
    ov::Tensor tensor = m_impl->infer_request.get_tensor(name);
    size_t copy_size = std::min(byteCount, it->second.byte_size);
    std::memcpy(tensor.data(), data, copy_size);
}

// ============================================================================
// GetOutputData (sync)
// ============================================================================
void OVInferenceEngine::GetOutputData(const std::string& name, void* data, size_t byteCount) {
    if (!m_impl->initialized) return;

    auto it = m_impl->output_meta.find(name);
    if (it == m_impl->output_meta.end()) {
        std::cerr << "[OVInference] ✗ Unknown output tensor: " << name << std::endl;
        return;
    }

    ov::Tensor tensor = m_impl->infer_request.get_tensor(name);
    size_t copy_size = std::min(byteCount, it->second.byte_size);
    std::memcpy(data, tensor.data(), copy_size);
}

// ============================================================================
// Async variants (stream parameter ignored — OpenVINO manages its own queues)
// ============================================================================
void OVInferenceEngine::SetInputDataAsync(const std::string& name, const void* data, size_t byteCount, cudaStream_t) {
    SetInputData(name, data, byteCount);
}

void OVInferenceEngine::GetOutputDataAsync(const std::string& name, void* data, size_t byteCount, cudaStream_t) {
    GetOutputData(name, data, byteCount);
}

// ============================================================================
// Enqueue (run inference)
// ============================================================================
bool OVInferenceEngine::Enqueue(cudaStream_t /*stream*/) {
    if (!m_impl->initialized) return false;

    try {
        m_impl->infer_request.infer();  // Synchronous inference
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[OVInference] ✗ Inference failed: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Metadata queries
// ============================================================================
std::vector<std::string> OVInferenceEngine::GetInputTensorNames() const {
    return m_impl->input_names;
}

std::vector<std::string> OVInferenceEngine::GetOutputTensorNames() const {
    return m_impl->output_names;
}

bool OVInferenceEngine::GetTensorShape(std::string name, std::vector<int64_t>& shape) const {
    // Check inputs
    auto it = m_impl->input_meta.find(name);
    if (it != m_impl->input_meta.end()) {
        shape.clear();
        for (auto dim : it->second.shape) shape.push_back(static_cast<int64_t>(dim));
        return true;
    }
    // Check outputs
    auto oit = m_impl->output_meta.find(name);
    if (oit != m_impl->output_meta.end()) {
        shape.clear();
        for (auto dim : oit->second.shape) shape.push_back(static_cast<int64_t>(dim));
        return true;
    }
    return false;
}

DataType OVInferenceEngine::GetTensorDataType(std::string name) const {
    auto it = m_impl->input_meta.find(name);
    if (it != m_impl->input_meta.end()) return OVTypeToDataType(it->second.element_type);

    auto oit = m_impl->output_meta.find(name);
    if (oit != m_impl->output_meta.end()) return OVTypeToDataType(oit->second.element_type);

    return DataType::UNKNOWN;
}

std::string OVInferenceEngine::GetDevice() const {
    return m_impl->device_used;
}
