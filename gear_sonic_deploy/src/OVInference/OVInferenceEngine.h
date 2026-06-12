#pragma once

/**
 * @file OVInferenceEngine.h
 * @brief OpenVINO-based inference engine — drop-in replacement for TRTInferenceEngine.
 *
 * Provides the same public interface (SetInputData / GetOutputData / Enqueue) so that
 * higher-level code (PolicyEngine, Encoder, Planner) can switch backends via a typedef
 * or compile-time flag without source changes.
 *
 * Default device priority: Intel GPU → CPU (configurable via Initialize()).
 */

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <cstring>
#include <stdexcept>

// --------------------------------------------------------------------------
// Shared enums (same as TRTInference for API compatibility)
// --------------------------------------------------------------------------
#ifndef INFERENCE_ENUMS_DEFINED
#define INFERENCE_ENUMS_DEFINED

enum class Precision {
    FP32,
    FP16
};

enum class DataType {
    FLOAT,
    HALF,
    INT8,
    INT32,
    BOOL,
    UINT8,
    INT64,
    UNKNOWN
};

#endif // INFERENCE_ENUMS_DEFINED

// --------------------------------------------------------------------------
// Options struct (compatible with TRT Options, only relevant fields used)
// --------------------------------------------------------------------------
struct Options {
    using AxisNames = std::map<std::string, std::map<int, std::string>>;
    using AxisSizes = std::map<std::string, std::tuple<int, int, int>>;
    using ShapeTensorSizes = std::map<std::string, std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>>;

    Precision        precision = Precision::FP32;
    AxisNames        dynamic_axes_names;
    AxisSizes        dynamic_axes_sizes;
    ShapeTensorSizes shape_tensor_sizes;
    std::tuple<int, int, int> defaultSizes = {1, 8, 16};
    int              deviceID = 0;

    // OpenVINO-specific: preferred device ("NPU", "GPU", "CPU", "AUTO:NPU,CPU", etc.)
    std::string ov_device = "NPU";
    // Fallback device when preferred is unavailable
    std::string ov_fallback_device = "CPU";
};

// --------------------------------------------------------------------------
// Stub: ConvertONNXToTRT is a no-op for OpenVINO (ONNX loaded directly)
// Returns the same path as "generated file" since OV reads ONNX natively.
// --------------------------------------------------------------------------
inline bool ConvertONNXToTRT(
    const Options& /*options*/,
    const std::string& onnxModelPath,
    std::string& generatedFile,
    const std::string /*prefix*/ = "",
    bool /*forceConvert*/ = false)
{
    // OpenVINO reads ONNX directly — no conversion step needed.
    generatedFile = onnxModelPath;
    return true;
}

// --------------------------------------------------------------------------
// Pinned-memory vector replacement: on CPU/iGPU we use standard allocator.
// The typedef is kept so existing code using TPinnedVector<T> compiles as-is.
// --------------------------------------------------------------------------
template<typename T>
using TPinnedVector = std::vector<T>;

// --------------------------------------------------------------------------
// Dummy cudaStream_t for API compatibility (OpenVINO doesn't use CUDA streams)
// --------------------------------------------------------------------------
#ifndef OV_CUDA_COMPAT_DEFINED
#define OV_CUDA_COMPAT_DEFINED
using cudaStream_t = void*;
#endif

// --------------------------------------------------------------------------
// OVInferenceEngine
// --------------------------------------------------------------------------
class OVInferenceEngine {
public:
    OVInferenceEngine();
    ~OVInferenceEngine();

    using AxisSizes = std::map<std::string, int>;

    /**
     * @brief Load and compile an ONNX model with OpenVINO.
     * @param modelPath Path to ONNX file (or OpenVINO IR .xml)
     * @param deviceID Ignored (kept for API compat); use Options::ov_device instead.
     * @param axisNames Dynamic axis metadata (used for reshape if needed)
     * @return true on success
     */
    bool Initialize(const std::string& modelPath, int deviceID = 0,
                    const Options::AxisNames& axisNames = {});

    /**
     * @brief Extended initialize with device selection.
     * @param modelPath Path to ONNX file
     * @param device OpenVINO device string ("GPU", "CPU", "AUTO:GPU,CPU")
     * @param precision Inference precision hint
     * @param npu_tiles Number of NPU tiles to use (0 = auto/default, 1-N = specific count)
     * @param model_priority OpenVINO model priority hint ("HIGH", "NORMAL", "LOW").
     *                       Maps to ov::hint::model_priority in compile_model config.
     * @return true on success
     */
    bool Initialize(const std::string& modelPath, const std::string& device,
                    Precision precision = Precision::FP32, int npu_tiles = 0,
                    const std::string& model_priority = "NORMAL");

    bool InitInputs(const AxisSizes& axisSizes = {});
    void Destroy();

    // Synchronous data transfer (memcpy into internal buffers)
    void SetInputData(const std::string& name, const void* data, size_t byteCount);
    template<typename T> void SetInputData(const std::string& name, const T* data, size_t elementCount);
    template<typename T> void SetInputData(const std::string& name, const std::vector<T>& data);

    void GetOutputData(const std::string& name, void* data, size_t byteCount);
    template<typename T> void GetOutputData(const std::string& name, T* data, size_t elementCount);
    template<typename T> void GetOutputData(const std::string& name, std::vector<T>& data);

    // Async variants — for OpenVINO these are identical to sync (stream param ignored)
    void SetInputDataAsync(const std::string& name, const void* data, size_t byteCount, cudaStream_t);
    template<typename T> void SetInputDataAsync(const std::string& name, const T* data, size_t elementCount, cudaStream_t);
    template<typename T> void SetInputDataAsync(const std::string& name, const std::vector<T>& data, cudaStream_t);

    void GetOutputDataAsync(const std::string& name, void* data, size_t byteCount, cudaStream_t);
    template<typename T> void GetOutputDataAsync(const std::string& name, T* data, size_t elementCount, cudaStream_t);
    template<typename T> void GetOutputDataAsync(const std::string& name, std::vector<T>& data, cudaStream_t);

    std::vector<std::string> GetInputTensorNames() const;
    std::vector<std::string> GetOutputTensorNames() const;

    bool GetTensorShape(std::string name, std::vector<int64_t>& shape) const;
    DataType GetTensorDataType(std::string name) const;

    /**
     * @brief Get direct pointer to input tensor's memory buffer (zero-copy).
     *
     * Write observation data directly into this buffer to avoid the extra
     * memcpy in SetInputData(). The pointer remains valid for the lifetime
     * of the engine (OV tensors are pre-allocated at compile time).
     *
     * @param name Input tensor name
     * @return Raw pointer to the tensor data, or nullptr if not found
     */
    void* GetInputTensorBuffer(const std::string& name);

    /**
     * @brief Get direct pointer to output tensor's memory buffer (zero-copy).
     *
     * Read action data directly from this buffer to avoid the extra
     * memcpy in GetOutputData().
     *
     * @param name Output tensor name
     * @return Raw pointer to the tensor data, or nullptr if not found
     */
    const void* GetOutputTensorBuffer(const std::string& name) const;

    /**
     * @brief Run inference (stream param ignored for OpenVINO).
     */
    bool Enqueue(cudaStream_t stream = nullptr);

    /**
     * @brief Get the actual device the model was compiled on.
     */
    std::string GetDevice() const;

private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

// --------------------------------------------------------------------------
// Typedef for drop-in replacement
// --------------------------------------------------------------------------
using TRTInferenceEngine = OVInferenceEngine;

#include "OVInferenceEngine.inl"
