#pragma once

/**
 * @file inference_backend.hpp
 * @brief Unified include for inference backend (TensorRT or OpenVINO).
 *
 * This header provides a single point to switch between TensorRT and OpenVINO
 * backends. All code should include this instead of directly including
 * TRTInference/InferenceEngine.h or OVInference/OVInferenceEngine.h.
 *
 * When USE_OPENVINO is defined:
 *   - TRTInferenceEngine is typedef'd to OVInferenceEngine
 *   - CUDA types (cudaStream_t, cudaGraph_t, etc.) become no-op stubs
 *   - TPinnedVector<T> becomes std::vector<T>
 *   - CUDA API calls become no-ops via macros
 */

#ifdef USE_OPENVINO

// ============================================================================
// OpenVINO Backend
// ============================================================================
#include <OVInference/OVInferenceEngine.h>

// CUDA type stubs (already defined in OVInferenceEngine.h via OV_CUDA_COMPAT_DEFINED)
using cudaGraph_t = void*;
using cudaGraphExec_t = void*;

// CUDA error type stubs
using cudaError_t = int;
constexpr int cudaSuccess = 0;

// CUDA stream capture modes (no-op)
enum cudaStreamCaptureMode { cudaStreamCaptureModeRelaxed = 0 };

// CUDA API no-op stubs
inline cudaError_t cudaStreamCreate(cudaStream_t*) { return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) { return cudaSuccess; }
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*) { return cudaSuccess; }
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t*, cudaGraph_t, void*, void*, unsigned long) { return cudaSuccess; }
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "no-op (OpenVINO mode)"; }

#else

// ============================================================================
// TensorRT Backend (original)
// ============================================================================
#include <cuda_runtime.h>
#include <TRTInference/InferenceEngine.h>

#endif // USE_OPENVINO
