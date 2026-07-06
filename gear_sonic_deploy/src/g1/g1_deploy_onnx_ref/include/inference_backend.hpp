#pragma once

/**
 * @file inference_backend.hpp
 * @brief Unified include for the OpenVINO inference backend.
 *
 * All inference code should include this header. It provides:
 *   - OVInferenceEngine (aliased as TRTInferenceEngine for compatibility)
 *   - TPinnedVector<T> as std::vector<T>
 *   - ConvertONNXToTRT() as a no-op passthrough (OpenVINO reads ONNX directly)
 */

#include <OVInference/OVInferenceEngine.h>
