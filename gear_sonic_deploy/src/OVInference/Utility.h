#pragma once

/**
 * @file Utility.h
 * @brief OpenVINO-compatible replacement for TRTInference/Utility.h
 *
 * On OpenVINO builds, TPinnedVector<T> is just std::vector<T> since we don't
 * use CUDA pinned memory. This header exists so that #include "Utility.h"
 * resolves correctly when included from the OVInference directory.
 */

#include <vector>
#include <stdexcept>

// No CUDA pinned memory — use standard allocator
template<typename T>
using TPinnedVector = std::vector<T>;
