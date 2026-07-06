cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(
  g1_deploy
  VERSION 1.0.0
  LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")

# Library target
add_library(${PROJECT_NAME} INTERFACE)

include(CheckLanguage)

# Optional builds
option(BUILD_SRCS "Build sources" ON)

# ============================================================================
# Inference Backend Selection
# ============================================================================
option(USE_OPENVINO "Use OpenVINO instead of TensorRT for inference" OFF)

# Auto-detect: if no CUDA/TensorRT but OpenVINO is available, default to OpenVINO
if(NOT USE_OPENVINO)
  # Check environment hint
  if(DEFINED ENV{USE_OPENVINO} AND "$ENV{USE_OPENVINO}" STREQUAL "1")
    set(USE_OPENVINO ON)
    message(STATUS "🔵 USE_OPENVINO enabled via environment variable")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(USE_OPENVINO)
  # ========================================================================
  # OpenVINO Backend (Intel x86 / iGPU / CPU)
  # ========================================================================
  message(STATUS "═══════════════════════════════════════════════════════════")
  message(STATUS "  Inference Backend: OpenVINO")
  message(STATUS "═══════════════════════════════════════════════════════════")

  find_package(OpenVINO REQUIRED COMPONENTS Runtime)
  message(STATUS "✅ OpenVINO found: ${OpenVINO_VERSION}")

  find_package(Threads REQUIRED)

  # ONNX Runtime is optional with OpenVINO (kept for reference motion playback)
  find_package(onnxruntime QUIET)
  if(onnxruntime_FOUND)
    message(STATUS "✅ ONNX Runtime found (optional, for reference motion)")
  else()
    message(STATUS "ℹ️  ONNX Runtime not found — reference motion via ORT disabled")
  endif()

  # No CUDA/TensorRT needed
  add_compile_definitions(USE_OPENVINO=1)
  add_compile_definitions(NO_CUDA=1)

else()
  # ========================================================================
  # TensorRT Backend (NVIDIA GPU — original path)
  # ========================================================================
  message(STATUS "═══════════════════════════════════════════════════════════")
  message(STATUS "  Inference Backend: TensorRT + CUDA")
  message(STATUS "═══════════════════════════════════════════════════════════")

  find_package(TensorRT REQUIRED)
  find_package(Threads REQUIRED)
  find_package(onnxruntime REQUIRED)

  # Find CUDA Toolkit
  if(DEFINED ENV{CUDAToolkit_ROOT})
      set(CUDAToolkit_ROOT $ENV{CUDAToolkit_ROOT})
      message(STATUS "🎯 Using CUDAToolkit_ROOT from environment: ${CUDAToolkit_ROOT}")
  endif()

  if(DEFINED ENV{CUDA_HOME})
      message(STATUS "🔍 Using CUDA_HOME from environment: $ENV{CUDA_HOME}")
      list(APPEND CMAKE_PREFIX_PATH $ENV{CUDA_HOME})
  endif()

  find_package(CUDAToolkit 10.2 QUIET)

  if(CUDAToolkit_FOUND)
      message(STATUS "✅ CUDA Toolkit found: version ${CUDAToolkit_VERSION}")
      message(STATUS "   CUDA Root: ${CUDAToolkit_TARGET_DIR}")
      message(STATUS "   CUDA Libraries: ${CUDAToolkit_LIBRARY_DIR}")

      if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.6")
          message(STATUS "   🚀 CUDA 12.6+ detected - enabling optimized features")
          add_compile_definitions(CUDA_12_6_PLUS=1)
      elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
          message(STATUS "   🔧 CUDA 12.x detected")
          add_compile_definitions(CUDA_12_PLUS=1)
      endif()
  else()
      # Fallback: search for CUDA runtime
      message(STATUS "🔍 CUDA Toolkit not found via find_package, searching for runtime...")

      file(GLOB CUDA_SBSA_PATHS "/usr/local/cuda*/targets/sbsa-linux/lib")
      file(GLOB CUDA_AARCH64_PATHS "/usr/local/cuda*/targets/aarch64-linux/lib")
      file(GLOB CUDA_LIB64_PATHS "/usr/local/cuda*/lib64")

      find_library(CUDA_RUNTIME_LIBRARY
          NAMES cudart libcudart
          HINTS
              $ENV{CUDAToolkit_ROOT}/targets/sbsa-linux/lib
              $ENV{CUDAToolkit_ROOT}/targets/aarch64-linux/lib
              $ENV{CUDAToolkit_ROOT}/lib64
              $ENV{CUDAToolkit_ROOT}/lib
              $ENV{CUDA_HOME}/lib64
              $ENV{CUDA_HOME}/lib
          PATHS
              ${CUDA_SBSA_PATHS}
              ${CUDA_AARCH64_PATHS}
              ${CUDA_LIB64_PATHS}
              /usr/local/cuda/lib64
              /usr/local/cuda/lib
              /usr/lib/aarch64-linux-gnu
              /usr/lib/x86_64-linux-gnu
              /usr/lib64
              /usr/lib
          NO_DEFAULT_PATH
      )

      find_path(CUDA_INCLUDE_DIR cuda_runtime.h
          HINTS
              $ENV{CUDAToolkit_ROOT}/include
              $ENV{CUDA_HOME}/include
          PATHS
              /usr/local/cuda/include
              /usr/include
          NO_DEFAULT_PATH
      )

      if(CUDA_RUNTIME_LIBRARY AND CUDA_INCLUDE_DIR)
          message(STATUS "✅ CUDA runtime found: ${CUDA_RUNTIME_LIBRARY}")
          add_library(CUDA::cudart SHARED IMPORTED)
          set_target_properties(CUDA::cudart PROPERTIES
              IMPORTED_LOCATION "${CUDA_RUNTIME_LIBRARY}"
              INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIR}")
      else()
          message(FATAL_ERROR "CUDA not found. Install CUDA or use -DUSE_OPENVINO=ON for Intel platforms.")
      endif()
  endif()

endif() # USE_OPENVINO

find_package(ZLIB REQUIRED)

# Use local unitree_sdk2 subfolder
set(BUILD_EXAMPLES OFF CACHE BOOL "Build unitree_sdk2 examples" FORCE)
add_subdirectory(thirdparty/unitree_sdk2)

# Add unitree_sdk2 include directories
target_include_directories(${PROJECT_NAME}
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/unitree_sdk2/include)
target_link_libraries(${PROJECT_NAME} INTERFACE unitree_sdk2)

# Examples
if(BUILD_SRCS)
  add_subdirectory(src)
endif()
