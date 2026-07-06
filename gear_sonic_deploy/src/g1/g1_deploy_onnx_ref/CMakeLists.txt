set(EXECUTABLE_NAME g1_deploy_onnx_ref)

# ROS2 discovery can be controlled via env var HAS_ROS2:
# - 1: try ROS2 (find_package QUIET)
# - 0 or unset: disable ROS2 entirely (skip find_package)
set(ROS2_ENV_REQUEST "off")
if(DEFINED ENV{HAS_ROS2})
  if("$ENV{HAS_ROS2}" STREQUAL "1")
    set(ROS2_ENV_REQUEST "on")
  elseif("$ENV{HAS_ROS2}" STREQUAL "0")
    set(ROS2_ENV_REQUEST "off")
  else()
    message(WARNING "HAS_ROS2 should be 0 or 1; got '$ENV{HAS_ROS2}'. Falling back to auto-detect.")
  endif()
endif()

if(ROS2_ENV_REQUEST STREQUAL "off")
  set(rclcpp_FOUND FALSE)
  set(std_msgs_FOUND FALSE)
  message(STATUS "ROS2 disabled (HAS_ROS2 unset or 0)")
else()
  # Try to locate ROS2 only when explicitly enabled
  find_package(rclcpp QUIET)
  find_package(std_msgs QUIET)
endif()

# Find msgpack for message serialization (used by ROS2 and ZMQ handlers)
find_path(MSGPACK_INCLUDE_DIR msgpack.hpp
  PATHS /usr/include /usr/local/include
  DOC "msgpack C++ header location"
)

if(MSGPACK_INCLUDE_DIR)
  message(STATUS "✅ msgpack found at: ${MSGPACK_INCLUDE_DIR}")
  set(MSGPACK_INCLUDE_DIRS ${MSGPACK_INCLUDE_DIR})
else()
  # Fetch msgpack-cxx as header-only dependency
  message(STATUS "📦 msgpack not found in system — fetching via FetchContent...")
  include(FetchContent)
  FetchContent_Declare(
    msgpack-cxx
    GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
    GIT_TAG cpp-6.1.1
    GIT_SHALLOW TRUE
  )
  set(MSGPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MSGPACK_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(msgpack-cxx)
  set(MSGPACK_INCLUDE_DIRS ${msgpack-cxx_SOURCE_DIR}/include)
  message(STATUS "✅ msgpack fetched to: ${MSGPACK_INCLUDE_DIRS}")
endif()

file(GLOB_RECURSE PROJECT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM PROJECT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_deploy_onnx_ref.cpp)


# Optional AddressSanitizer support
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
  message(STATUS "🔬 Building ${EXECUTABLE_NAME} with AddressSanitizer")
  target_compile_options(${EXECUTABLE_NAME} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(${EXECUTABLE_NAME} PRIVATE -fsanitize=address)
endif()

# ============================================================================
# Platform-specific setup
# ============================================================================
if(USE_OPENVINO)
  # OpenVINO path: no CUDA link dirs needed
  set(CUDA_LINK_DIRS "")
  set(CUDLA_LIBRARY "")
  message(STATUS "✅ x86_64 + OpenVINO mode — no CUDA dependencies")

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  # Check for Thor (no DLA support)
  set(IS_THOR FALSE)
  if(DEFINED ENV{IS_THOR} AND "$ENV{IS_THOR}" STREQUAL "true")
    set(IS_THOR TRUE)
  elseif(EXISTS "/proc/device-tree/model")
    file(READ "/proc/device-tree/model" DEVICE_MODEL)
    string(FIND "${DEVICE_MODEL}" "Thor" THOR_POS)
    if(NOT THOR_POS EQUAL -1)
      set(IS_THOR TRUE)
    endif()
  endif()

  # ARM/Jetson platform - conditionally add DLA support
  if(DEFINED ENV{CUDAToolkit_ROOT})
    set(CUDA_ARM_LIB_PATH "$ENV{CUDAToolkit_ROOT}/targets/aarch64-linux/lib")
  elseif(EXISTS "/usr/local/cuda/targets/aarch64-linux/lib")
    set(CUDA_ARM_LIB_PATH "/usr/local/cuda/targets/aarch64-linux/lib")
  else()
    file(GLOB CUDA_ARM_DIRS "/usr/local/cuda-*/targets/aarch64-linux/lib")
    if(CUDA_ARM_DIRS)
      list(GET CUDA_ARM_DIRS 0 CUDA_ARM_LIB_PATH)
    else()
      message(WARNING "No CUDA ARM libraries found, using fallback path")
      set(CUDA_ARM_LIB_PATH "/usr/local/cuda/lib64")
    endif()
  endif()

  set(CUDA_LINK_DIRS ${CUDA_ARM_LIB_PATH})

  if(IS_THOR)
    set(CUDLA_LIBRARY "")
    message(STATUS "🚀 ARM64 (Thor) detected - building without DLA support")
  else()
    set(CUDLA_LIBRARY "cudla")
    message(STATUS "✅ ARM64 detected - enabling DLA (cudla) support")
  endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
  # x86_64 with TensorRT (NVIDIA dGPU)
  if(DEFINED ENV{CUDAToolkit_ROOT})
    set(CUDA_X64_LIB_PATH "$ENV{CUDAToolkit_ROOT}/lib64")
  elseif(EXISTS "/usr/local/cuda/lib64")
    set(CUDA_X64_LIB_PATH "/usr/local/cuda/lib64")
  else()
    file(GLOB CUDA_X64_DIRS "/usr/local/cuda-*/lib64")
    if(CUDA_X64_DIRS)
      list(GET CUDA_X64_DIRS 0 CUDA_X64_LIB_PATH)
    else()
      message(WARNING "No CUDA x86_64 libraries found, using fallback path")
      set(CUDA_X64_LIB_PATH "/usr/local/cuda/lib64")
    endif()
  endif()

  set(CUDA_LINK_DIRS ${CUDA_X64_LIB_PATH})
  set(CUDLA_LIBRARY "")
  message(STATUS "✅ x86_64 + TensorRT mode")
endif()

# Find nlohmann/json.hpp (needed by ZMQ tests and main executable)
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
  PATHS /usr/include /usr/local/include
        /usr/share/openvino/samples/cpp/thirdparty/nlohmann_json/single_include
  DOC "nlohmann JSON header location"
)
if(NLOHMANN_JSON_INCLUDE_DIR)
  message(STATUS "✅ nlohmann/json found at: ${NLOHMANN_JSON_INCLUDE_DIR}")
else()
  # Fetch it
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
  )
  set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nlohmann_json)
  set(NLOHMANN_JSON_INCLUDE_DIR ${nlohmann_json_SOURCE_DIR}/single_include)
  message(STATUS "✅ nlohmann/json fetched to: ${NLOHMANN_JSON_INCLUDE_DIR}")
endif()

# ZMQ support (required for main executable and tests)
find_package(PkgConfig QUIET)
if(PkgConfig)
  pkg_check_modules(ZMQ QUIET libzmq)
endif()

find_path(ZMQ_INCLUDE_DIR zmq.h
  PATHS /usr/include /usr/local/include)
find_library(ZMQ_LIBRARY zmq
  PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)

if(NOT ZMQ_INCLUDE_DIR OR NOT ZMQ_LIBRARY)
  message(FATAL_ERROR "❌ libzmq is required but not found. Please install libzmq-dev (apt) or zeromq (brew)")
endif()

message(STATUS "✅ ZMQ support enabled for main executable")

# Build ZMQ tests
if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
  add_executable(zmq_pose_subscriber_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/zmq_pose_subscriber_test.cpp)
  target_include_directories(zmq_pose_subscriber_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ZMQ_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR})
  target_link_libraries(zmq_pose_subscriber_test PRIVATE ${ZMQ_LIBRARY})
  set_target_properties(zmq_pose_subscriber_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  message(STATUS "✅ Added zmq_pose_subscriber_test (ZMQ found)")

  add_executable(zmq_python_sender_test
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/zmq_python_sender_test.cpp)
  target_include_directories(zmq_python_sender_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ZMQ_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR})
  target_link_libraries(zmq_python_sender_test PRIVATE ${ZMQ_LIBRARY})
  set_target_properties(zmq_python_sender_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  message(STATUS "✅ Added zmq_python_sender_test (ZMQ found)")
else()
  message(WARNING "⚠️  libzmq not found - skipping zmq_pose_subscriber_test")
endif()

find_package(Eigen3 REQUIRED)

set(EXE_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
  ${ZMQ_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${NLOHMANN_JSON_INCLUDE_DIR}
  ${MSGPACK_INCLUDE_DIRS}
)


# Build a static library, used by both the main executable and the tests:
add_library(g1_deploy_shared STATIC ${PROJECT_SRCS})
target_include_directories(g1_deploy_shared PUBLIC ${EXE_INCLUDES})

# ============================================================================
# Link libraries — backend-specific
# ============================================================================
if(USE_OPENVINO)
  set(INFERENCE_LIB OVInference)
  set(EXE_LIBS
    g1_deploy_shared
    audio_thread
    unitree_sdk2 ZLIB::ZLIB
    ${INFERENCE_LIB}
    ${ZMQ_LIBRARY}
  )
  # Optionally link ONNX Runtime if found (for reference motion ORT path)
  if(onnxruntime_FOUND)
    list(APPEND EXE_LIBS onnxruntime::onnxruntime)
  endif()
else()
  set(INFERENCE_LIB TRTInference)
  set(EXE_LIBS
    g1_deploy_shared
    audio_thread
    unitree_sdk2 ZLIB::ZLIB
    onnxruntime::onnxruntime ${INFERENCE_LIB}
    CUDA::cudart
    ${TensorRT_nvinfer_LIBRARY} ${TensorRT_nvonnxparser_LIBRARY}
    ${CUDLA_LIBRARY}
    ${ZMQ_LIBRARY}
  )
endif()

# Add compile definition to g1_deploy_shared so headers can switch
if(USE_OPENVINO)
  target_compile_definitions(g1_deploy_shared PUBLIC USE_OPENVINO=1 NO_CUDA=1)
endif()

# build the main executable:
add_executable(${EXECUTABLE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/g1_deploy_onnx_ref.cpp)
target_include_directories(${EXECUTABLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/ ${PROJECT_SOURCE_DIR}/src)

if(CUDA_LINK_DIRS)
  target_link_directories(${EXECUTABLE_NAME} PRIVATE ${CUDA_LINK_DIRS})
endif()

target_link_libraries(
  ${EXECUTABLE_NAME} PRIVATE ${PROJECT_NAME}
  ${EXE_LIBS}
)

# Setting our output directory and C++ standard
set_target_properties(
  ${EXECUTABLE_NAME}
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
             OUTPUT_NAME ${EXECUTABLE_NAME}
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF
             BUILD_RPATH_USE_ORIGIN ON)

# Force RPATH (not RUNPATH) so unitree's DDS libs take priority over LD_LIBRARY_PATH.
# RUNPATH is searched AFTER LD_LIBRARY_PATH, causing ROS2's incompatible libddsc to be
# loaded instead of unitree's bundled version. RPATH is searched BEFORE LD_LIBRARY_PATH.
target_link_options(${EXECUTABLE_NAME} PRIVATE "LINKER:--disable-new-dtags")

# Optional ROS2 support for main executable and tests
if(rclcpp_FOUND AND std_msgs_FOUND)
  # Load ROS2 configuration (handles both main executable and tests)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS2.cmake)

  # Add ROS2 support to main executable (requires msgpack)
  target_include_directories(${EXECUTABLE_NAME} PRIVATE ${ROS2_INCLUDE_DIRS} ${MSGPACK_INCLUDE_DIRS})
  target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${ROS2_LIBS} pthread)
  target_compile_definitions(${EXECUTABLE_NAME} PRIVATE HAS_ROS2=1)

  message(STATUS "✅ ROS2InputHandler support enabled with msgpack")
else()
  target_compile_definitions(${EXECUTABLE_NAME} PRIVATE HAS_ROS2=0)
  message(STATUS "ROS2 not found - building without ROS2InputHandler support")
endif()

# test executable:
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

find_package(GTest REQUIRED)

file(GLOB_RECURSE TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/*.cpp)

add_executable(run_tests ${TEST_SRCS})
if(CUDA_LINK_DIRS)
  target_link_directories(run_tests PRIVATE ${CUDA_LINK_DIRS})
endif()
target_link_libraries(run_tests GTest::GTest GTest::Main ${EXE_LIBS})
target_include_directories(run_tests PUBLIC ${EXE_INCLUDES})

set_target_properties(
  run_tests
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/target/release/"
             OUTPUT_NAME run_tests
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)
