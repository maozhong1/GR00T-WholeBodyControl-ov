# FindOpenVINO.cmake
# Locate Intel OpenVINO Runtime installation.
#
# This module looks for OpenVINO's native CMake config (OpenVINOConfig.cmake)
# in common installation paths. If OpenVINO_DIR is set (e.g., by setup_env.sh),
# it will be found directly.
#
# Sets:
#   OpenVINO_FOUND
#   OpenVINO_VERSION
#   openvino::runtime (imported target)

# Common OpenVINO installation paths
set(_OV_SEARCH_PATHS
    /opt/intel/openvino_2026/runtime/cmake
    /opt/intel/openvino/runtime/cmake
    /opt/intel/openvino_2025/runtime/cmake
    $ENV{OpenVINO_DIR}
    $ENV{INTEL_OPENVINO_DIR}/runtime/cmake
)

# Try to find the config file
find_package(OpenVINO QUIET CONFIG
    PATHS ${_OV_SEARCH_PATHS}
    NO_DEFAULT_PATH
)

if(NOT OpenVINO_FOUND)
    # Fallback: try system paths
    find_package(OpenVINO QUIET CONFIG)
endif()

if(OpenVINO_FOUND)
    message(STATUS "FindOpenVINO: Found OpenVINO ${OpenVINO_VERSION}")
else()
    if(OpenVINO_FIND_REQUIRED)
        message(FATAL_ERROR
            "OpenVINO not found. Please install OpenVINO 2026 and either:\n"
            "  1. Source /opt/intel/openvino_2026/setupvars.sh before building, or\n"
            "  2. Set OpenVINO_DIR to point to <openvino>/runtime/cmake/\n"
            "  Download: https://docs.openvino.ai/2026/get-started/install-openvino.html")
    endif()
endif()
