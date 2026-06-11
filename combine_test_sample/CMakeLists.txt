cmake_minimum_required(VERSION 3.16)
project(combine_test_sample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED OpenVINO_DIR)
    if(EXISTS "/opt/intel/openvino_2026/runtime/cmake")
        set(OpenVINO_DIR "/opt/intel/openvino_2026/runtime/cmake")
    elseif(EXISTS "/opt/intel/openvino_2016/runtime/cmake")
        set(OpenVINO_DIR "/opt/intel/openvino_2016/runtime/cmake")
    endif()
endif()
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(combine_test_sample main.cpp)

target_link_libraries(combine_test_sample PRIVATE
    openvino::runtime
    ${OpenCV_LIBS}
    Threads::Threads
)
