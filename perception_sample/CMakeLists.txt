cmake_minimum_required(VERSION 3.16)
project(perception_sample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenVINO
set(OpenVINO_DIR "/opt/intel/openvino_2016/runtime/cmake")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# OpenCV (typically bundled with OpenVINO or installed separately)
find_package(OpenCV REQUIRED)

add_executable(perception_sample main.cpp)

target_link_libraries(perception_sample PRIVATE
    openvino::runtime
    ${OpenCV_LIBS}
)
