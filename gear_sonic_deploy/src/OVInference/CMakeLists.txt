set(SRC_FILES
    "OVInferenceEngine.cpp"
    "OVInferenceEngine.h"
    "OVInferenceEngine.inl"
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_FILES})

add_library(OVInference ${SRC_FILES})

set_target_properties(OVInference PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Find OpenVINO
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

target_include_directories(OVInference
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(OVInference PUBLIC openvino::runtime)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(OVInference PRIVATE -Wno-deprecated-declarations)
endif()
