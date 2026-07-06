# freq_test requires ONNX Runtime — skip if not available
if(onnxruntime_FOUND)
  add_subdirectory(freq_test)
else()
  message(STATUS "ℹ️  Skipping freq_test (ONNX Runtime not found)")
endif()

add_subdirectory(g1_deploy_onnx_ref)
