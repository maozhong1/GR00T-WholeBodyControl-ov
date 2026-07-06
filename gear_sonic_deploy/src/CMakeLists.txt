if(USE_OPENVINO)
  add_subdirectory(OVInference)
else()
  add_subdirectory(TRTInference)
endif()

add_subdirectory(audio_thread)
add_subdirectory(g1)
