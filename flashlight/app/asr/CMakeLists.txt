cmake_minimum_required(VERSION 3.10)

cmake_dependent_option(FL_BUILD_APP_ASR_TOOLS "Build ASR App tools" ON "FL_BUILD_APP_ASR" OFF)

# ----------------------------- Binaries -----------------------------
add_executable(fl_asr_train ${CMAKE_CURRENT_LIST_DIR}/Train.cpp)
add_executable(fl_asr_test ${CMAKE_CURRENT_LIST_DIR}/Test.cpp)
add_executable(fl_asr_decode ${CMAKE_CURRENT_LIST_DIR}/Decode.cpp)

target_link_libraries(
  fl_asr_train
  fl_pkg_speech
  fl_pkg_runtime
  fl_lib_sequence
  ${CMAKE_DL_LIBS}
  )
target_link_libraries(
  fl_asr_test
  fl_pkg_speech
  fl_pkg_runtime
  ${CMAKE_DL_LIBS}
  )
target_link_libraries(
  fl_asr_decode
  fl_pkg_speech
  fl_pkg_runtime
  ${CMAKE_DL_LIBS}
  )

set_executable_output_directory(fl_asr_train "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")
set_executable_output_directory(fl_asr_test "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")
set_executable_output_directory(fl_asr_decode "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")

install(TARGETS fl_asr_train RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
install(TARGETS fl_asr_test RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
install(TARGETS fl_asr_decode RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})

# ----------------------------- Tutorial Binaries -----------------------------

add_executable(fl_asr_tutorial_inference_ctc ${CMAKE_CURRENT_LIST_DIR}/tutorial/InferenceCTC.cpp)
add_executable(fl_asr_tutorial_finetune_ctc ${CMAKE_CURRENT_LIST_DIR}/tutorial/FinetuneCTC.cpp)
add_executable(fl_asr_inference_ctc ${CMAKE_CURRENT_LIST_DIR}/prod/InferenceCTC.cpp)

target_link_libraries(fl_asr_tutorial_inference_ctc flashlight-app-asr ${CMAKE_DL_LIBS})
target_link_libraries(fl_asr_tutorial_finetune_ctc flashlight-app-asr ${CMAKE_DL_LIBS})
target_link_libraries(fl_asr_inference_ctc flashlight-app-asr ${CMAKE_DL_LIBS} jsonbuilder)
add_executable(fl_asr_transferlearn_ctc ${CMAKE_CURRENT_LIST_DIR}/prod/TransferLearnCTC.cpp)

target_link_libraries(
  fl_asr_tutorial_inference_ctc
  fl_pkg_speech
  fl_pkg_runtime
  ${CMAKE_DL_LIBS})

target_link_libraries(
  fl_asr_tutorial_finetune_ctc
  fl_pkg_speech
  fl_pkg_runtime
  ${CMAKE_DL_LIBS})

target_link_libraries(
  fl_asr_inference_ctc
  fl_pkg_speech
  fl_pkg_runtime
  ${CMAKE_DL_LIBS})

target_link_libraries(
  fl_asr_transferlearn_ctc
  fl_pkg_speech
  fl_pkg_runtime
  ${CMAKE_DL_LIBS})

    
set_executable_output_directory(fl_asr_tutorial_inference_ctc "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")
set_executable_output_directory(fl_asr_tutorial_finetune_ctc "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")
set_executable_output_directory(fl_asr_inference_ctc "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")
set_executable_output_directory(fl_asr_transferlearn_ctc "${FL_BUILD_BINARY_OUTPUT_DIR}/asr")

# --------------------------- Tools ---------------------------
if (FL_BUILD_APP_ASR_TOOLS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools)
endif()
