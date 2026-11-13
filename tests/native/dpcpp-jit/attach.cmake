# CMAKE_PROJECT_LLVM_INCLUDE attaches the proof after upstream defines its complete target graph.
# Defer instead of copying its Clang/LLVM component list or importing cyclic package exports.
if(NOT IS_ABSOLUTE "${MOM_SOURCE_REPO}")
  message(FATAL_ERROR "MOM_SOURCE_REPO must select the absolute proof repository")
endif()
cmake_language(DEFER CALL include "${MOM_SOURCE_REPO}/tests/native/dpcpp-jit/CMakeLists.txt")
