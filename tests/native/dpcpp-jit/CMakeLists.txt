cmake_minimum_required(VERSION 3.20)
if(NOT TARGET sycl-jit OR NOT SYCL_JIT_STATIC)
  message(FATAL_ERROR "Attach to the original pinned static source-JIT build")
endif()
get_target_property(jit_type sycl-jit TYPE)
if(NOT jit_type STREQUAL "STATIC_LIBRARY")
  message(FATAL_ERROR "The proof must not link the original shared compiler")
endif()

# HeatUtils has a source-only -fno-builtin option in the hash-bound Windows host policy.
# Skip only its target PCH so it is not compiled with a PCH made under different flags.
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  if(NOT TARGET LLVMAnalysis)
    message(FATAL_ERROR "Windows IntelLLVM HeatUtils PCH policy requires LLVMAnalysis")
  endif()
  set_property(SOURCE "${CMAKE_SOURCE_DIR}/lib/Analysis/HeatUtils.cpp"
      TARGET_DIRECTORY LLVMAnalysis PROPERTY SKIP_PRECOMPILE_HEADERS ON)
endif()

# CMake's Windows-IntelLLVM rules wrap linker switches for icx-cl. Our explicit direct
# LLD rules cannot consume that driver prefix; retain the two original delay-load switches.
function(mom_strip_direct_lld_delayload_wrappers target)
  if(NOT TARGET "${target}")
    return()
  endif()
  foreach(property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
    get_target_property(current "${target}" "${property}")
    if(NOT "${current}" STREQUAL "" AND NOT "${current}" MATCHES "-NOTFOUND$")
      set(adapted "${current}")
      foreach(system_dll IN ITEMS shell32 ole32)
        string(REPLACE "-Qoption,link,>-delayload:${system_dll}.dll" ">-delayload:${system_dll}.dll" adapted "${adapted}")
      endforeach()
      if(NOT "${adapted}" STREQUAL "${current}")
        set_property(TARGET "${target}" PROPERTY "${property}" "${adapted}")
      endif()
    endif()
  endforeach()
endfunction()

get_filename_component(mom_jit_linker_name "${CMAKE_LINKER}" NAME)
string(TOLOWER "${mom_jit_linker_name}" mom_jit_linker_name)
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND
    mom_jit_linker_name MATCHES "^lld-link(\\.exe)?$" AND
    CMAKE_C_LINK_EXECUTABLE MATCHES "<CMAKE_LINKER>" AND CMAKE_CXX_LINK_EXECUTABLE MATCHES "<CMAKE_LINKER>")
  mom_strip_direct_lld_delayload_wrappers(LLVMSupport)
endif()

set(native_dpcpp_jit_sources "${MOM_SOURCE_REPO}/tests/reference/native_dpcpp_jit.cpp"
    "${MOM_SOURCE_REPO}/tests/reference/native_kawpow_source.cpp")
if(NOT WIN32)
  # The Linux-only loader interposition boundary is unrelated to this CPU-only source proof.
  list(APPEND native_dpcpp_jit_sources "${MOM_SOURCE_REPO}/runtime/driver-library.cpp")
endif()
add_executable(native-dpcpp-jit ${native_dpcpp_jit_sources})
find_program(MOM_NODE node REQUIRED)
set(jit_resources "${CMAKE_BINARY_DIR}/mom-kawpow/kawpow-jit-resources.inc")
add_custom_command(OUTPUT "${jit_resources}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/mom-kawpow"
    COMMAND "${MOM_NODE}" "${MOM_SOURCE_REPO}/scripts/native-resources.js" "${jit_resources}"
        kawpow-device.inc "${MOM_SOURCE_REPO}/sycl/kawpow/device.inc"
        kawpow-keccak.inc "${MOM_SOURCE_REPO}/sycl/kawpow/keccak.inc"
    DEPENDS "${MOM_SOURCE_REPO}/scripts/native-resources.js"
        "${MOM_SOURCE_REPO}/sycl/kawpow/device.inc" "${MOM_SOURCE_REPO}/sycl/kawpow/keccak.inc"
    VERBATIM)
target_sources(native-dpcpp-jit PRIVATE "${jit_resources}")
target_compile_definitions(native-dpcpp-jit PRIVATE MOM_SYCL_HAS_CUDA MOM_EMBEDDED_SYCL_JIT)
target_compile_features(native-dpcpp-jit PRIVATE cxx_std_17)
# Hermetic clang-cl suppresses -imsvc, even for these explicitly selected project roots.
set(native_dpcpp_jit_system SYSTEM)
if(WIN32)
  set(native_dpcpp_jit_system)
endif()
target_include_directories(native-dpcpp-jit ${native_dpcpp_jit_system} PRIVATE
    "${LLVM_EXTERNAL_SYCL_SOURCE_DIR}/include" "${CMAKE_BINARY_DIR}/include"
    "${SYCL_JIT_RESOURCE_OVERLAY}/include" "${CMAKE_BINARY_DIR}/mom-kawpow")
# The original target owns every compiler/pass dependency and embedded resource; no second list.
if(WIN32)
  # Keep the source-JIT fixture on the same static CRT as the original compiler archives.
  set_property(TARGET native-dpcpp-jit PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded)
  target_compile_options(native-dpcpp-jit PRIVATE /EHsc)
  # LLVM Support and the original SYCL-JIT target own these Windows system dependencies;
  # compiler/pass archives remain transitive through sycl-jit.
  target_link_libraries(native-dpcpp-jit PRIVATE sycl-jit psapi shell32 ole32 uuid advapi32 ws2_32 ntdll shlwapi)
  target_link_options(native-dpcpp-jit PRIVATE /SUBSYSTEM:CONSOLE /INCREMENTAL:NO)
else()
  target_link_libraries(native-dpcpp-jit PRIVATE sycl-jit dl pthread m)
  target_link_options(native-dpcpp-jit PRIVATE -Wl,--gc-sections -Wl,--wrap=dlopen -Wl,--wrap=dlerror -static-libstdc++ -static-libgcc)
endif()
