cmake_minimum_required(VERSION 3.16)
project("llama.cpp" C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Force release-style codegen with PowerPC 750 tuning/LTO for Wii U
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG -mcpu=750 -mtune=750 -flto" CACHE STRING "C flags for Release" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=750 -mtune=750 -flto" CACHE STRING "CXX flags for Release" FORCE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON CACHE BOOL "Enable LTO" FORCE)

if (NOT CMAKE_SYSTEM_NAME STREQUAL "CafeOS")
    message(FATAL_ERROR "This tree only supports Wii U (CafeOS) builds.")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

set(LLAMA_ALL_WARNINGS ON CACHE BOOL "llama: enable all compiler warnings" FORCE)
set(LLAMA_FATAL_WARNINGS OFF CACHE BOOL "llama: enable -Werror flag" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)

set(LLAMA_BUILD_NUMBER ${BUILD_NUMBER})
set(LLAMA_BUILD_COMMIT ${BUILD_COMMIT})
set(LLAMA_INSTALL_VERSION 0.0.${LLAMA_BUILD_NUMBER})

# ggml is still built from source, but every non-Wii U backend is disabled
set(GGML_ALL_WARNINGS ${LLAMA_ALL_WARNINGS} CACHE BOOL "" FORCE)
set(GGML_FATAL_WARNINGS ${LLAMA_FATAL_WARNINGS} CACHE BOOL "" FORCE)
set(GGML_BUILD_NUMBER ${LLAMA_BUILD_NUMBER})
set(GGML_BUILD_COMMIT ${LLAMA_BUILD_COMMIT})
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_POWERPC_CPUTYPE 750 CACHE STRING "ggml: CPU type for PowerPC" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_MUSA OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_KOMPUTE OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)
set(GGML_MPI OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_CPU_HBM OFF CACHE BOOL "" FORCE)
set(GGML_CPU_REPACK ON CACHE BOOL "" FORCE)

add_subdirectory(ggml)
add_subdirectory(src)

add_executable(llama-wiiu wiiu/main.cpp wiiu/wut_shims.c)
target_sources(llama-wiiu PRIVATE ggml/src/ggml-cpu/ggml-cpu.cpp)
target_include_directories(llama-wiiu PRIVATE ggml/src ggml/src/ggml-cpu)
target_link_libraries(llama-wiiu PRIVATE llama ggml ggml-cpu ggml-base)
wut_create_rpx(llama-wiiu)

set(WIIU_WUHB_OUTPUT "${CMAKE_BINARY_DIR}/llama-wiiu.wuhb")
wut_create_wuhb(llama-wiiu-wuhb
    TARGET llama-wiiu
    OUTPUT ${WIIU_WUHB_OUTPUT}
    NAME "llama.cpp"
    SHORTNAME "llama"
    AUTHOR "ggml-org")
