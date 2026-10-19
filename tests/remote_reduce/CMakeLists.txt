cmake_minimum_required(VERSION 3.24)
project(remote_reduce_parity LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 120)
find_package(CUDAToolkit REQUIRED)
get_filename_component(STRATA_ROOT ../.. ABSOLUTE)
set(GGML_ROOT "" CACHE PATH "Existing llama.cpp checkout (headers only)")
add_executable(remote_reduce_parity remote_reduce_parity.cu
    ${STRATA_ROOT}/src/core/remote_expert_opt.cu
    ${STRATA_ROOT}/src/core/remote_experts.cpp
    ${STRATA_ROOT}/src/core/expert_cache.cpp
    ${STRATA_ROOT}/src/core/vmm.cpp)
target_include_directories(remote_reduce_parity PRIVATE ${STRATA_ROOT}/include
    ${GGML_ROOT}/ggml/include ${GGML_ROOT}/ggml/src)
# Keep actual constructors/destructors/prepare/reduce/finish; discard unrelated
# model-loading and inference entry points, not replaced by test stubs.
target_compile_options(remote_reduce_parity PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections;-fdata-sections>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections,-fdata-sections>)
target_link_options(remote_reduce_parity PRIVATE -Wl,--gc-sections)
target_link_libraries(remote_reduce_parity PRIVATE CUDA::cudart CUDA::cuda_driver)
