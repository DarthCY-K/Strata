#!/bin/bash
set -u
T=/root/strata/Strata-nvfp4
LOG=/root/strata/build-nvfp4.log
V=/root/strata/Strata-145/.venv/bin
NVCC=/usr/local/cuda-13.1/bin/nvcc
{
echo "== $(date -Is) build against Strata-145 vendored llama.cpp"
ls $T/CMakeLists.txt || exit 11
export PATH=/usr/local/cuda-13.1/bin:$PATH
rm -rf $T/build
$V/cmake -G Ninja -DCMAKE_MAKE_PROGRAM=$V/ninja -S $T -B $T/build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=$NVCC -DSTRATA_GGML_DIR=/root/strata/Strata-145/third_party/llama.cpp > $LOG.cmake 2>&1 || { echo CMAKE_FAIL; tail -30 $LOG.cmake; exit 14; }
nice -n 10 $V/cmake --build $T/build --target strata -j 12 > $LOG.build 2>&1 || { echo BUILD_FAIL; grep -nE 'error #|FAILED: |error:' $LOG.build | head -30; exit 15; }
mkdir -p $T/engine && cp -f $T/build/strata $T/engine/strata && sha256sum $T/engine/strata
echo BUILD_NVFP4_OK
} 2>&1 | tee -a $LOG
