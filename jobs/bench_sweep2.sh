#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_sweep2

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"

echo "=== v5 (Tiling_B) block sweep @ N=8192 ==="
for k in 768 1024 1536 2048; do
  for n in 240 288 384 432 480 528 576 672; do
    $NUMA ./bench v5 8192 8192 8192 3 $k $n
  done
done

echo "=== v3 (Base-kernel) block sweep @ N=8192 ==="
for k in 768 1024 1536 2048; do
  for n in 192 256 320 384 480 512 576 640; do
    $NUMA ./bench v3 8192 8192 8192 3 $k $n
  done
done

echo "=== v5 sweep @ N=10000 (代表的な (k,n)) ==="
for k in 1024 1536 2048; do
  for n in 288 480 576 672; do
    $NUMA ./bench v5 10000 10000 10000 3 $k $n
  done
done

echo "=== v3 sweep @ N=10000 (代表的な (k,n)) ==="
for k in 1024 1536 2048; do
  for n in 256 384 512 576; do
    $NUMA ./bench v3 10000 10000 10000 3 $k $n
  done
done

echo "=== MKL @ N=10000 ==="
$NUMA ./bench mkl 10000 10000 10000 3
$NUMA ./bench mkl 8192 8192 8192 3
echo DONE
