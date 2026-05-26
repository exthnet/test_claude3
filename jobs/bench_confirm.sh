#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_confirm

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"

echo "===== Multi-run high-iter confirmation ====="
# 各サイズで MKL, v9, v12 を 30 反復 × 3 回
for i in 1 2 3; do
  echo "--- pass $i ---"
  for N in 8192 10000 12000 16384; do
    ITERS=10
    if [ "$N" -le 8192 ]; then ITERS=20; fi
    if [ "$N" -ge 12000 ]; then ITERS=5; fi
    $NUMA ./bench mkl $N $N $N $ITERS
    $NUMA ./bench v9  $N $N $N $ITERS 1280 528
    $NUMA ./bench v12 $N $N $N $ITERS 1024 640
  done
done

echo "===== Comprehensive sweep at 1024-16384 ====="
for N in 1024 2048 4096 8192 10000 12000 14000 16384; do
  ITERS=10
  if [ "$N" -le 2048 ]; then ITERS=30; fi
  if [ "$N" -ge 12000 ]; then ITERS=3; fi
  $NUMA ./bench mkl $N $N $N $ITERS
  $NUMA ./bench v9  $N $N $N $ITERS 1280 528
  $NUMA ./bench v12 $N $N $N $ITERS 1024 640
done

echo DONE
