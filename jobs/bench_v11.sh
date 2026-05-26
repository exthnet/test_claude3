#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_v11

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"

echo "===== v9 stability check: many iterations @ N=10000 ====="
$NUMA ./bench v9 10000 10000 10000 20 1280 528
$NUMA ./bench v9 10000 10000 10000 20 1280 528
$NUMA ./bench v9 10000 10000 10000 20 1280 528

echo "===== v11 M-panel inside-nc sweep @ N=10000 ====="
for mpanel in 256 512 1024 2048 5120 10016; do
  $NUMA ./bench v11 10000 10000 10000 5 1280 528 $mpanel
done

echo "===== v11 at all sizes ====="
for N in 8192 10000 12000 16384; do
  ITERS=5
  if [ "$N" -ge 12000 ]; then ITERS=3; fi
  $NUMA ./bench v11 $N $N $N $ITERS 1280 528 1024
done

echo "===== v9 best with longer iterations to find real best ====="
$NUMA ./bench v9 10000 10000 10000 20 1024 528
$NUMA ./bench v9 10000 10000 10000 20 1280 432
$NUMA ./bench v9 10000 10000 10000 20 1280 480
$NUMA ./bench v9 10000 10000 10000 20 1536 384

echo "===== MKL multi-iter for accurate baseline ====="
$NUMA ./bench mkl 10000 10000 10000 20
$NUMA ./bench mkl 10000 10000 10000 20
$NUMA ./bench mkl 8192 8192 8192 20

echo DONE
