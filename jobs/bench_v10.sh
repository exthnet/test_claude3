#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_v10

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"

echo "===== v10 (M-panel) sweep @ N=10000, k=1280 n=528 ====="
for mpanel in 64 96 128 160 192 256 320 384 512 640 768 1024 2048; do
  $NUMA ./bench v10 10000 10000 10000 5 1280 528 $mpanel
done

echo "===== v10 vs v9 head-to-head ====="
for N in 8192 10000 12000 16384; do
  ITERS=5
  if [ "$N" -ge 12000 ]; then ITERS=3; fi
  $NUMA ./bench v9  $N $N $N $ITERS 1280 528
  $NUMA ./bench v10 $N $N $N $ITERS 1280 528 256
  $NUMA ./bench v10 $N $N $N $ITERS 1280 528 512
done

echo "===== Reproducibility @ best blocks (more iters) ====="
$NUMA ./bench v9  10000 10000 10000 10 1280 528
$NUMA ./bench v9  10000 10000 10000 10 1280 528
$NUMA ./bench v10 10000 10000 10000 10 1280 528 256

echo DONE
