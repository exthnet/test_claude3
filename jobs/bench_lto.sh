#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_lto

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"

echo "===== LTO build comparison ====="
for N in 8192 10000 12000 16384; do
  ITERS=10
  if [ "$N" -le 8192 ]; then ITERS=20; fi
  if [ "$N" -ge 12000 ]; then ITERS=5; fi
  $NUMA ./bench mkl $N $N $N $ITERS
  $NUMA ./bench v9  $N $N $N $ITERS 1280 528
  $NUMA ./bench v12 $N $N $N $ITERS 1024 640
done

echo "===== Reproducibility v9 + LTO @ N=10000 ====="
for i in 1 2 3 4 5; do
  $NUMA ./bench v9 10000 10000 10000 20 1280 528
done

echo "===== v12 + LTO @ N=10000 with various blocks ====="
for k in 1024 1280; do
  for n in 480 528 640; do
    $NUMA ./bench v12 10000 10000 10000 10 $k $n
  done
done

echo DONE
