#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_v8

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"

echo "===== v8 (variable K_block) vs v5 ====="
for N in 4096 8192 10000 12000 16384; do
  if [ "$N" -le 4096 ]; then ITERS=5; else ITERS=3; fi
  echo "--- N=$N ---"
  $NUMA ./bench mkl $N $N $N $ITERS
  $NUMA ./bench v5 $N $N $N $ITERS 1024 480
  $NUMA ./bench v8 $N $N $N $ITERS 1024 480
done

echo "===== v8 block sweep @ N=10000 ====="
for k in 768 1024 1280 1536; do
  for n in 384 432 480 528 576 624; do
    $NUMA ./bench v8 10000 10000 10000 3 $k $n
  done
done

echo DONE
