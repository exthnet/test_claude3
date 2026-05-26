#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_v9

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"

echo "===== v9 vs v8 vs v5 head-to-head @ best blocks ====="
for N in 8192 10000 12000; do
  if [ "$N" -le 8192 ]; then ITERS=3; else ITERS=3; fi
  echo "--- N=$N k=1024 n=480 ---"
  $NUMA ./bench v5 $N $N $N $ITERS 1024 480
  $NUMA ./bench v8 $N $N $N $ITERS 1024 480
  $NUMA ./bench v9 $N $N $N $ITERS 1024 480
  echo "--- N=$N k=1280 n=528 (v9 expected best) ---"
  $NUMA ./bench v5 $N $N $N $ITERS 1280 528
  $NUMA ./bench v8 $N $N $N $ITERS 1280 528
  $NUMA ./bench v9 $N $N $N $ITERS 1280 528
done

echo "===== v9 fine block sweep @ N=10000 ====="
for k in 768 1024 1280 1536 1792 1920 2048; do
  for n in 384 432 480 528 576 624; do
    $NUMA ./bench v9 10000 10000 10000 3 $k $n
  done
done

echo "===== v9 vs MKL @ all sizes ====="
for N in 4096 8192 10000 12000 16384; do
  ITERS=3
  $NUMA ./bench mkl $N $N $N $ITERS
  $NUMA ./bench v9 $N $N $N $ITERS 1280 528
done

echo DONE
