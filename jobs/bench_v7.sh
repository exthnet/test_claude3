#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_v7

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"

echo "===== v7 (C-resident) sweep ====="
SIZES="2048 4096 8192 10000 12000"
for N in $SIZES; do
  if [ "$N" -le 4096 ]; then ITERS=5; else ITERS=3; fi
  $NUMA ./bench v7 $N $N $N $ITERS 1024 480
done

echo "===== huge page test (no HP / HP) ====="
for V in v3 v5 v7; do
  for N in 4096 8192 10000 12000; do
    if [ "$N" -le 4096 ]; then ITERS=5; else ITERS=3; fi
    if [ "$V" = "v3" ]; then NB=640; else NB=480; fi
    echo "--- $V N=$N noHP ---"
    $NUMA ./bench $V $N $N $N $ITERS 1024 $NB
    echo "--- $V N=$N HP ---"
    AMX_HUGEPAGE=1 $NUMA ./bench $V $N $N $N $ITERS 1024 $NB
  done
done

echo "===== Fine block sweep v7 @ N=10000 ====="
for k in 768 1024 1280 1536; do
  for n in 384 432 480 528 576; do
    AMX_HUGEPAGE=1 $NUMA ./bench v7 10000 10000 10000 3 $k $n
  done
done

echo DONE
