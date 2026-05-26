#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_final2

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"

echo "===== v12 (Base + Kb spec) sweep @ N=10000 ====="
for k in 1024 1280 1536; do
  for n in 320 384 480 512 640 768; do
    $NUMA ./bench v12 10000 10000 10000 5 $k $n
  done
done

echo "===== v12 vs others @ all sizes ====="
for N in 4096 8192 10000 12000 16384; do
  ITERS=5
  if [ "$N" -ge 12000 ]; then ITERS=3; fi
  $NUMA ./bench mkl $N $N $N $ITERS
  $NUMA ./bench v9  $N $N $N $ITERS 1280 528
  $NUMA ./bench v12 $N $N $N $ITERS 1024 640
done

echo "===== Reproducibility @ best blocks ====="
for i in 1 2 3 4; do
  $NUMA ./bench v9  10000 10000 10000 10 1280 528
done
for i in 1 2 3 4; do
  $NUMA ./bench v12 10000 10000 10000 10 1024 640
done
for i in 1 2 3 4; do
  $NUMA ./bench mkl 10000 10000 10000 10
done

echo DONE
