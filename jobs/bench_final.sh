#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_final

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"

# ============================================================
# 1) 全サイズ × 主要バリアント比較
# ============================================================
echo "===== Section 1: size sweep, main variants ====="
SIZES="1024 2048 4096 8192 10000 12000 16384"
for N in $SIZES; do
  if [ "$N" -le "2048" ]; then ITERS=10; elif [ "$N" -le "4096" ]; then ITERS=5; else ITERS=3; fi
  echo "--- N=$N ---"
  $NUMA ./bench mkl  $N $N $N $ITERS
  if [ "$N" -le 4096 ]; then
    $NUMA ./bench v1   $N $N $N $ITERS
    $NUMA ./bench v2   $N $N $N $ITERS 1024  0
  fi
  $NUMA ./bench v3   $N $N $N $ITERS 1024 640
  $NUMA ./bench v4   $N $N $N $ITERS 1024 640
  $NUMA ./bench v5   $N $N $N $ITERS 1024 480
  $NUMA ./bench v6   $N $N $N $ITERS 1024 480
done

# ============================================================
# 2) 巨大ページの効果 (v5 最良ブロックで)
# ============================================================
echo "===== Section 2: hugepage test ====="
for N in 4096 8192 10000 12000; do
  if [ "$N" -le 4096 ]; then ITERS=5; else ITERS=3; fi
  echo "--- N=$N (no HP) ---"
  $NUMA ./bench v5 $N $N $N $ITERS 1024 480
  echo "--- N=$N (HP) ---"
  AMX_HUGEPAGE=1 $NUMA ./bench v5 $N $N $N $ITERS 1024 480
done

# ============================================================
# 3) N=10000 ピーク探索 (細かいブロックサイズ)
# ============================================================
echo "===== Section 3: fine block sweep @ N=10000 ====="
for k in 896 1024 1152; do
  for n in 432 480 528 576 624; do
    $NUMA ./bench v5 10000 10000 10000 3 $k $n
  done
done

echo DONE
