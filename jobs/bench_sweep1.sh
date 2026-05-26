#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_sweep1

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"

# ベース: 各バリアントを N=4096, 8192 で同じブロックサイズ条件比較
echo "=== Baseline @ 4096 ==="
$NUMA ./bench mkl 4096 4096 4096 5
$NUMA ./bench v3 4096 4096 4096 5 1024 256
$NUMA ./bench v4 4096 4096 4096 5 1024 256
$NUMA ./bench v5 4096 4096 4096 5 1536 480
$NUMA ./bench v6 4096 4096 4096 5 1536 480

echo "=== Baseline @ 8192 ==="
$NUMA ./bench mkl 8192 8192 8192 3
$NUMA ./bench v3 8192 8192 8192 3 1024 256
$NUMA ./bench v4 8192 8192 8192 3 1024 256
$NUMA ./bench v5 8192 8192 8192 3 1536 480
$NUMA ./bench v6 8192 8192 8192 3 1536 480

echo "=== v3 (Base-kernel) block sweep @ 4096 ==="
for k in 1024 1536 2048; do
  for n in 256 288 480 512 576; do
    $NUMA ./bench v3 4096 4096 4096 5 $k $n
  done
done

echo "=== v5 (Tiling_B no pf) block sweep @ 4096 ==="
for k in 1024 1536 2048; do
  for n in 288 480 576; do
    $NUMA ./bench v5 4096 4096 4096 5 $k $n
  done
done

echo "=== v6 (Tiling_B + light pf) block sweep @ 4096 ==="
for k in 1024 1536 2048; do
  for n in 288 480 576; do
    $NUMA ./bench v6 4096 4096 4096 5 $k $n
  done
done

echo DONE
