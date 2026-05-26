#!/bin/bash
#PJM -L rscgrp=a-batch-low
#PJM -L node=1
#PJM -L elapse=10:00
#PJM -j
#PJM -N amx_smoke

set -eu
module load gcc-toolset/13

ROOT=/home/pj24001603/ku40000105/work/test_claude3
cd $ROOT/src

export LD_LIBRARY_PATH=/home/app/inteloneapi/2025.1.3/mkl/latest/lib/intel64:/home/app/inteloneapi/2025.1.3/2025.1/lib:${LD_LIBRARY_PATH:-}
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "==== CPU info ===="
lscpu | grep -E 'Model name|MHz|Socket|Core|NUMA' | head -10
echo "==== AMX flags ===="
grep -m1 -oE 'amx[_a-z0-9]*' /proc/cpuinfo | sort -u

echo "==== Smoke test: N=2048 with numactl single NUMA ===="
NUMA="numactl --cpunodebind=0 --membind=0 --physcpubind=0"
for V in mkl v1 v2 v3 v4 v5; do
  KB=1024; NB=256
  if [ "$V" = "v5" ]; then KB=1536; NB=480; fi
  $NUMA ./bench $V 2048 2048 2048 10 $KB $NB
done

echo "==== Smoke test: N=4096 ===="
for V in mkl v3 v4 v5; do
  KB=1024; NB=256
  if [ "$V" = "v5" ]; then KB=1536; NB=480; fi
  $NUMA ./bench $V 4096 4096 4096 5 $KB $NB
done

echo "==== Smoke test: N=8192 ===="
for V in mkl v3 v4 v5; do
  KB=1024; NB=256
  if [ "$V" = "v5" ]; then KB=1536; NB=480; fi
  $NUMA ./bench $V 8192 8192 8192 3 $KB $NB
done

echo DONE
