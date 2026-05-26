# Makefile for AMX BF16 GEMM benchmark
# 1 コアのみ、AMX/BF16 を有効化。MKL と OpenBLAS にリンク。

MKLROOT ?= /home/app/inteloneapi/2025.1.3/mkl/latest

CC = gcc
CFLAGS = -O3 -march=sapphirerapids -mamx-tile -mamx-bf16 -mamx-int8 \
         -mavx512f -mavx512bf16 -funroll-loops -Wall \
         -I$(MKLROOT)/include

# MKL: シーケンシャル版にリンク (1 スレッド固定のため)
MKL_LIBS = -L$(MKLROOT)/lib/intel64 -Wl,--start-group \
           -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
           -Wl,--end-group -lpthread -lm -ldl

LDFLAGS = $(MKL_LIBS)

OBJS = bench.o gemm_amx.o gemm_libs.o

all: bench

bench: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c common.h amx_kernels.h gemm_variants.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o bench

.PHONY: all clean
