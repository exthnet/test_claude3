// gemm_libs.c: MKL ラッパ (OpenBLAS はシステム版が古く sbgemm 非対応のため省略)
#include "common.h"
#include "gemm_variants.h"
#include <mkl.h>

void gemm_mkl(int M, int N, int K,
              const bf16_t *A, int lda,
              const bf16_t *B, int ldb,
              float *C, int ldc)
{
    cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                           M, N, K,
                           1.0f,
                           (const MKL_BF16 *)A, lda,
                           (const MKL_BF16 *)B, ldb,
                           1.0f,
                           C, ldc);
}

// OpenBLAS sbgemm はシステムの openblas (0.3.15) が未提供のためスタブ
void gemm_openblas(int M, int N, int K,
                   const bf16_t *A, int lda,
                   const bf16_t *B, int ldb,
                   float *C, int ldc)
{
    (void)M; (void)N; (void)K; (void)A; (void)lda; (void)B; (void)ldb; (void)C; (void)ldc;
    fprintf(stderr, "OpenBLAS sbgemm unavailable on this system (need >=0.3.27)\n");
    exit(1);
}
