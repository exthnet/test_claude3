// gemm_variants.h: GEMM 実装バリアント宣言
#ifndef GEMM_VARIANTS_H
#define GEMM_VARIANTS_H

#include "common.h"

// すべて C = C + A * B (row-major, transA=transB=N) 形式。
// A: M x K (bf16), B: K x N (bf16), C: M x N (fp32)
// lda/ldb/ldc は要素数。

void gemm_mkl(int M, int N, int K,
              const bf16_t *A, int lda,
              const bf16_t *B, int ldb,
              float *C, int ldc);

void gemm_openblas(int M, int N, int K,
                   const bf16_t *A, int lda,
                   const bf16_t *B, int ldb,
                   float *C, int ldc);

// v1: 最小 AMX (パック / ブロッキングなし、16x16 単タイル出力ループ、内側 K=32 stride)
void gemm_v1_amx_naive(int M, int N, int K,
                       const bf16_t *A, int lda,
                       const bf16_t *B, int ldb,
                       float *C, int ldc);

// v2: Base-kernel + 単一 K ブロック + パッキング (n ブロック化なし、n_block = N)
void gemm_v2_base_pack(int M, int N, int K,
                       const bf16_t *A, int lda,
                       const bf16_t *B, int ldb,
                       float *C, int ldc,
                       int K_block);

// v3: Base-kernel + (k, n) 2 レベルキャッシュブロッキング + パッキング
void gemm_v3_base_blocked(int M, int N, int K,
                          const bf16_t *A, int lda,
                          const bf16_t *B, int ldb,
                          float *C, int ldc,
                          int K_block, int N_block);

// v4: v3 + プリフェッチ (BUFFER_A を L1 へ)
void gemm_v4_base_pf(int M, int N, int K,
                     const bf16_t *A, int lda,
                     const bf16_t *B, int ldb,
                     float *C, int ldc,
                     int K_block, int N_block);

// v5: Tiling_B カーネル + ブロッキング (プリフェッチなし)
void gemm_v5_tilingB(int M, int N, int K,
                     const bf16_t *A, int lda,
                     const bf16_t *B, int ldb,
                     float *C, int ldc,
                     int K_block, int N_block);

// v6: Tiling_B カーネル + ブロッキング + 軽量プリフェッチ
void gemm_v6_tilingB_pf(int M, int N, int K,
                        const bf16_t *A, int lda,
                        const bf16_t *B, int ldb,
                        float *C, int ldc,
                        int K_block, int N_block);

// v7: Tiling_B + kc 反転ループ + C タイル kc 跨ぎ常駐
//   A 全体を一度パック (M_panel × K_padded)、B は nc 毎にパック (K_padded × N_block)。
//   kc ループは (mp, np) の中。C tile は (mp, np) ごとに 1 度だけ load/store。
void gemm_v7_tilingB_cresid(int M, int N, int K,
                            const bf16_t *A, int lda,
                            const bf16_t *B, int ldb,
                            float *C, int ldc,
                            int K_block, int N_block);

// v8: v5 + 可変 K_block (最終 kc を K の残りに合わせ縮小して K パディングを最小化)
void gemm_v8_tilingB_varkblock(int M, int N, int K,
                               const bf16_t *A, int lda,
                               const bf16_t *B, int ldb,
                               float *C, int ldc,
                               int K_block, int N_block);

// v9: v8 + カーネル特殊化 (Kb をコンパイル時定数として実体化)
void gemm_v9_tilingB_kbspec(int M, int N, int K,
                            const bf16_t *A, int lda,
                            const bf16_t *B, int ldb,
                            float *C, int ldc,
                            int K_block, int N_block);

// v10: v9 + M_panel ブロッキング (mp_panel OUTSIDE nc)
//   bufA panel が L2 で温まることを狙う構造。しかし bufB を (M/M_panel) 倍多く
//   パックすることになり、パッキングオーバーヘッドが大きくなる (実測で失敗確認)。
void gemm_v10_tilingB_mpanel(int M, int N, int K,
                             const bf16_t *A, int lda,
                             const bf16_t *B, int ldb,
                             float *C, int ldc,
                             int K_block, int N_block, int M_panel);

// v11: 改良 M-panel ブロッキング (mp_panel INSIDE nc)
//   bufB は v9 と同じ頻度でパック。mp_panel は nc 内で実行。
//   このループ順では bufA panel の L2 再利用は限定的だが、
//   nc 内の mp 反復で空間局所性を改善する可能性。
void gemm_v11_tilingB_mpanel(int M, int N, int K,
                             const bf16_t *A, int lda,
                             const bf16_t *B, int ldb,
                             float *C, int ldc,
                             int K_block, int N_block, int M_panel);

// v12: Base-kernel (32x32) + 可変 K_block + Kb 特殊化 (v9 と同じ手法を Base-kernel に)
void gemm_v12_base_kbspec(int M, int N, int K,
                          const bf16_t *A, int lda,
                          const bf16_t *B, int ldb,
                          float *C, int ldc,
                          int K_block, int N_block);

#endif
