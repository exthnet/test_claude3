// gemm_amx.c: 自前 AMX GEMM 実装群
#include "common.h"
#include "amx_kernels.h"
#include "gemm_variants.h"

// パディング対応版 pack_A_block: 範囲外を 0 で埋める
// 出力 bufA は (M_panel/16) * (K_block/32) 個の 16x32 マイクロブロックを格納
static void pack_A_padded(const bf16_t *A, int lda,
                          int m_off, int k_off,
                          int M_panel, int K_block,
                          int M_real, int K_real,
                          bf16_t *bufA)
{
    int Mb = M_panel / 16;
    int Kb = K_block / 32;
    for (int mb = 0; mb < Mb; ++mb) {
        for (int kb = 0; kb < Kb; ++kb) {
            bf16_t *dst = bufA + (mb * Kb + kb) * 16 * 32;
            for (int i = 0; i < 16; ++i) {
                int m = m_off + mb*16 + i;
                if (m < M_real) {
                    int k_start = k_off + kb*32;
                    int valid = K_real - k_start;
                    if (valid <= 0) {
                        memset(dst + i*32, 0, 32 * sizeof(bf16_t));
                    } else if (valid >= 32) {
                        memcpy(dst + i*32, &A[m * lda + k_start], 32 * sizeof(bf16_t));
                    } else {
                        memcpy(dst + i*32, &A[m * lda + k_start], valid * sizeof(bf16_t));
                        memset(dst + i*32 + valid, 0, (32 - valid) * sizeof(bf16_t));
                    }
                } else {
                    memset(dst + i*32, 0, 32 * sizeof(bf16_t));
                }
            }
        }
    }
}

// パディング対応版 pack_B (VNNI)。AVX-512 で高速化。
static void pack_B_padded(const bf16_t *B, int ldb,
                          int k_off, int n_off,
                          int K_block, int N_block,
                          int K_real, int N_real,
                          bf16_t *bufB)
{
    int Nb = N_block / 16;
    int Kb = K_block / 32;
    for (int nb = 0; nb < Nb; ++nb) {
        for (int kb = 0; kb < Kb; ++kb) {
            bf16_t *dst = bufB + (nb * Kb + kb) * 16 * 32;
            for (int kp = 0; kp < 16; ++kp) {
                int k0 = k_off + kb*32 + 2*kp;
                int k1 = k0 + 1;
                int nbase = n_off + nb*16;
                bf16_t *out = dst + kp * 32;
                int nvalid = N_real - nbase;
                if (nvalid < 0) nvalid = 0;
                if (nvalid > 16) nvalid = 16;
                int k0_ok = (k0 < K_real);
                int k1_ok = (k1 < K_real);
                if (k0_ok && k1_ok && nvalid == 16) {
                    const bf16_t *r0 = &B[k0 * ldb + nbase];
                    const bf16_t *r1 = &B[k1 * ldb + nbase];
                    // AVX2 でインターリーブ:
                    //   a = [r0[0..15]] (16 bf16 = 256 bit)
                    //   b = [r1[0..15]]
                    //   out = [r0[0],r1[0], r0[1],r1[1], ..., r0[15],r1[15]] (32 bf16 = 512 bit)
                    __m256i a = _mm256_loadu_si256((const __m256i *)r0);
                    __m256i b = _mm256_loadu_si256((const __m256i *)r1);
                    __m256i lo = _mm256_unpacklo_epi16(a, b);
                    __m256i hi = _mm256_unpackhi_epi16(a, b);
                    __m256i out_lo = _mm256_permute2x128_si256(lo, hi, 0x20);
                    __m256i out_hi = _mm256_permute2x128_si256(lo, hi, 0x31);
                    _mm256_storeu_si256((__m256i *)out,        out_lo);
                    _mm256_storeu_si256((__m256i *)(out + 16), out_hi);
                } else {
                    for (int nc = 0; nc < 16; ++nc) {
                        out[2*nc + 0] = (k0_ok && nc < nvalid) ? B[k0*ldb + nbase + nc] : 0;
                        out[2*nc + 1] = (k1_ok && nc < nvalid) ? B[k1*ldb + nbase + nc] : 0;
                    }
                }
            }
        }
    }
}

// ============================================================
// v1: 最小 AMX — Base-kernel を 1 個だけ (32x32 出力), パックなし
//   ただし B はパック必須 (VNNI レイアウト要求)。
//   ここでは「ブロッキングなし」のデモンストレーション目的なので、
//   K_block = K, n_block = N, k=K で 1 回パッキング。
//   実質 v2 と同じだが「Kブロックを切らない」点が違う。
// ============================================================
void gemm_v1_amx_naive(int M, int N, int K,
                       const bf16_t *A, int lda,
                       const bf16_t *B, int ldb,
                       float *C, int ldc)
{
    amx_config_8tiles();

    // 32 倍数にパディング
    int Mp = (M + 31) & ~31;
    int Np = (N + 31) & ~31;
    int Kp = (K + 31) & ~31;
    int Kb = Kp / 32;

    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * Kp * sizeof(bf16_t));
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)Np * Kp * sizeof(bf16_t));

    pack_A_padded(A, lda, 0, 0, Mp, Kp, M, K, bufA);
    pack_B_padded(B, ldb, 0, 0, Kp, Np, K, N, bufB);

    // 出力 C パディング: M, N が 32 倍数でない場合は内部バッファを使う
    float *Cp = C;
    int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    // m_pair (×32 stride), n_pair (×32 stride) のループ。インナーは K 全体
    for (int mp = 0; mp < Mp; mp += 32) {
        const bf16_t *bufA0 = bufA + (size_t)(mp/16    ) * Kb * 16*32;
        const bf16_t *bufA1 = bufA + (size_t)(mp/16 + 1) * Kb * 16*32;
        for (int np = 0; np < Np; np += 32) {
            const bf16_t *bufB0 = bufB + (size_t)(np/16    ) * Kb * 16*32;
            const bf16_t *bufB1 = bufB + (size_t)(np/16 + 1) * Kb * 16*32;
            kernel_base_32x32(&Cp[mp*ldcp + np], ldcp,
                              bufA0, bufA1, bufB0, bufB1, Kb);
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) {
            memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        }
        free(Cp);
    }
    free(bufA);
    free(bufB);
    amx_release();
}

// ============================================================
// v2: Base-kernel + K ブロック化 + 部分パッキング
//   各 K_block ごとに A 全体 (M x K_block) と B 全体 (K_block x N) をパック。
//   n ブロック化は無し (n_block = N に固定相当)。
// ============================================================
void gemm_v2_base_pack(int M, int N, int K,
                       const bf16_t *A, int lda,
                       const bf16_t *B, int ldb,
                       float *C, int ldc,
                       int K_block)
{
    amx_config_8tiles();
    int Mp = (M + 31) & ~31;
    int Np = (N + 31) & ~31;
    if (K_block <= 0) K_block = 1024;
    K_block = (K_block + 31) & ~31;
    if (K_block < 32) K_block = 32;
    int Kp = ((K + K_block - 1) / K_block) * K_block;
    if (Kp == 0) Kp = K_block;

    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * K_block * sizeof(bf16_t));
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)Np * K_block * sizeof(bf16_t));

    float *Cp = C; int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    int Kb_blk = K_block / 32;
    for (int kc = 0; kc < Kp; kc += K_block) {
        pack_A_padded(A, lda, 0, kc, Mp, K_block, M, K, bufA);
        pack_B_padded(B, ldb, kc, 0, K_block, Np, K, N, bufB);
        for (int mp = 0; mp < Mp; mp += 32) {
            const bf16_t *bufA0 = bufA + (size_t)(mp/16    ) * Kb_blk * 16*32;
            const bf16_t *bufA1 = bufA + (size_t)(mp/16 + 1) * Kb_blk * 16*32;
            for (int np = 0; np < Np; np += 32) {
                const bf16_t *bufB0 = bufB + (size_t)(np/16    ) * Kb_blk * 16*32;
                const bf16_t *bufB1 = bufB + (size_t)(np/16 + 1) * Kb_blk * 16*32;
                kernel_base_32x32(&Cp[mp*ldcp + np], ldcp,
                                  bufA0, bufA1, bufB0, bufB1, Kb_blk);
            }
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        free(Cp);
    }
    free(bufA); free(bufB);
    amx_release();
}

// ============================================================
// v3: Base-kernel + (K, N) 2 レベルキャッシュブロッキング + パッキング
//   論文 Sec 4.1 と同じ構造:
//     for kc: pack A[M, kc:kc+K_block]
//       for nc: pack B[kc:kc+K_block, nc:nc+N_block]
//         for mp: for np: kernel
// ============================================================
void gemm_v3_base_blocked(int M, int N, int K,
                          const bf16_t *A, int lda,
                          const bf16_t *B, int ldb,
                          float *C, int ldc,
                          int K_block, int N_block)
{
    amx_config_8tiles();
    int Mp = (M + 31) & ~31;
    if (K_block <= 0) K_block = 1024;
    if (N_block <= 0) N_block = 256;
    K_block = (K_block + 31) & ~31;
    N_block = (N_block + 31) & ~31; // 32 倍数にしておく
    if (K_block < 32) K_block = 32;
    if (N_block < 32) N_block = 32;
    int Np = ((N + N_block - 1) / N_block) * N_block;
    int Kp = ((K + K_block - 1) / K_block) * K_block;
    if (Np == 0) Np = N_block;
    if (Kp == 0) Kp = K_block;

    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * K_block * sizeof(bf16_t));
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)N_block * K_block * sizeof(bf16_t));

    float *Cp = C; int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    int Kb_blk = K_block / 32;
    for (int kc = 0; kc < Kp; kc += K_block) {
        pack_A_padded(A, lda, 0, kc, Mp, K_block, M, K, bufA);
        for (int nc = 0; nc < Np; nc += N_block) {
            pack_B_padded(B, ldb, kc, nc, K_block, N_block, K, N, bufB);
            for (int mp = 0; mp < Mp; mp += 32) {
                const bf16_t *bufA0 = bufA + (size_t)(mp/16    ) * Kb_blk * 16*32;
                const bf16_t *bufA1 = bufA + (size_t)(mp/16 + 1) * Kb_blk * 16*32;
                for (int np = 0; np < N_block; np += 32) {
                    const bf16_t *bufB0 = bufB + (size_t)(np/16    ) * Kb_blk * 16*32;
                    const bf16_t *bufB1 = bufB + (size_t)(np/16 + 1) * Kb_blk * 16*32;
                    kernel_base_32x32(&Cp[mp*ldcp + nc + np], ldcp,
                                      bufA0, bufA1, bufB0, bufB1, Kb_blk);
                }
            }
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        free(Cp);
    }
    free(bufA); free(bufB);
    amx_release();
}

// ============================================================
// v4: v3 + プリフェッチ (m_pair ループの「次の」反復で使う A を L1 へ)
// ============================================================
void gemm_v4_base_pf(int M, int N, int K,
                     const bf16_t *A, int lda,
                     const bf16_t *B, int ldb,
                     float *C, int ldc,
                     int K_block, int N_block)
{
    amx_config_8tiles();
    int Mp = (M + 31) & ~31;
    if (K_block <= 0) K_block = 1024;
    if (N_block <= 0) N_block = 256;
    K_block = (K_block + 31) & ~31;
    N_block = (N_block + 31) & ~31;
    if (K_block < 32) K_block = 32;
    if (N_block < 32) N_block = 32;
    int Np = ((N + N_block - 1) / N_block) * N_block;
    int Kp = ((K + K_block - 1) / K_block) * K_block;
    if (Np == 0) Np = N_block;
    if (Kp == 0) Kp = K_block;

    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * K_block * sizeof(bf16_t));
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)N_block * K_block * sizeof(bf16_t));

    float *Cp = C; int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    int Kb_blk = K_block / 32;
    for (int kc = 0; kc < Kp; kc += K_block) {
        pack_A_padded(A, lda, 0, kc, Mp, K_block, M, K, bufA);
        for (int nc = 0; nc < Np; nc += N_block) {
            pack_B_padded(B, ldb, kc, nc, K_block, N_block, K, N, bufB);
            for (int mp = 0; mp < Mp; mp += 32) {
                const bf16_t *bufA0 = bufA + (size_t)(mp/16    ) * Kb_blk * 16*32;
                const bf16_t *bufA1 = bufA + (size_t)(mp/16 + 1) * Kb_blk * 16*32;
                int mp_next = mp + 32;
                const bf16_t *bufA0_next = NULL, *bufA1_next = NULL;
                if (mp_next < Mp) {
                    bufA0_next = bufA + (size_t)(mp_next/16    ) * Kb_blk * 16*32;
                    bufA1_next = bufA + (size_t)(mp_next/16 + 1) * Kb_blk * 16*32;
                }
                for (int np = 0; np < N_block; np += 32) {
                    const bf16_t *bufB0 = bufB + (size_t)(np/16    ) * Kb_blk * 16*32;
                    const bf16_t *bufB1 = bufB + (size_t)(np/16 + 1) * Kb_blk * 16*32;
                    kernel_base_32x32_pf(&Cp[mp*ldcp + nc + np], ldcp,
                                         bufA0, bufA1, bufB0, bufB1,
                                         bufA0_next, bufA1_next, Kb_blk);
                }
            }
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        free(Cp);
    }
    free(bufA); free(bufB);
    amx_release();
}

// ============================================================
// v5: Tiling_B (2x3 = 32 行 x 48 列 C 出力) + ブロッキング (プリフェッチなし)
// ============================================================
void gemm_v5_tilingB(int M, int N, int K,
                     const bf16_t *A, int lda,
                     const bf16_t *B, int ldb,
                     float *C, int ldc,
                     int K_block, int N_block)
{
    amx_config_8tiles();
    int Mp = (M + 31) & ~31;
    if (K_block <= 0) K_block = 1536;
    if (N_block <= 0) N_block = 480;
    K_block = (K_block + 31) & ~31;
    if (N_block < 48) N_block = 48;
    N_block = ((N_block + 47) / 48) * 48;
    if (K_block < 32) K_block = 32;
    int Np = ((N + N_block - 1) / N_block) * N_block;
    int Kp = ((K + K_block - 1) / K_block) * K_block;
    if (Np == 0) Np = N_block;
    if (Kp == 0) Kp = K_block;

    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * K_block * sizeof(bf16_t));
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)N_block * K_block * sizeof(bf16_t));

    float *Cp = C; int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    int Kb_blk = K_block / 32;
    for (int kc = 0; kc < Kp; kc += K_block) {
        pack_A_padded(A, lda, 0, kc, Mp, K_block, M, K, bufA);
        for (int nc = 0; nc < Np; nc += N_block) {
            pack_B_padded(B, ldb, kc, nc, K_block, N_block, K, N, bufB);
            for (int mp = 0; mp < Mp; mp += 32) {
                const bf16_t *bufA_top = bufA + (size_t)(mp/16    ) * Kb_blk * 16*32;
                const bf16_t *bufA_bot = bufA + (size_t)(mp/16 + 1) * Kb_blk * 16*32;
                for (int np = 0; np < N_block; np += 48) {
                    const bf16_t *bufB0 = bufB + (size_t)((np+0)/16) * Kb_blk * 16*32;
                    const bf16_t *bufB1 = bufB + (size_t)((np+16)/16) * Kb_blk * 16*32;
                    const bf16_t *bufB2 = bufB + (size_t)((np+32)/16) * Kb_blk * 16*32;
                    kernel_tilingB_32x48(&Cp[mp*ldcp + nc + np], ldcp,
                                         bufA_top, bufA_bot,
                                         bufB0, bufB1, bufB2,
                                         Kb_blk);
                }
            }
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        free(Cp);
    }
    free(bufA); free(bufB);
    amx_release();
}

// ============================================================
// v12: Base-kernel + 可変 K_block + Kb 特殊化
//   v3/v9 の組合せ。Base-kernel (32x32 出力) を採用しつつ、
//   v9 と同様に Kb をコンパイル時定数として実体化。
// ============================================================
void gemm_v12_base_kbspec(int M, int N, int K,
                          const bf16_t *A, int lda,
                          const bf16_t *B, int ldb,
                          float *C, int ldc,
                          int K_block, int N_block)
{
    amx_config_8tiles();
    int Mp = (M + 31) & ~31;
    if (K_block <= 0) K_block = 1024;
    if (N_block <= 0) N_block = 640;
    K_block = (K_block + 31) & ~31;
    N_block = (N_block + 31) & ~31;
    if (K_block < 32) K_block = 32;
    if (N_block < 32) N_block = 32;
    int K_block_max = K_block;
    int Np = ((N + N_block - 1) / N_block) * N_block;
    int Kp = ((K + 31) & ~31);
    if (Np == 0) Np = N_block;
    if (Kp == 0) Kp = 32;

    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * K_block_max * sizeof(bf16_t));
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)N_block * K_block_max * sizeof(bf16_t));

    float *Cp = C; int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    for (int kc = 0; kc < Kp; kc += K_block_max) {
        int K_blk_cur = K_block_max;
        if (kc + K_blk_cur > Kp) K_blk_cur = Kp - kc;
        int Kb_blk = K_blk_cur / 32;

        pack_A_padded(A, lda, 0, kc, Mp, K_blk_cur, M, K, bufA);
        for (int nc = 0; nc < Np; nc += N_block) {
            pack_B_padded(B, ldb, kc, nc, K_blk_cur, N_block, K, N, bufB);
            for (int mp = 0; mp < Mp; mp += 32) {
                const bf16_t *bufA0 = bufA + (size_t)(mp/16    ) * Kb_blk * 16*32;
                const bf16_t *bufA1 = bufA + (size_t)(mp/16 + 1) * Kb_blk * 16*32;
                for (int np = 0; np < N_block; np += 32) {
                    const bf16_t *bufB0 = bufB + (size_t)(np/16    ) * Kb_blk * 16*32;
                    const bf16_t *bufB1 = bufB + (size_t)(np/16 + 1) * Kb_blk * 16*32;
                    kernel_base_32x32_dispatch(&Cp[mp*ldcp + nc + np], ldcp,
                                               bufA0, bufA1, bufB0, bufB1,
                                               Kb_blk);
                }
            }
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        free(Cp);
    }
    free(bufA); free(bufB);
    amx_release();
}

// ============================================================
// v11: v9 + 改良 M-panel ブロッキング
//   設計: bufB を 1 度だけ (kc, nc) ごとにパック (v9 と同じ)。
//          mp_panel は nc の INSIDE で、bufB を共有する。
//   ループ: kc → nc → mp_panel → mp → np
//   bufA panel が L2 で温まることを期待。
// ============================================================
void gemm_v11_tilingB_mpanel(int M, int N, int K,
                             const bf16_t *A, int lda,
                             const bf16_t *B, int ldb,
                             float *C, int ldc,
                             int K_block, int N_block, int M_panel)
{
    amx_config_8tiles();
    int Mp = (M + 31) & ~31;
    if (K_block <= 0) K_block = 1280;
    if (N_block <= 0) N_block = 528;
    if (M_panel <= 0) M_panel = 1024;
    K_block = (K_block + 31) & ~31;
    if (N_block < 48) N_block = 48;
    N_block = ((N_block + 47) / 48) * 48;
    if (K_block < 32) K_block = 32;
    if (M_panel < 32) M_panel = 32;
    M_panel = (M_panel + 31) & ~31;
    int K_block_max = K_block;
    int Np = ((N + N_block - 1) / N_block) * N_block;
    int Kp = ((K + 31) & ~31);
    if (Np == 0) Np = N_block;
    if (Kp == 0) Kp = 32;
    if (M_panel > Mp) M_panel = Mp;

    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * K_block_max * sizeof(bf16_t));
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)N_block * K_block_max * sizeof(bf16_t));

    float *Cp = C; int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    for (int kc = 0; kc < Kp; kc += K_block_max) {
        int K_blk_cur = K_block_max;
        if (kc + K_blk_cur > Kp) K_blk_cur = Kp - kc;
        int Kb_blk = K_blk_cur / 32;

        pack_A_padded(A, lda, 0, kc, Mp, K_blk_cur, M, K, bufA);

        for (int nc = 0; nc < Np; nc += N_block) {
            pack_B_padded(B, ldb, kc, nc, K_blk_cur, N_block, K, N, bufB);
            // mp_panel ループは nc の INSIDE。同じ bufB を共有する。
            for (int mp0 = 0; mp0 < Mp; mp0 += M_panel) {
                int M_blk_cur = M_panel;
                if (mp0 + M_blk_cur > Mp) M_blk_cur = Mp - mp0;
                for (int mp = mp0; mp < mp0 + M_blk_cur; mp += 32) {
                    const bf16_t *bufA_top = bufA + (size_t)(mp/16    ) * Kb_blk * 16*32;
                    const bf16_t *bufA_bot = bufA + (size_t)(mp/16 + 1) * Kb_blk * 16*32;
                    for (int np = 0; np < N_block; np += 48) {
                        const bf16_t *bufB0 = bufB + (size_t)((np+0)/16) * Kb_blk * 16*32;
                        const bf16_t *bufB1 = bufB + (size_t)((np+16)/16) * Kb_blk * 16*32;
                        const bf16_t *bufB2 = bufB + (size_t)((np+32)/16) * Kb_blk * 16*32;
                        kernel_tilingB_32x48_dispatch(&Cp[mp*ldcp + nc + np], ldcp,
                                                      bufA_top, bufA_bot,
                                                      bufB0, bufB1, bufB2,
                                                      Kb_blk);
                    }
                }
            }
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        free(Cp);
    }
    free(bufA); free(bufB);
    amx_release();
}

// ============================================================
// v10: v9 + M_panel ブロッキング (L2 内に A panel 常駐)
//   ループ: kc → mp_panel → nc → mp → np
//   A: 全体 (M × K_block) を一度パック → L3 にある状態
//   M_panel 範囲 (M_panel × K_block) を L2 に常駐させて、複数の nc で再利用
// ============================================================
void gemm_v10_tilingB_mpanel(int M, int N, int K,
                             const bf16_t *A, int lda,
                             const bf16_t *B, int ldb,
                             float *C, int ldc,
                             int K_block, int N_block, int M_panel)
{
    amx_config_8tiles();
    int Mp = (M + 31) & ~31;
    if (K_block <= 0) K_block = 1280;
    if (N_block <= 0) N_block = 528;
    if (M_panel <= 0) M_panel = 256;
    K_block = (K_block + 31) & ~31;
    if (N_block < 48) N_block = 48;
    N_block = ((N_block + 47) / 48) * 48;
    if (K_block < 32) K_block = 32;
    if (M_panel < 32) M_panel = 32;
    M_panel = (M_panel + 31) & ~31;  // 32 倍数
    int K_block_max = K_block;
    int Np = ((N + N_block - 1) / N_block) * N_block;
    int Kp = ((K + 31) & ~31);
    if (Np == 0) Np = N_block;
    if (Kp == 0) Kp = 32;
    int Mp_panel = Mp;
    if (M_panel > Mp_panel) M_panel = Mp_panel;

    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * K_block_max * sizeof(bf16_t));
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)N_block * K_block_max * sizeof(bf16_t));

    float *Cp = C; int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    for (int kc = 0; kc < Kp; kc += K_block_max) {
        int K_blk_cur = K_block_max;
        if (kc + K_blk_cur > Kp) K_blk_cur = Kp - kc;
        int Kb_blk = K_blk_cur / 32;

        // bufA 全体 (M × K_blk_cur) を 1 度パック
        pack_A_padded(A, lda, 0, kc, Mp, K_blk_cur, M, K, bufA);

        for (int mp0 = 0; mp0 < Mp; mp0 += M_panel) {
            int M_blk_cur = M_panel;
            if (mp0 + M_blk_cur > Mp) M_blk_cur = Mp - mp0;

            for (int nc = 0; nc < Np; nc += N_block) {
                pack_B_padded(B, ldb, kc, nc, K_blk_cur, N_block, K, N, bufB);
                for (int mp = mp0; mp < mp0 + M_blk_cur; mp += 32) {
                    const bf16_t *bufA_top = bufA + (size_t)(mp/16    ) * Kb_blk * 16*32;
                    const bf16_t *bufA_bot = bufA + (size_t)(mp/16 + 1) * Kb_blk * 16*32;
                    for (int np = 0; np < N_block; np += 48) {
                        const bf16_t *bufB0 = bufB + (size_t)((np+0)/16) * Kb_blk * 16*32;
                        const bf16_t *bufB1 = bufB + (size_t)((np+16)/16) * Kb_blk * 16*32;
                        const bf16_t *bufB2 = bufB + (size_t)((np+32)/16) * Kb_blk * 16*32;
                        kernel_tilingB_32x48_dispatch(&Cp[mp*ldcp + nc + np], ldcp,
                                                      bufA_top, bufA_bot,
                                                      bufB0, bufB1, bufB2,
                                                      Kb_blk);
                    }
                }
            }
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        free(Cp);
    }
    free(bufA); free(bufB);
    amx_release();
}

// ============================================================
// v9: v8 + カーネル特殊化 (Kb 定数化)
//   v8 と同じループ構造だが、kernel_tilingB_32x48_dispatch を経由して
//   コンパイル時定数 Kb の特殊化版を選ぶ。
// ============================================================
void gemm_v9_tilingB_kbspec(int M, int N, int K,
                            const bf16_t *A, int lda,
                            const bf16_t *B, int ldb,
                            float *C, int ldc,
                            int K_block, int N_block)
{
    amx_config_8tiles();
    int Mp = (M + 31) & ~31;
    if (K_block <= 0) K_block = 1280;
    if (N_block <= 0) N_block = 528;
    K_block = (K_block + 31) & ~31;
    if (N_block < 48) N_block = 48;
    N_block = ((N_block + 47) / 48) * 48;
    if (K_block < 32) K_block = 32;
    int K_block_max = K_block;
    int Np = ((N + N_block - 1) / N_block) * N_block;
    int Kp = ((K + 31) & ~31);
    if (Np == 0) Np = N_block;
    if (Kp == 0) Kp = 32;

    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * K_block_max * sizeof(bf16_t));
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)N_block * K_block_max * sizeof(bf16_t));

    float *Cp = C; int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    for (int kc = 0; kc < Kp; kc += K_block_max) {
        int K_blk_cur = K_block_max;
        if (kc + K_blk_cur > Kp) K_blk_cur = Kp - kc;
        int Kb_blk = K_blk_cur / 32;

        pack_A_padded(A, lda, 0, kc, Mp, K_blk_cur, M, K, bufA);
        for (int nc = 0; nc < Np; nc += N_block) {
            pack_B_padded(B, ldb, kc, nc, K_blk_cur, N_block, K, N, bufB);
            for (int mp = 0; mp < Mp; mp += 32) {
                const bf16_t *bufA_top = bufA + (size_t)(mp/16    ) * Kb_blk * 16*32;
                const bf16_t *bufA_bot = bufA + (size_t)(mp/16 + 1) * Kb_blk * 16*32;
                for (int np = 0; np < N_block; np += 48) {
                    const bf16_t *bufB0 = bufB + (size_t)((np+0)/16) * Kb_blk * 16*32;
                    const bf16_t *bufB1 = bufB + (size_t)((np+16)/16) * Kb_blk * 16*32;
                    const bf16_t *bufB2 = bufB + (size_t)((np+32)/16) * Kb_blk * 16*32;
                    kernel_tilingB_32x48_dispatch(&Cp[mp*ldcp + nc + np], ldcp,
                                                  bufA_top, bufA_bot,
                                                  bufB0, bufB1, bufB2,
                                                  Kb_blk);
                }
            }
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        free(Cp);
    }
    free(bufA); free(bufB);
    amx_release();
}

// ============================================================
// v8: Tiling_B + 可変 K_block
//   最後の kc 反復のみ、K の残りに合わせ K_block を 32 倍数で縮める。
//   これにより K = 10000, K_block = 1024 のとき Kp が 10240 → 10016 になる。
// ============================================================
void gemm_v8_tilingB_varkblock(int M, int N, int K,
                               const bf16_t *A, int lda,
                               const bf16_t *B, int ldb,
                               float *C, int ldc,
                               int K_block, int N_block)
{
    amx_config_8tiles();
    int Mp = (M + 31) & ~31;
    if (K_block <= 0) K_block = 1024;
    if (N_block <= 0) N_block = 480;
    K_block = (K_block + 31) & ~31;
    if (N_block < 48) N_block = 48;
    N_block = ((N_block + 47) / 48) * 48;
    if (K_block < 32) K_block = 32;
    int Np = ((N + N_block - 1) / N_block) * N_block;
    // 可変 K: 最終 kc panel のみ短くする
    int K_block_max = K_block;
    int Kp = ((K + 31) & ~31);  // 32 倍数まで切上げ (1024 は不要)
    if (Np == 0) Np = N_block;
    if (Kp == 0) Kp = 32;

    // バッファ確保: 最大 K_block サイズ
    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * K_block_max * sizeof(bf16_t));
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)N_block * K_block_max * sizeof(bf16_t));

    float *Cp = C; int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    for (int kc = 0; kc < Kp; kc += K_block_max) {
        int K_blk_cur = K_block_max;
        if (kc + K_blk_cur > Kp) K_blk_cur = Kp - kc;
        // K_blk_cur は 32 倍数 (Kp が 32 倍数なので保証)
        int Kb_blk = K_blk_cur / 32;

        pack_A_padded(A, lda, 0, kc, Mp, K_blk_cur, M, K, bufA);
        for (int nc = 0; nc < Np; nc += N_block) {
            pack_B_padded(B, ldb, kc, nc, K_blk_cur, N_block, K, N, bufB);
            for (int mp = 0; mp < Mp; mp += 32) {
                const bf16_t *bufA_top = bufA + (size_t)(mp/16    ) * Kb_blk * 16*32;
                const bf16_t *bufA_bot = bufA + (size_t)(mp/16 + 1) * Kb_blk * 16*32;
                for (int np = 0; np < N_block; np += 48) {
                    const bf16_t *bufB0 = bufB + (size_t)((np+0)/16) * Kb_blk * 16*32;
                    const bf16_t *bufB1 = bufB + (size_t)((np+16)/16) * Kb_blk * 16*32;
                    const bf16_t *bufB2 = bufB + (size_t)((np+32)/16) * Kb_blk * 16*32;
                    kernel_tilingB_32x48(&Cp[mp*ldcp + nc + np], ldcp,
                                         bufA_top, bufA_bot,
                                         bufB0, bufB1, bufB2,
                                         Kb_blk);
                }
            }
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        free(Cp);
    }
    free(bufA); free(bufB);
    amx_release();
}

// ============================================================
// v7: Tiling_B + kc 反転 + C タイル kc 跨ぎ常駐
//   A 全体を一度パック (size = Mp × Kp * 2 byte; 10000^2 で 約 200 MB)
//   B は nc 毎にパック (size = K_pad × N_block * 2 byte)
//   ループ: nc → mp → np → load C → kc → kernel_acc → store C
//   C タイルの load/store 回数を K_block 倍数だけ削減できる。
// ============================================================
void gemm_v7_tilingB_cresid(int M, int N, int K,
                            const bf16_t *A, int lda,
                            const bf16_t *B, int ldb,
                            float *C, int ldc,
                            int K_block, int N_block)
{
    amx_config_8tiles();
    int Mp = (M + 31) & ~31;
    if (K_block <= 0) K_block = 1024;
    if (N_block <= 0) N_block = 480;
    K_block = (K_block + 31) & ~31;
    if (N_block < 48) N_block = 48;
    N_block = ((N_block + 47) / 48) * 48;
    if (K_block < 32) K_block = 32;
    int Np = ((N + N_block - 1) / N_block) * N_block;
    int Kp = ((K + K_block - 1) / K_block) * K_block;
    if (Np == 0) Np = N_block;
    if (Kp == 0) Kp = K_block;

    // bufA は 全 M × K_pad をまとめてパック (mp_panel ごと連続)
    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * Kp * sizeof(bf16_t));
    // 全 K について一回でパック (kc ループの中で必要に応じて A の異なる kc panel を読み出す)
    {
        int Kb_full = Kp / 32;
        // A 全体を 1 つの大きなパネル (M_panel = Mp, K_block = Kp) として pack
        pack_A_padded(A, lda, 0, 0, Mp, Kp, M, K, bufA);
        (void)Kb_full;
    }

    // bufB は (K_pad × N_block) を nc 毎にパック
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)Kp * N_block * sizeof(bf16_t));

    float *Cp = C; int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    int Kb_full = Kp / 32;
    int Kb_blk = K_block / 32;
    int K_blocks = Kp / K_block;
    (void)K_blocks;

    for (int nc = 0; nc < Np; nc += N_block) {
        // bufB を 全 Kp × N_block パック
        // pack_B_padded のシグネチャに合わせ、K_block=Kp として一度に詰める
        pack_B_padded(B, ldb, 0, nc, Kp, N_block, K, N, bufB);
        for (int mp = 0; mp < Mp; mp += 32) {
            // A のこの mp パネル先頭 (全 Kp 範囲)
            const bf16_t *A_top0 = bufA + (size_t)(mp/16    ) * Kb_full * 16*32;
            const bf16_t *A_bot0 = bufA + (size_t)(mp/16 + 1) * Kb_full * 16*32;
            for (int np = 0; np < N_block; np += 48) {
                // C のこの (mp, nc+np) ブロックを 6 タイルにロード
                float *Cbase = &Cp[mp*ldcp + nc + np];
                _tile_loadd(0, Cbase + 0*ldcp + 0,    ldcp * sizeof(float));
                _tile_loadd(1, Cbase + 0*ldcp + 16,   ldcp * sizeof(float));
                _tile_loadd(2, Cbase + 0*ldcp + 32,   ldcp * sizeof(float));
                _tile_loadd(3, Cbase + 16*ldcp + 0,   ldcp * sizeof(float));
                _tile_loadd(4, Cbase + 16*ldcp + 16,  ldcp * sizeof(float));
                _tile_loadd(5, Cbase + 16*ldcp + 32,  ldcp * sizeof(float));

                // kc 反復: A の各 kc 部分パネル、B の各 kc 部分を使う
                for (int kc = 0; kc < Kp; kc += K_block) {
                    int kc_panel_off = (kc / 32) * 16 * 32;  // bufA: 16x32 unit 数
                    const bf16_t *A_top = A_top0 + kc_panel_off;
                    const bf16_t *A_bot = A_bot0 + kc_panel_off;
                    // bufB layout: [n_block][kc_panel][16x32]
                    // n_block 内の (np+0)/16, (np+16)/16, (np+32)/16 列で kc_panel をずらす
                    int Kb_in_bufB = Kp / 32;
                    const bf16_t *B0 = bufB + (size_t)((np+0)/16) * Kb_in_bufB * 16*32 + kc_panel_off;
                    const bf16_t *B1 = bufB + (size_t)((np+16)/16) * Kb_in_bufB * 16*32 + kc_panel_off;
                    const bf16_t *B2 = bufB + (size_t)((np+32)/16) * Kb_in_bufB * 16*32 + kc_panel_off;
                    kernel_tilingB_32x48_acc(A_top, A_bot, B0, B1, B2, Kb_blk);
                }

                _tile_stored(0, Cbase + 0*ldcp + 0,    ldcp * sizeof(float));
                _tile_stored(1, Cbase + 0*ldcp + 16,   ldcp * sizeof(float));
                _tile_stored(2, Cbase + 0*ldcp + 32,   ldcp * sizeof(float));
                _tile_stored(3, Cbase + 16*ldcp + 0,   ldcp * sizeof(float));
                _tile_stored(4, Cbase + 16*ldcp + 16,  ldcp * sizeof(float));
                _tile_stored(5, Cbase + 16*ldcp + 32,  ldcp * sizeof(float));
            }
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        free(Cp);
    }
    free(bufA); free(bufB);
    amx_release();
}

// ============================================================
// v6: Tiling_B + ブロッキング + 軽量プリフェッチ
// ============================================================
void gemm_v6_tilingB_pf(int M, int N, int K,
                        const bf16_t *A, int lda,
                        const bf16_t *B, int ldb,
                        float *C, int ldc,
                        int K_block, int N_block)
{
    amx_config_8tiles();
    int Mp = (M + 31) & ~31;
    if (K_block <= 0) K_block = 1536;
    if (N_block <= 0) N_block = 480;
    K_block = (K_block + 31) & ~31;
    // N_block は 48 (Tiling_B のカーネル幅) と 16 (B のパック単位) の LCM = 48 の倍数
    if (N_block < 48) N_block = 48;
    N_block = ((N_block + 47) / 48) * 48;
    if (K_block < 32) K_block = 32;
    int Np = ((N + N_block - 1) / N_block) * N_block;
    int Kp = ((K + K_block - 1) / K_block) * K_block;
    if (Np == 0) Np = N_block;
    if (Kp == 0) Kp = K_block;

    bf16_t *bufA = (bf16_t *)xaligned_alloc((size_t)Mp * K_block * sizeof(bf16_t));
    bf16_t *bufB = (bf16_t *)xaligned_alloc((size_t)N_block * K_block * sizeof(bf16_t));

    float *Cp = C; int ldcp = ldc;
    int need_pad = (M != Mp) || (N != Np);
    if (need_pad) {
        Cp = (float *)xaligned_alloc((size_t)Mp * Np * sizeof(float));
        ldcp = Np;
        for (int i = 0; i < Mp; ++i) {
            if (i < M) {
                memcpy(&Cp[i*Np], &C[i*ldc], N * sizeof(float));
                memset(&Cp[i*Np + N], 0, (Np - N) * sizeof(float));
            } else {
                memset(&Cp[i*Np], 0, Np * sizeof(float));
            }
        }
    }

    int Kb_blk = K_block / 32;
    for (int kc = 0; kc < Kp; kc += K_block) {
        pack_A_padded(A, lda, 0, kc, Mp, K_block, M, K, bufA);
        for (int nc = 0; nc < Np; nc += N_block) {
            pack_B_padded(B, ldb, kc, nc, K_block, N_block, K, N, bufB);
            for (int mp = 0; mp < Mp; mp += 32) {
                const bf16_t *bufA_top = bufA + (size_t)(mp/16    ) * Kb_blk * 16*32;
                const bf16_t *bufA_bot = bufA + (size_t)(mp/16 + 1) * Kb_blk * 16*32;
                int mp_next = mp + 32;
                const bf16_t *bufA_top_next = NULL, *bufA_bot_next = NULL;
                if (mp_next < Mp) {
                    bufA_top_next = bufA + (size_t)(mp_next/16    ) * Kb_blk * 16*32;
                    bufA_bot_next = bufA + (size_t)(mp_next/16 + 1) * Kb_blk * 16*32;
                }
                for (int np = 0; np < N_block; np += 48) {
                    const bf16_t *bufB0 = bufB + (size_t)((np+0)/16) * Kb_blk * 16*32;
                    const bf16_t *bufB1 = bufB + (size_t)((np+16)/16) * Kb_blk * 16*32;
                    const bf16_t *bufB2 = bufB + (size_t)((np+32)/16) * Kb_blk * 16*32;
                    kernel_tilingB_32x48_pf(&Cp[mp*ldcp + nc + np], ldcp,
                                            bufA_top, bufA_bot,
                                            bufB0, bufB1, bufB2,
                                            bufA_top_next, bufA_bot_next,
                                            Kb_blk);
                }
            }
        }
    }

    if (need_pad) {
        for (int i = 0; i < M; ++i) memcpy(&C[i*ldc], &Cp[i*Np], N * sizeof(float));
        free(Cp);
    }
    free(bufA); free(bufB);
    amx_release();
}
