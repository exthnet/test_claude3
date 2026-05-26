// amx_kernels.h: AMX マイクロカーネル群
#ifndef AMX_KERNELS_H
#define AMX_KERNELS_H

#include "common.h"

// ============================================================
// AMX タイル設定
// 8 タイルそれぞれ 16 行 × 64 byte。
//   t0..t3: C タイル (16x16 fp32)
//   t4,t5 : A タイル (16x32 bf16)
//   t6,t7 : B タイル (VNNI 16x32 bf16)
// ============================================================
static inline void amx_config_8tiles(void) {
    tilecfg_t cfg = {0};
    cfg.palette_id = 1;
    for (int i = 0; i < 8; ++i) {
        cfg.colsb[i] = 64;
        cfg.rows[i] = 16;
    }
    _tile_loadconfig(&cfg);
}

static inline void amx_release(void) {
    _tile_release();
}

// ============================================================
// パッキング
// ============================================================
// A を BUFFER_A 形式へパック。形状:
//   bufA[m_block][k_block][16 rows][32 cols bf16]  (連続メモリ)
//   m_block = m / 16, k_block = k / 32
//   ストライド 64 byte で _tile_loadd できる。
// 引数:
//   A: 元行列、lda=ストライド (要素数)
//   m_off, k_off: パッキング対象の左上座標
//   M_panel: パネル行数 (16 の倍数)
//   K_block: パネル K サイズ (32 の倍数)
static inline void pack_A_block(const bf16_t *A, int lda,
                                int m_off, int k_off,
                                int M_panel, int K_block,
                                bf16_t *bufA) {
    int Mb = M_panel / 16;
    int Kb = K_block / 32;
    for (int mb = 0; mb < Mb; ++mb) {
        for (int kb = 0; kb < Kb; ++kb) {
            bf16_t *dst = bufA + (mb * Kb + kb) * 16 * 32;
            for (int i = 0; i < 16; ++i) {
                const bf16_t *src = &A[(m_off + mb*16 + i) * lda + (k_off + kb*32)];
                memcpy(dst + i*32, src, 32 * sizeof(bf16_t));
            }
        }
    }
}

// B を BUFFER_B 形式へパック (VNNI)。形状:
//   bufB[n_block][k_block][16 rows of pairs][32 bf16]
//   n_block = n / 16, k_block = k / 32 (= 16 行のペア)
// 中身: 位置 (kp, 2*nc+0) = B[k_off + kb*32 + 2*kp, n_off + nb*16 + nc]
//      位置 (kp, 2*nc+1) = B[k_off + kb*32 + 2*kp + 1, n_off + nb*16 + nc]
static inline void pack_B_block(const bf16_t *B, int ldb,
                                int k_off, int n_off,
                                int K_block, int N_block,
                                bf16_t *bufB) {
    int Nb = N_block / 16;
    int Kb = K_block / 32;
    for (int nb = 0; nb < Nb; ++nb) {
        for (int kb = 0; kb < Kb; ++kb) {
            bf16_t *dst = bufB + (nb * Kb + kb) * 16 * 32;
            for (int kp = 0; kp < 16; ++kp) {
                int k0 = k_off + kb*32 + 2*kp;
                int k1 = k0 + 1;
                const bf16_t *r0 = &B[k0 * ldb + n_off + nb*16];
                const bf16_t *r1 = &B[k1 * ldb + n_off + nb*16];
                bf16_t *out = dst + kp * 32;
                for (int nc = 0; nc < 16; ++nc) {
                    out[2*nc + 0] = r0[nc];
                    out[2*nc + 1] = r1[nc];
                }
            }
        }
    }
}

// ============================================================
// Base-kernel (Section 4.2): 32x32 C 出力
// 4 C tiles + 2 A tiles + 2 B tiles を使う。
// ldc は要素数。bufA0/bufA1 は (Kb 個の 16x32 マイクロブロック) の先頭。
// ============================================================
static inline void kernel_base_32x32(float *C, int ldc,
                                     const bf16_t *bufA0,
                                     const bf16_t *bufA1,
                                     const bf16_t *bufB0,
                                     const bf16_t *bufB1,
                                     int Kb)
{
    _tile_loadd(0, C + 0*ldc + 0,    ldc * sizeof(float));
    _tile_loadd(1, C + 0*ldc + 16,   ldc * sizeof(float));
    _tile_loadd(2, C + 16*ldc + 0,   ldc * sizeof(float));
    _tile_loadd(3, C + 16*ldc + 16,  ldc * sizeof(float));
    for (int kb = 0; kb < Kb; ++kb) {
        _tile_loadd(4, bufA0 + (size_t)kb*16*32, 64);
        _tile_loadd(5, bufA1 + (size_t)kb*16*32, 64);
        _tile_loadd(6, bufB0 + (size_t)kb*16*32, 64);
        _tile_loadd(7, bufB1 + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(0, 4, 6);
        _tile_dpbf16ps(1, 4, 7);
        _tile_dpbf16ps(2, 5, 6);
        _tile_dpbf16ps(3, 5, 7);
    }
    _tile_stored(0, C + 0*ldc + 0,    ldc * sizeof(float));
    _tile_stored(1, C + 0*ldc + 16,   ldc * sizeof(float));
    _tile_stored(2, C + 16*ldc + 0,   ldc * sizeof(float));
    _tile_stored(3, C + 16*ldc + 16,  ldc * sizeof(float));
}

// Base-kernel + 軽量プリフェッチ
// kb 内ループ内では、現在の (kb 番目の) A_next スラブのうち、
// 1 〜 2 本だけ T1 プリフェッチを発行する。これで命令オーバーヘッドを抑える。
// 完全なプリフェッチは、別途 m_pair 移行直前に呼ぶ prefetch_panel() に任せる。
static inline void kernel_base_32x32_pf(float *C, int ldc,
                                        const bf16_t *bufA0,
                                        const bf16_t *bufA1,
                                        const bf16_t *bufB0,
                                        const bf16_t *bufB1,
                                        const bf16_t *bufA0_next,
                                        const bf16_t *bufA1_next,
                                        int Kb)
{
    _tile_loadd(0, C + 0*ldc + 0,    ldc * sizeof(float));
    _tile_loadd(1, C + 0*ldc + 16,   ldc * sizeof(float));
    _tile_loadd(2, C + 16*ldc + 0,   ldc * sizeof(float));
    _tile_loadd(3, C + 16*ldc + 16,  ldc * sizeof(float));
    for (int kb = 0; kb < Kb; ++kb) {
        _tile_loadd(4, bufA0 + (size_t)kb*16*32, 64);
        _tile_loadd(5, bufA1 + (size_t)kb*16*32, 64);
        _tile_loadd(6, bufB0 + (size_t)kb*16*32, 64);
        _tile_loadd(7, bufB1 + (size_t)kb*16*32, 64);
        // 1024 byte/A_next slab のうち先頭 128 byte (2 cache line) を毎 kb プリフェッチ
        // → 1 kernel 呼び出しあたり 4 prefetch のみ
        if (bufA0_next) {
            _mm_prefetch((const char *)(bufA0_next + (size_t)kb*16*32),       _MM_HINT_T1);
            _mm_prefetch((const char *)(bufA0_next + (size_t)kb*16*32) + 64,  _MM_HINT_T1);
            _mm_prefetch((const char *)(bufA1_next + (size_t)kb*16*32),       _MM_HINT_T1);
            _mm_prefetch((const char *)(bufA1_next + (size_t)kb*16*32) + 64,  _MM_HINT_T1);
        }
        _tile_dpbf16ps(0, 4, 6);
        _tile_dpbf16ps(1, 4, 7);
        _tile_dpbf16ps(2, 5, 6);
        _tile_dpbf16ps(3, 5, 7);
    }
    _tile_stored(0, C + 0*ldc + 0,    ldc * sizeof(float));
    _tile_stored(1, C + 0*ldc + 16,   ldc * sizeof(float));
    _tile_stored(2, C + 16*ldc + 0,   ldc * sizeof(float));
    _tile_stored(3, C + 16*ldc + 16,  ldc * sizeof(float));
}

// バルクプリフェッチ: m_pair 移行直前に、A_next 全体 (32 行 × K_block 列)
// を 64 byte ストライドで T1 プリフェッチする。
static inline void prefetch_A_next(const bf16_t *bufA0_next,
                                   const bf16_t *bufA1_next,
                                   int Kb)
{
    const char *p0 = (const char *)bufA0_next;
    const char *p1 = (const char *)bufA1_next;
    size_t bytes = (size_t)Kb * 16 * 32 * 2;
    for (size_t o = 0; o < bytes; o += 64) {
        _mm_prefetch(p0 + o, _MM_HINT_T1);
        _mm_prefetch(p1 + o, _MM_HINT_T1);
    }
}

// ============================================================
// Tiling_B カーネル (論文 Sec 5.2.2): 32 行 x 48 列 C 出力
//   C タイル 6 枚 (R0..R5): C[2m,2n], C[2m,2n+1], C[2m,2n+2],
//                            C[2m+1,2n], C[2m+1,2n+1], C[2m+1,2n+2]
//   A タイル 1 枚 (R6): A[2m,k], A[2m+1,k] を交互ロード
//   B タイル 1 枚 (R7): B[k,2n], B[k,2n+1], B[k,2n+2] を交互ロード
//
//   ステップ:
//     (1) R6=A[2m,k],   R7=B[k,2n]    -> R0  (C[2m,2n])
//     (2)               R7=B[k,2n+1]  -> R1  (C[2m,2n+1])
//     (3)               R7=B[k,2n+2]  -> R2  (C[2m,2n+2])
//     (4) R6=A[2m+1,k]                -> R5  (C[2m+1,2n+2])
//     (5)               R7=B[k,2n]    -> R3  (C[2m+1,2n])
//     (6)               R7=B[k,2n+1]  -> R4  (C[2m+1,2n+1])
//
// ldc は要素数。bufA_top/bot, bufB0/1/2 はそれぞれ Kb 個の (16x32) マイクロブロック先頭。
// ============================================================
static inline void kernel_tilingB_32x48(float *C, int ldc,
                                        const bf16_t *bufA_top,
                                        const bf16_t *bufA_bot,
                                        const bf16_t *bufB0,
                                        const bf16_t *bufB1,
                                        const bf16_t *bufB2,
                                        int Kb)
{
    _tile_loadd(0, C + 0*ldc + 0,    ldc * sizeof(float));
    _tile_loadd(1, C + 0*ldc + 16,   ldc * sizeof(float));
    _tile_loadd(2, C + 0*ldc + 32,   ldc * sizeof(float));
    _tile_loadd(3, C + 16*ldc + 0,   ldc * sizeof(float));
    _tile_loadd(4, C + 16*ldc + 16,  ldc * sizeof(float));
    _tile_loadd(5, C + 16*ldc + 32,  ldc * sizeof(float));
    for (int kb = 0; kb < Kb; ++kb) {
        _tile_loadd(6, bufA_top + (size_t)kb*16*32, 64);
        _tile_loadd(7, bufB0   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(0, 6, 7);
        _tile_loadd(7, bufB1   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(1, 6, 7);
        _tile_loadd(7, bufB2   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(2, 6, 7);
        _tile_loadd(6, bufA_bot + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(5, 6, 7);   // C[2m+1, 2n+2] (B2 still in R7)
        _tile_loadd(7, bufB0   + (size_t)kb*16*32, 64);  // L1 hit 期待
        _tile_dpbf16ps(3, 6, 7);
        _tile_loadd(7, bufB1   + (size_t)kb*16*32, 64);  // L1 hit 期待
        _tile_dpbf16ps(4, 6, 7);
    }
    _tile_stored(0, C + 0*ldc + 0,    ldc * sizeof(float));
    _tile_stored(1, C + 0*ldc + 16,   ldc * sizeof(float));
    _tile_stored(2, C + 0*ldc + 32,   ldc * sizeof(float));
    _tile_stored(3, C + 16*ldc + 0,   ldc * sizeof(float));
    _tile_stored(4, C + 16*ldc + 16,  ldc * sizeof(float));
    _tile_stored(5, C + 16*ldc + 32,  ldc * sizeof(float));
}

// Tiling_B カーネル: C タイルロード/ストアを行わない版
// (C は呼び出し側で事前に t0..t5 にロード済みと仮定)
// kc ループの内側で呼び、kc 跨ぎで C タイルを保持する用途。
static inline void kernel_tilingB_32x48_acc(const bf16_t *bufA_top,
                                            const bf16_t *bufA_bot,
                                            const bf16_t *bufB0,
                                            const bf16_t *bufB1,
                                            const bf16_t *bufB2,
                                            int Kb)
{
    for (int kb = 0; kb < Kb; ++kb) {
        _tile_loadd(6, bufA_top + (size_t)kb*16*32, 64);
        _tile_loadd(7, bufB0   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(0, 6, 7);
        _tile_loadd(7, bufB1   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(1, 6, 7);
        _tile_loadd(7, bufB2   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(2, 6, 7);
        _tile_loadd(6, bufA_bot + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(5, 6, 7);
        _tile_loadd(7, bufB0   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(3, 6, 7);
        _tile_loadd(7, bufB1   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(4, 6, 7);
    }
}

// Base-kernel: C 持ち越し版 (32x32, 4 C tiles in t0..t3)
static inline void kernel_base_32x32_acc(const bf16_t *bufA0,
                                         const bf16_t *bufA1,
                                         const bf16_t *bufB0,
                                         const bf16_t *bufB1,
                                         int Kb)
{
    for (int kb = 0; kb < Kb; ++kb) {
        _tile_loadd(4, bufA0 + (size_t)kb*16*32, 64);
        _tile_loadd(5, bufA1 + (size_t)kb*16*32, 64);
        _tile_loadd(6, bufB0 + (size_t)kb*16*32, 64);
        _tile_loadd(7, bufB1 + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(0, 4, 6);
        _tile_dpbf16ps(1, 4, 7);
        _tile_dpbf16ps(2, 5, 6);
        _tile_dpbf16ps(3, 5, 7);
    }
}

// ============================================================
// Tiling_B カーネル: Kb をコンパイル時定数として与え、完全アンロールを促す
// gcc の -funroll-loops でも変動 Kb では unroll が控えめになるので、
// macro でテンプレート化して Kb を定数化する。
// ============================================================
#define DEFINE_TILINGB_KB(KB)                                                  \
static inline __attribute__((always_inline))                                   \
void kernel_tilingB_32x48_kb##KB(float *C, int ldc,                            \
                                 const bf16_t *bufA_top,                       \
                                 const bf16_t *bufA_bot,                       \
                                 const bf16_t *bufB0,                          \
                                 const bf16_t *bufB1,                          \
                                 const bf16_t *bufB2)                          \
{                                                                              \
    _tile_loadd(0, C + 0*ldc + 0,    ldc * sizeof(float));                     \
    _tile_loadd(1, C + 0*ldc + 16,   ldc * sizeof(float));                     \
    _tile_loadd(2, C + 0*ldc + 32,   ldc * sizeof(float));                     \
    _tile_loadd(3, C + 16*ldc + 0,   ldc * sizeof(float));                     \
    _tile_loadd(4, C + 16*ldc + 16,  ldc * sizeof(float));                     \
    _tile_loadd(5, C + 16*ldc + 32,  ldc * sizeof(float));                     \
    _Pragma("GCC unroll 4")                                                    \
    for (int kb = 0; kb < (KB); ++kb) {                                        \
        _tile_loadd(6, bufA_top + (size_t)kb*16*32, 64);                       \
        _tile_loadd(7, bufB0   + (size_t)kb*16*32, 64);                        \
        _tile_dpbf16ps(0, 6, 7);                                               \
        _tile_loadd(7, bufB1   + (size_t)kb*16*32, 64);                        \
        _tile_dpbf16ps(1, 6, 7);                                               \
        _tile_loadd(7, bufB2   + (size_t)kb*16*32, 64);                        \
        _tile_dpbf16ps(2, 6, 7);                                               \
        _tile_loadd(6, bufA_bot + (size_t)kb*16*32, 64);                       \
        _tile_dpbf16ps(5, 6, 7);                                               \
        _tile_loadd(7, bufB0   + (size_t)kb*16*32, 64);                        \
        _tile_dpbf16ps(3, 6, 7);                                               \
        _tile_loadd(7, bufB1   + (size_t)kb*16*32, 64);                        \
        _tile_dpbf16ps(4, 6, 7);                                               \
    }                                                                          \
    _tile_stored(0, C + 0*ldc + 0,    ldc * sizeof(float));                    \
    _tile_stored(1, C + 0*ldc + 16,   ldc * sizeof(float));                    \
    _tile_stored(2, C + 0*ldc + 32,   ldc * sizeof(float));                    \
    _tile_stored(3, C + 16*ldc + 0,   ldc * sizeof(float));                    \
    _tile_stored(4, C + 16*ldc + 16,  ldc * sizeof(float));                    \
    _tile_stored(5, C + 16*ldc + 32,  ldc * sizeof(float));                    \
}

// 良く使われる Kb 値を具体化 (kb 24, 25, 32, 33, 40, 48, 56, 60, 64)
DEFINE_TILINGB_KB(24)
DEFINE_TILINGB_KB(25)
DEFINE_TILINGB_KB(32)
DEFINE_TILINGB_KB(33)
DEFINE_TILINGB_KB(40)
DEFINE_TILINGB_KB(48)
DEFINE_TILINGB_KB(56)
DEFINE_TILINGB_KB(60)
DEFINE_TILINGB_KB(64)

// ============================================================
// Base-kernel: Kb をコンパイル時定数として特殊化
// ============================================================
#define DEFINE_BASE_KB(KB)                                                     \
static inline __attribute__((always_inline))                                   \
void kernel_base_32x32_kb##KB(float *C, int ldc,                               \
                              const bf16_t *bufA0,                             \
                              const bf16_t *bufA1,                             \
                              const bf16_t *bufB0,                             \
                              const bf16_t *bufB1)                             \
{                                                                              \
    _tile_loadd(0, C + 0*ldc + 0,    ldc * sizeof(float));                     \
    _tile_loadd(1, C + 0*ldc + 16,   ldc * sizeof(float));                     \
    _tile_loadd(2, C + 16*ldc + 0,   ldc * sizeof(float));                     \
    _tile_loadd(3, C + 16*ldc + 16,  ldc * sizeof(float));                     \
    _Pragma("GCC unroll 4")                                                    \
    for (int kb = 0; kb < (KB); ++kb) {                                        \
        _tile_loadd(4, bufA0 + (size_t)kb*16*32, 64);                          \
        _tile_loadd(5, bufA1 + (size_t)kb*16*32, 64);                          \
        _tile_loadd(6, bufB0 + (size_t)kb*16*32, 64);                          \
        _tile_loadd(7, bufB1 + (size_t)kb*16*32, 64);                          \
        _tile_dpbf16ps(0, 4, 6);                                               \
        _tile_dpbf16ps(1, 4, 7);                                               \
        _tile_dpbf16ps(2, 5, 6);                                               \
        _tile_dpbf16ps(3, 5, 7);                                               \
    }                                                                          \
    _tile_stored(0, C + 0*ldc + 0,    ldc * sizeof(float));                    \
    _tile_stored(1, C + 0*ldc + 16,   ldc * sizeof(float));                    \
    _tile_stored(2, C + 16*ldc + 0,   ldc * sizeof(float));                    \
    _tile_stored(3, C + 16*ldc + 16,  ldc * sizeof(float));                    \
}

DEFINE_BASE_KB(24)
DEFINE_BASE_KB(25)
DEFINE_BASE_KB(32)
DEFINE_BASE_KB(33)
DEFINE_BASE_KB(40)
DEFINE_BASE_KB(48)
DEFINE_BASE_KB(56)
DEFINE_BASE_KB(60)
DEFINE_BASE_KB(64)

static inline void kernel_base_32x32_dispatch(float *C, int ldc,
                                              const bf16_t *bufA0,
                                              const bf16_t *bufA1,
                                              const bf16_t *bufB0,
                                              const bf16_t *bufB1,
                                              int Kb)
{
    switch (Kb) {
    case 24: kernel_base_32x32_kb24(C, ldc, bufA0, bufA1, bufB0, bufB1); break;
    case 25: kernel_base_32x32_kb25(C, ldc, bufA0, bufA1, bufB0, bufB1); break;
    case 32: kernel_base_32x32_kb32(C, ldc, bufA0, bufA1, bufB0, bufB1); break;
    case 33: kernel_base_32x32_kb33(C, ldc, bufA0, bufA1, bufB0, bufB1); break;
    case 40: kernel_base_32x32_kb40(C, ldc, bufA0, bufA1, bufB0, bufB1); break;
    case 48: kernel_base_32x32_kb48(C, ldc, bufA0, bufA1, bufB0, bufB1); break;
    case 56: kernel_base_32x32_kb56(C, ldc, bufA0, bufA1, bufB0, bufB1); break;
    case 60: kernel_base_32x32_kb60(C, ldc, bufA0, bufA1, bufB0, bufB1); break;
    case 64: kernel_base_32x32_kb64(C, ldc, bufA0, bufA1, bufB0, bufB1); break;
    default: kernel_base_32x32(C, ldc, bufA0, bufA1, bufB0, bufB1, Kb); break;
    }
}

// Dispatcher: 既知の Kb なら特殊化版を呼ぶ。それ以外は generic kernel_tilingB_32x48 にフォールバック
static inline void kernel_tilingB_32x48_dispatch(float *C, int ldc,
                                                 const bf16_t *bufA_top,
                                                 const bf16_t *bufA_bot,
                                                 const bf16_t *bufB0,
                                                 const bf16_t *bufB1,
                                                 const bf16_t *bufB2,
                                                 int Kb)
{
    switch (Kb) {
    case 24: kernel_tilingB_32x48_kb24(C, ldc, bufA_top, bufA_bot, bufB0, bufB1, bufB2); break;
    case 25: kernel_tilingB_32x48_kb25(C, ldc, bufA_top, bufA_bot, bufB0, bufB1, bufB2); break;
    case 32: kernel_tilingB_32x48_kb32(C, ldc, bufA_top, bufA_bot, bufB0, bufB1, bufB2); break;
    case 33: kernel_tilingB_32x48_kb33(C, ldc, bufA_top, bufA_bot, bufB0, bufB1, bufB2); break;
    case 40: kernel_tilingB_32x48_kb40(C, ldc, bufA_top, bufA_bot, bufB0, bufB1, bufB2); break;
    case 48: kernel_tilingB_32x48_kb48(C, ldc, bufA_top, bufA_bot, bufB0, bufB1, bufB2); break;
    case 56: kernel_tilingB_32x48_kb56(C, ldc, bufA_top, bufA_bot, bufB0, bufB1, bufB2); break;
    case 60: kernel_tilingB_32x48_kb60(C, ldc, bufA_top, bufA_bot, bufB0, bufB1, bufB2); break;
    case 64: kernel_tilingB_32x48_kb64(C, ldc, bufA_top, bufA_bot, bufB0, bufB1, bufB2); break;
    default: kernel_tilingB_32x48(C, ldc, bufA_top, bufA_bot, bufB0, bufB1, bufB2, Kb); break;
    }
}

// Tiling_B + 軽量プリフェッチ
// kb 毎に 2 cache line のみプリフェッチ (T1)。
static inline void kernel_tilingB_32x48_pf(float *C, int ldc,
                                           const bf16_t *bufA_top,
                                           const bf16_t *bufA_bot,
                                           const bf16_t *bufB0,
                                           const bf16_t *bufB1,
                                           const bf16_t *bufB2,
                                           const bf16_t *bufA_top_next,
                                           const bf16_t *bufA_bot_next,
                                           int Kb)
{
    _tile_loadd(0, C + 0*ldc + 0,    ldc * sizeof(float));
    _tile_loadd(1, C + 0*ldc + 16,   ldc * sizeof(float));
    _tile_loadd(2, C + 0*ldc + 32,   ldc * sizeof(float));
    _tile_loadd(3, C + 16*ldc + 0,   ldc * sizeof(float));
    _tile_loadd(4, C + 16*ldc + 16,  ldc * sizeof(float));
    _tile_loadd(5, C + 16*ldc + 32,  ldc * sizeof(float));
    for (int kb = 0; kb < Kb; ++kb) {
        _tile_loadd(6, bufA_top + (size_t)kb*16*32, 64);
        _tile_loadd(7, bufB0   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(0, 6, 7);
        _tile_loadd(7, bufB1   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(1, 6, 7);
        _tile_loadd(7, bufB2   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(2, 6, 7);
        _tile_loadd(6, bufA_bot + (size_t)kb*16*32, 64);
        if (bufA_top_next) {
            _mm_prefetch((const char *)(bufA_top_next + (size_t)kb*16*32),      _MM_HINT_T1);
            _mm_prefetch((const char *)(bufA_top_next + (size_t)kb*16*32) + 64, _MM_HINT_T1);
            _mm_prefetch((const char *)(bufA_bot_next + (size_t)kb*16*32),      _MM_HINT_T1);
            _mm_prefetch((const char *)(bufA_bot_next + (size_t)kb*16*32) + 64, _MM_HINT_T1);
        }
        _tile_dpbf16ps(5, 6, 7);
        _tile_loadd(7, bufB0   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(3, 6, 7);
        _tile_loadd(7, bufB1   + (size_t)kb*16*32, 64);
        _tile_dpbf16ps(4, 6, 7);
    }
    _tile_stored(0, C + 0*ldc + 0,    ldc * sizeof(float));
    _tile_stored(1, C + 0*ldc + 16,   ldc * sizeof(float));
    _tile_stored(2, C + 0*ldc + 32,   ldc * sizeof(float));
    _tile_stored(3, C + 16*ldc + 0,   ldc * sizeof(float));
    _tile_stored(4, C + 16*ldc + 16,  ldc * sizeof(float));
    _tile_stored(5, C + 16*ldc + 32,  ldc * sizeof(float));
}

#endif
