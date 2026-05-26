// common.h: 共通ヘッダ（型、bf16 変換、タイマー、AMX 設定、ベリファイ）
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <immintrin.h>

typedef uint16_t bf16_t;

#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#define XFEATURE_XTILEDATA 18

static inline int enable_amx_perm(void) {
    long r = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    return (r < 0) ? -1 : 0;
}

typedef struct __attribute__((packed, aligned(64))) {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} tilecfg_t;

// fp32 -> bf16 with round-to-nearest-even (RTNE)
static inline bf16_t fp32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    // Handle NaN: ensure result is NaN.
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) {
        u |= 0x00400000u;
    } else {
        uint32_t lsb = (u >> 16) & 1u;
        u += 0x7fffu + lsb; // RTNE
    }
    return (bf16_t)(u >> 16);
}

static inline float bf16_to_fp32(bf16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f; memcpy(&f, &u, 4);
    return f;
}

// monotonic wall clock in seconds
static inline double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// 巨大ページ要求フラグ (環境変数 AMX_HUGEPAGE=1 で有効)
static inline int amx_use_hugepage(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("AMX_HUGEPAGE");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached;
}

#include <sys/mman.h>

// 行列確保 (64 byte アライン)。AMX_HUGEPAGE=1 なら madvise(MADV_HUGEPAGE)
// をヒントとして与え THP を促す (解放は通常の free でよい)。
static inline void *xaligned_alloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) {
        fprintf(stderr, "alloc fail %zu\n", bytes);
        exit(1);
    }
    if (amx_use_hugepage() && bytes >= 2*1024*1024) {
        // 2MB アライン位置から madvise (途中まで小ページが混在しても可)
        const uintptr_t HP = 2UL * 1024 * 1024;
        uintptr_t addr = (uintptr_t)p;
        uintptr_t hp_aligned = (addr + HP - 1) & ~(HP - 1);
        size_t prefix = (size_t)(hp_aligned - addr);
        if (prefix < bytes) {
            madvise((void *)hp_aligned, bytes - prefix, MADV_HUGEPAGE);
        }
    }
    return p;
}


// 行列を乱数で初期化（fp32→bf16）。値域を小さくして fp32 比較で誤差を抑える。
static inline void init_bf16_matrix(bf16_t *A, int M, int N, unsigned seed) {
    srand(seed);
    for (int i = 0; i < M * N; ++i) {
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // [-1, 1]
        A[i] = fp32_to_bf16(v);
    }
}

static inline void init_fp32_matrix(float *C, int M, int N, float val) {
    for (int i = 0; i < M * N; ++i) C[i] = val;
}

// 単純な fp32 参照 GEMM（小さいサイズの検証用）
// C = A*B (bf16 を fp32 に拡張して計算)。row-major.
static inline void gemm_ref_bf16(int M, int N, int K,
                                 const bf16_t *A, int lda,
                                 const bf16_t *B, int ldb,
                                 float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = C[i*ldc + j];
            for (int k = 0; k < K; ++k) {
                s += bf16_to_fp32(A[i*lda + k]) * bf16_to_fp32(B[k*ldb + j]);
            }
            C[i*ldc + j] = s;
        }
    }
}

// |C1 - C2|_max / |C1|_inf
static inline double rel_max_diff(const float *C1, const float *C2, int M, int N) {
    double maxd = 0.0, maxv = 0.0;
    for (int i = 0; i < M*N; ++i) {
        double d = fabs((double)C1[i] - (double)C2[i]);
        double v = fabs((double)C1[i]);
        if (d > maxd) maxd = d;
        if (v > maxv) maxv = v;
    }
    return maxd / (maxv + 1e-30);
}

static inline double gflops_of(int M, int N, int K, double sec) {
    double ops = 2.0 * (double)M * (double)N * (double)K;
    return ops / sec / 1e9;
}

#endif
