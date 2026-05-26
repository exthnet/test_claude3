// bench.c: ベンチマーク本体
#include "common.h"
#include "gemm_variants.h"

typedef struct {
    const char *name;
    void (*fn)(int, int, int, const bf16_t *, int, const bf16_t *, int, float *, int);
    int needs_blocks;            // 1: K_block/N_block 引数あり (キャスト)
} variant_t;

// blocking 引数付きラッパ (グローバルで現在のブロックサイズを使う)
static int g_kblock = 1024;
static int g_nblock = 256;

static void w_v2(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc) {
    gemm_v2_base_pack(M, N, K, A, lda, B, ldb, C, ldc, g_kblock);
}
static void w_v3(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc) {
    gemm_v3_base_blocked(M, N, K, A, lda, B, ldb, C, ldc, g_kblock, g_nblock);
}
static void w_v4(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc) {
    gemm_v4_base_pf(M, N, K, A, lda, B, ldb, C, ldc, g_kblock, g_nblock);
}
static void w_v5(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc) {
    gemm_v5_tilingB(M, N, K, A, lda, B, ldb, C, ldc, g_kblock, g_nblock);
}
static void w_v6(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc) {
    gemm_v6_tilingB_pf(M, N, K, A, lda, B, ldb, C, ldc, g_kblock, g_nblock);
}
static void w_v7(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc) {
    gemm_v7_tilingB_cresid(M, N, K, A, lda, B, ldb, C, ldc, g_kblock, g_nblock);
}
static void w_v8(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc) {
    gemm_v8_tilingB_varkblock(M, N, K, A, lda, B, ldb, C, ldc, g_kblock, g_nblock);
}
static void w_v9(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc) {
    gemm_v9_tilingB_kbspec(M, N, K, A, lda, B, ldb, C, ldc, g_kblock, g_nblock);
}
static int g_mpanel = 256;
static void w_v10(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc) {
    gemm_v10_tilingB_mpanel(M, N, K, A, lda, B, ldb, C, ldc, g_kblock, g_nblock, g_mpanel);
}
static void w_v11(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc) {
    gemm_v11_tilingB_mpanel(M, N, K, A, lda, B, ldb, C, ldc, g_kblock, g_nblock, g_mpanel);
}
static void w_v12(int M, int N, int K, const bf16_t *A, int lda, const bf16_t *B, int ldb, float *C, int ldc) {
    gemm_v12_base_kbspec(M, N, K, A, lda, B, ldb, C, ldc, g_kblock, g_nblock);
}

static variant_t variants[] = {
    {"mkl",     gemm_mkl,           0},
    {"ob",      gemm_openblas,      0},
    {"v1",      gemm_v1_amx_naive,  0},
    {"v2",      w_v2,               1},
    {"v3",      w_v3,               1},
    {"v4",      w_v4,               1},
    {"v5",      w_v5,               1},
    {"v6",      w_v6,               1},
    {"v7",      w_v7,               1},
    {"v8",      w_v8,               1},
    {"v9",      w_v9,               1},
    {"v10",     w_v10,              1},
    {"v11",     w_v11,              1},
    {"v12",     w_v12,              1},
};
static const int n_variants = sizeof(variants)/sizeof(variants[0]);

static const variant_t *find_variant(const char *name) {
    for (int i = 0; i < n_variants; ++i)
        if (strcmp(variants[i].name, name) == 0) return &variants[i];
    return NULL;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s VARIANT M N K [iters] [kblock] [nblock] [--verify]\n"
        "  VARIANT: mkl, ob, v1, v2, v3, v4, v5\n"
        "  iters: 計測反復回数 (デフォルト 5)\n"
        "  kblock, nblock: ブロックサイズ (v2..v5)\n"
        "  --verify: M<=512 のとき fp32 リファレンスと比較\n", prog);
}

int main(int argc, char **argv) {
    if (argc < 5) { usage(argv[0]); return 1; }
    const char *vname = argv[1];
    int M = atoi(argv[2]);
    int N = atoi(argv[3]);
    int K = atoi(argv[4]);
    int iters = (argc > 5) ? atoi(argv[5]) : 5;
    if (iters <= 0) iters = 5;
    if (argc > 6) g_kblock = atoi(argv[6]);
    if (argc > 7) g_nblock = atoi(argv[7]);
    if (argc > 8 && argv[8][0] != '-') g_mpanel = atoi(argv[8]);
    int verify = 0;
    for (int i = 8; i < argc; ++i) if (strcmp(argv[i], "--verify") == 0) verify = 1;
    // -- も許容
    for (int i = 5; i < argc; ++i) if (strcmp(argv[i], "--verify") == 0) verify = 1;

    const variant_t *v = find_variant(vname);
    if (!v) { fprintf(stderr, "unknown variant: %s\n", vname); usage(argv[0]); return 1; }

    if (enable_amx_perm() != 0) {
        fprintf(stderr, "warning: enable_amx_perm failed (errno may be set). Continuing.\n");
    }

    bf16_t *A = (bf16_t *)xaligned_alloc((size_t)M * K * sizeof(bf16_t));
    bf16_t *B = (bf16_t *)xaligned_alloc((size_t)K * N * sizeof(bf16_t));
    float  *C = (float  *)xaligned_alloc((size_t)M * N * sizeof(float));

    init_bf16_matrix(A, M, K, 1);
    init_bf16_matrix(B, K, N, 2);
    init_fp32_matrix(C, M, N, 0.0f);

    // 検証
    if (verify && M <= 512 && N <= 512 && K <= 512) {
        float *C_ref = (float *)xaligned_alloc((size_t)M * N * sizeof(float));
        memcpy(C_ref, C, (size_t)M * N * sizeof(float));
        gemm_ref_bf16(M, N, K, A, K, B, N, C_ref, N);

        float *C_test = (float *)xaligned_alloc((size_t)M * N * sizeof(float));
        memcpy(C_test, C, (size_t)M * N * sizeof(float));
        v->fn(M, N, K, A, K, B, N, C_test, N);
        double r = rel_max_diff(C_ref, C_test, M, N);
        printf("[verify] %s  M=%d N=%d K=%d  rel_max_diff=%.3e %s\n",
               vname, M, N, K, r, (r < 5e-2) ? "OK" : "FAIL");
        free(C_ref); free(C_test);
    }

    // ウォームアップ 1 回 + 計測 iters 回
    memset(C, 0, (size_t)M * N * sizeof(float));
    v->fn(M, N, K, A, K, B, N, C, N);

    double best = 1e30, worst = 0.0, sum = 0.0;
    for (int t = 0; t < iters; ++t) {
        memset(C, 0, (size_t)M * N * sizeof(float));
        double t0 = wtime();
        v->fn(M, N, K, A, K, B, N, C, N);
        double t1 = wtime();
        double sec = t1 - t0;
        if (sec < best)  best  = sec;
        if (sec > worst) worst = sec;
        sum += sec;
    }
    double mean = sum / iters;
    double gflops_best  = gflops_of(M, N, K, best);
    double gflops_mean  = gflops_of(M, N, K, mean);

    if (v->needs_blocks) {
        printf("%-6s M=%5d N=%5d K=%5d  kb=%4d nb=%4d  "
               "best=%.4fs mean=%.4fs  GF_best=%.2f GF_mean=%.2f\n",
               vname, M, N, K, g_kblock, g_nblock,
               best, mean, gflops_best, gflops_mean);
    } else {
        printf("%-6s M=%5d N=%5d K=%5d                "
               "best=%.4fs mean=%.4fs  GF_best=%.2f GF_mean=%.2f\n",
               vname, M, N, K, best, mean, gflops_best, gflops_mean);
    }

    free(A); free(B); free(C);
    return 0;
}
