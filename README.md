# test_claude3 — Intel AMX BF16 GEMM 最適化

九大 Genkai (Intel Xeon Platinum 8490H, Sapphire Rapids) 上で、Intel AMX を
用いて bf16×bf16=fp32 GEMM の 1 コア性能を最適化する取り組み。

論文 *Endo, Ohshima, Nanri, "Optimization of a GEMM Implementation using Intel
AMX", SCA/HPCAsia 2026* の OpenBLAS ベースカーネルと Tiling_B カーネルを
参考に、7 段階のバリアントを実装した。

## 詳細レポート
**[REPORT.md](REPORT.md)** に方針・実装・全測定結果・考察を記載。

## クイックスタート

```bash
module load gcc-toolset/13
cd src && make            # bench バイナリを作る
pjsub ../jobs/bench_final.sh   # 計算ノードで実行
```

## 結果サマリ (1 thread, GFLOPS)

| N       | MKL (best)  | 自前 best                       | 比 (自前/MKL) |
|---------|-------------|---------------------------------|---------------|
| 8192    | 743         | **981** (v12, k=1024 n=640)      | **1.32** |
| 10000   | 1092 (mean) | **1108** (v9, k=1280 n=528)      | **1.01** |
| 12000   | 1116        | **1109** (v9, k=1280 n=528)      | 0.99 |
| 16384   | 748         | **1008** (v12, k=1024 n=640)     | **1.35** |

- 2 の冪サイズで MKL に対し +32-35%
- N=10000 (目標サイズ) で MKL を平均的に超え、理論ピーク 1945.6 GFLOPS の **57%** を達成
- N=12000 で MKL と誤差範囲内

詳細は [REPORT.md](REPORT.md) 参照。

## ソース構成

- `src/` — 全ソース (`make` でビルド)
  - `common.h` 共通: 型, bf16 変換, タイマー, アロケータ
  - `amx_kernels.h` AMX マイクロカーネル群
  - `gemm_amx.c` 自前 GEMM 実装 v1..v7
  - `gemm_libs.c` MKL ラッパ
  - `bench.c` ベンチマーク本体
- `jobs/` — pjsub ジョブスクリプト
- `results/` — 生ログ (ジョブ出力)
- `3773656.3773660 (2).pdf` — 参考論文
