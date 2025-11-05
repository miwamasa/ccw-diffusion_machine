# DTMハードウェアシミュレータ - 理論編

## 目次

1. [はじめに](#はじめに)
2. [Denoising Thermodynamic Models (DTM) の基礎](#denoising-thermodynamic-models-dtm-の基礎)
3. [Boltzmann Machine](#boltzmann-machine)
4. [Gibbs Sampling](#gibbs-sampling)
5. [Forward Process（ノイズ付加過程）](#forward-processノイズ付加過程)
6. [Reverse Process（デノイジング過程）](#reverse-processデノイジング過程)
7. [ハードウェア実装の理論](#ハードウェア実装の理論)
8. [制約充足問題への応用](#制約充足問題への応用)
9. [参考文献](#参考文献)

---

## はじめに

### DTMとは

Denoising Thermodynamic Models (DTM) は、確率的拡散モデルと熱力学的サンプリングを組み合わせた新しいアーキテクチャです。論文「An efficient probabilistic hardware architecture for diffusion-like models」で提案され、以下の特徴を持ちます：

- **エネルギー効率**: GPUと比較して10³〜10⁴倍のエネルギー効率
- **ハードウェア実装に最適**: 疎結合グラフ構造、局所的な計算
- **汎用性**: 制約充足問題、組合せ最適化、画像生成など幅広い応用

### 本ドキュメントの目的

このドキュメントでは、DTMの理論的基礎を数式とともに詳細に解説します。実装の詳細については、別の「実装解説編」を参照してください。

---

## Denoising Thermodynamic Models (DTM) の基礎

### 基本概念

DTMは、データをノイズから段階的に復元する**拡散モデル**の一種です。以下の2つの過程から構成されます：

1. **Forward Process（順過程）**: データに徐々にノイズを加える
2. **Reverse Process（逆過程）**: ノイズからデータを復元する

```
データ x₀ → x₁ → x₂ → ... → xₜ → 完全なノイズ
   ↑                                    ↓
   └────── Reverse Process で復元 ──────┘
```

### DTMの特徴

従来の拡散モデルと異なり、DTMは：

- **エネルギーベースモデル (EBM)** を使用
- **Gibbs Sampling** による確率的サンプリング
- **多層構造** で段階的なデノイジング
- **ハードウェア効率** を重視した設計

---

## Boltzmann Machine

### エネルギー関数

Boltzmann Machine (BM) は、状態 **x** のエネルギーを以下の関数で定義します：

```
E(x) = -β * (Σᵢⱼ xᵢ Jᵢⱼ xⱼ + Σᵢ hᵢ xᵢ)   [論文 Eq. (10)]
```

**パラメータ**:
- **x**: 二値状態ベクトル {-1, +1}^N
- **J**: 結合重み行列 (N×N、疎行列)
- **h**: バイアスベクトル (N次元)
- **β**: 逆温度パラメータ（通常 β=1）

**物理的意味**:
- 第1項（相互作用項）: ノード間の相互作用
- 第2項（バイアス項）: 各ノードの個別の傾向
- エネルギーが**低い**状態ほど**安定**（確率が高い）

### ボルツマン分布

状態 **x** の確率分布は、ボルツマン分布に従います：

```
P(x) = exp(-E(x)) / Z

Z = Σₓ exp(-E(x))  (分配関数)
```

**性質**:
- エネルギーが低い状態ほど高確率
- 温度が高い（β小）→ 確率分布が平坦
- 温度が低い（β大）→ 低エネルギー状態に集中

### グラフ構造

DTMでは、2Dグリッド上のBoltzmann Machineを使用します：

```
L×L グリッド、N = L² ノード
```

**接続パターン** (論文 Table I):

| パターン | 接続数/ノード | 説明 |
|---------|--------------|------|
| G8      | 8            | 4近傍 + 対角4方向 |
| G12     | 12           | G8 + 追加の近傍 |
| G16     | 16           | 拡張接続 |
| G20     | 20           | より広範囲の接続 |
| G24     | 24           | 最大接続 |

**疎結合の利点**:
- メモリ効率が良い (O(N) vs O(N²))
- 並列計算が可能
- ハードウェア実装が容易

---

## Gibbs Sampling

### 条件付き確率

Gibbs Samplingは、一つのノード **i** を他のノードの状態で条件付けて更新します：

```
P(xᵢ = +1 | x₋ᵢ) = σ(2β(Σⱼ Jᵢⱼxⱼ + hᵢ))   [論文 Eq. (11)]
```

ここで、**σ(z)** はシグモイド関数：

```
σ(z) = 1 / (1 + exp(-z))
```

**導出**:

エネルギー差を計算すると：

```
ΔE = E(xᵢ=-1) - E(xᵢ=+1) = 2β(Σⱼ Jᵢⱼxⱼ + hᵢ)

P(xᵢ=+1) / P(xᵢ=-1) = exp(ΔE)
```

これをシグモイドで表現すると上式になります。

### Chromatic Gibbs Sampling

並列化のため、グリッドを2色（チェッカーボードパターン）に分割：

```
色0: (i+j) が偶数のノード
色1: (i+j) が奇数のノード
```

**アルゴリズム**:
1. 色0のノードを並列更新
2. 色1のノードを並列更新
3. 1-2を繰り返す

**利点**:
- 同じ色のノードは相互に接続していない
- 並列更新しても統計的性質が保たれる
- ハードウェア並列実行が可能

### 混合時間

サンプリングが定常分布に収束するまでの時間：

```
K ≈ 250-1000 イテレーション  (論文より)
```

**自己相関関数**で収束を監視：

```
ρ(τ) = Cov(x(t), x(t+τ)) / Var(x(t))
```

ρ(τ) < 0.1 となる τ が混合時間の目安です。

---

## Forward Process（ノイズ付加過程）

### 離散変数の拡散過程

DTMでは、離散的な二値変数に対するマルコフ跳躍過程を使用します（論文 Appendix A.1.b）。

### ノイズスケジュール

時刻 **t** でのノイズレベルを制御するパラメータ Γ(t)：

```
Γ(t) = ln((1 + (M-1)exp(-γMt)) / (1 - exp(-γMt)))   [論文 Eq. (A20)]
```

**パラメータ**:
- **M**: カテゴリ数（二値の場合 M=2）
- **γ**: ノイズレート（画像で 0.7-1.5）
- **t**: 正規化時刻 t ∈ [0, 1]

### ビット反転確率

時刻 **t** でのビット反転確率：

```
p_flip(t) = 1 / (1 + exp(Γ(t)))
```

**特性**:
- t=0: p_flip ≈ 0（ほとんど反転しない）
- t=1: p_flip ≈ 0.5（完全なランダム）

### 遷移確率

状態 x^(t-1) から x^t への遷移：

```
Q(x^t | x^(t-1)) ∝ exp(Σₖ Γₖ(t) * δ(x^t[k], x^(t-1)[k]))
```

δ はクロネッカーのデルタ関数（同じなら1、異なれば0）。

---

## Reverse Process（デノイジング過程）

### 条件付き分布

ノイズ x^t から x^(t-1) を復元する分布：

```
P_θ(x^(t-1) | x^t) ∝ Σ_z exp(-E^f(x^(t-1), x^t) - E^θ(x^(t-1), z))   [論文 Eq. (7-8)]
```

**構成要素**:
- **E^f**: Forward energyエネルギー（時間的一貫性）
- **E^θ**: 学習可能なEBMエネルギー
- **z**: 潜在変数（補助変数）

### Forward Energy

隣接時刻間の一貫性を保つエネルギー：

```
E^f(x^(t-1), x^t) = -Σᵢ log Q(x^t[i] | x^(t-1)[i])   [論文 Eq. (C1)]
```

**意味**: x^t が x^(t-1) から自然に遷移した確率を高めます。

### 学習可能なEBM

各時刻 **t** で独立なBoltzmann Machine:

```
E^θ_t(x, z) = -x^T J_t z - h_t^T x - b_t^T z
```

**学習の目的**:
- データ分布を学習
- デノイジングの方向性を学習
- 制約充足を学習

---

## ハードウェア実装の理論

### RNG（乱数生成器）

シグモイドバイアス付きベルヌーイサンプリング：

```
p = σ(V_bias / V_s - φ)   [論文 Eq. (D6)]

V_bias: バイアス電圧
V_s: スケール電圧（通常 1V）
φ: オフセット（通常 0.5）
```

**性能**（論文 Fig. 4）:
- サンプリングレート: ~10 MHz
- エネルギー: ~350 aJ/bit
- 相関時間: ~100 ns

### エネルギー消費モデル

総エネルギー消費：

```
E_total = T × (E_samp + E_init + E_read)   [論文 Eq. (D12)]
```

**サンプリングエネルギー**:

```
E_samp = K × N × (E_rng + E_bias + E_clock + E_nb)
```

**各コンポーネント**:
- **E_rng**: RNGエネルギー (~350 aJ)
- **E_bias**: バイアス回路 (~100 aJ)
- **E_clock**: クロック配信 (~50 aJ)
- **E_nb**: 隣接通信 (~200 aJ)
- **E_init**: 初期化 (~1 pJ/層)
- **E_read**: 読み出し (~1 pJ/層)

### GPU比較

```
η = E_GPU / E_DTM

論文の結果: η ≈ 10³ - 10⁴
```

**要因**:
- 疎結合によるメモリアクセス削減
- 局所的計算（全結合なし）
- 低精度ビット演算
- 専用ハードウェア最適化

---

## 制約充足問題への応用

### エネルギー関数化

制約充足問題をBMのエネルギー関数に変換：

```
E_total = Σ_c α_c E_c

E_c: 制約 c のエネルギー（違反度）
α_c: 制約の重み
```

**原理**:
- 制約を満たす → エネルギー低い
- 制約を違反 → エネルギー高い
- BMは低エネルギー状態を探索 → 制約を満たす解を発見

### Sudoku問題

9×9×9 二値変数 x[i,j,k]:
- x[i,j,k]=1 ⟺ セル(i,j)に数字k

**制約のエネルギー**:

```
# 各セルに1つの数字
E_cell = -Σ_{i,j} (Σ_k x[i,j,k] - 1)²

# 行制約
E_row = -Σ_{i,k} (Σ_j x[i,j,k] - 1)²

# 列制約
E_col = -Σ_{j,k} (Σ_i x[i,j,k] - 1)²

# ブロック制約
E_block = -Σ_{b,k} (Σ_{(i,j)∈b} x[i,j,k] - 1)²
```

総エネルギー:

```
E_total = α(E_cell + E_row + E_col + E_block)
```

### N-Queen問題

N×N 二値変数 x[i,j]:
- x[i,j]=1 ⟺ 位置(i,j)にクイーン

**制約のエネルギー**:

```
# 各行に1つのクイーン
E_row = -Σ_i (Σ_j x[i,j] - 1)²

# 各列に1つのクイーン
E_col = -Σ_j (Σ_i x[i,j] - 1)²

# 対角線制約
E_diag = -Σ_d max(0, (Σ_{i+j=d} x[i,j]) - 1)²
```

### 性能評価

**評価指標**:

```
# 制約充足率
satisfaction_rate = satisfied_constraints / total_constraints

# エネルギー効率
efficiency = E_GPU / E_DTM

# 収束ステップ数
convergence_steps = min{k : satisfaction_rate(k) > threshold}
```

---

## 参考文献

1. **論文**: "An efficient probabilistic hardware architecture for diffusion-like models"
   - DTMの提案論文
   - ハードウェア実装の詳細
   - 実験結果と性能評価

2. **Boltzmann Machine**:
   - Ackley, D. H., Hinton, G. E., & Sejnowski, T. J. (1985). "A learning algorithm for Boltzmann machines"
   - 基礎理論

3. **Diffusion Models**:
   - Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models"
   - Sohl-Dickstein, J., et al. (2015). "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"

4. **Gibbs Sampling**:
   - Geman, S., & Geman, D. (1984). "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images"

---

## 付録: 重要な数式まとめ

### エネルギー関数
```
E(x) = -β * (Σᵢⱼ xᵢ Jᵢⱼ xⱼ + Σᵢ hᵢ xᵢ)
```

### Gibbs更新
```
P(xᵢ = +1 | x₋ᵢ) = σ(2β(Σⱼ Jᵢⱼxⱼ + hᵢ))
```

### ノイズスケジュール
```
Γ(t) = ln((1 + (M-1)exp(-γMt)) / (1 - exp(-γMt)))
```

### Reverse過程
```
P_θ(x^(t-1) | x^t) ∝ Σ_z exp(-E^f(x^(t-1), x^t) - E^θ(x^(t-1), z))
```

### ハードウェアエネルギー
```
E_total = T × K × N × (E_rng + E_bias + E_clock + E_nb) + T × (E_init + E_read)
```

---

## まとめ

DTMは、以下の理論的基礎に基づいています：

1. **エネルギーベースモデル**: Boltzmann Machineによる確率的モデリング
2. **Gibbs Sampling**: マルコフ連鎖モンテカルロ法による効率的サンプリング
3. **拡散過程**: Forward/Reverse過程による段階的なノイズ除去
4. **ハードウェア効率**: 疎結合、局所計算、低精度演算

これらの理論が組み合わさることで、GPUと比較して10³〜10⁴倍のエネルギー効率を実現します。

実装の詳細については、[実装解説編](implementation.md)を参照してください。
