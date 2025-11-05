# DTMハードウェアシミュレータ 仕様書

## 1. 概要

### 1.1 目的
本システムは、論文「An efficient probabilistic hardware architecture for diffusion-like models」で提案されたDenoising Thermodynamic Models (DTM)のハードウェア動作を模擬するシミュレータです。Sudoku、N-queen問題などの制約充足問題での動作検証を目的とします。

### 1.2 システムアーキテクチャ
```
┌─────────────────────────────────────────┐
│          DTM Simulator                  │
├─────────────────────────────────────────┤
│  Problem Interface Layer                │
│  - Sudoku Encoder/Decoder              │
│  - N-Queen Encoder/Decoder             │
│  - Generic Constraint Encoder           │
├─────────────────────────────────────────┤
│  DTM Core Engine                        │
│  - Forward Process                      │
│  - Reverse Process                      │
│  - Multi-Layer EBM Chain               │
├─────────────────────────────────────────┤
│  EBM (Boltzmann Machine) Layer         │
│  - Energy Function                      │
│  - Gibbs Sampler                        │
│  - Sparse Graph Connectivity            │
├─────────────────────────────────────────┤
│  Hardware Simulation Layer              │
│  - RNG Simulator                        │
│  - Bias Circuit Simulator               │
│  - Energy Consumption Tracker           │
└─────────────────────────────────────────┘
```

## 2. コアコンポーネント仕様

### 2.1 Boltzmann Machine (BM)

#### 2.1.1 エネルギー関数
```python
# 論文 Eq. (10)
E(x) = -β * (Σ(i≠j) xi * Jij * xj + Σi hi * xi)
```

**パラメータ:**
- `x`: 二値変数ベクトル {-1, +1}^N
- `J`: 結合重み行列 (N×N, 疎行列)
- `h`: バイアスベクトル (N次元)
- `β`: 逆温度パラメータ (デフォルト=1.0)

**グラフ構造:**
- 2Dグリッド構造 (L×L)
- 接続パターン: Table I (G8, G12, G16, G20, G24)
- 例: G12 = {(0,1), (4,1), (9,10)}

#### 2.1.2 Gibbs Sampling

**条件付き更新則 (論文 Eq. 11):**
```python
P(xi = +1 | X[-i]) = σ(2β * (Σj Jij*xj + hi))
```
ここで σ(x) = 1/(1+exp(-x)) はシグモイド関数

**実装要件:**
- Chromatic Gibbs Sampling (2色での並列更新)
- 混合時間: K ≈ 250-1000 イテレーション
- 自己相関関数によるモニタリング

### 2.2 Forward Process (ノイズ付加過程)

#### 2.2.1 離散変数用マルコフジャンプ過程
論文 Appendix A.1.b に基づく:

```python
# 論文 Eq. (A20)
Q(x^t | x^{t-1}) ∝ exp(Σ_k Γ_k(t) * δ(x^t[k], x^{t-1}[k]))

Γ(t) = ln((1 + (M-1)*exp(-γMt)) / (1 - exp(-γMt)))
```

**パラメータ:**
- `T`: 総ステップ数 (2-8層)
- `γ`: ノイズレート (0.7-1.5 for images)
- `M`: カテゴリ数 (二値の場合 M=2)

### 2.3 Reverse Process (デノイジング過程)

#### 2.3.1 条件付き分布
論文 Eq. (7-8):

```python
P_θ(x^{t-1} | x^t) ∝ Σ_{z^{t-1}} exp(-E^f_{t-1}(x^{t-1}, x^t) 
                                      - E^θ_{t-1}(x^{t-1}, z^{t-1}))
```

**構成要素:**
- `E^f`: Forward process エネルギー (論文 Eq. C1)
- `E^θ`: 学習可能なEBMエネルギー
- `z`: 潜在変数

### 2.4 DTMトレーニング

#### 2.4.1 損失関数
論文 Eq. (14) + Eq. (18):

```python
L = L_DN + Σ_t λ_t * L^TC_t

# Denoising loss
∇_θ L_DN = Σ_t E_Q[E_{P_θ}[∇_θ E^m] - E_{P_θ}[∇_θ E^m]]

# Total correlation penalty
L^TC_t = D(∏_i P_θ(s^i|x^t) || P_θ(s|x^t))
```

#### 2.4.2 Adaptive Correlation Penalty (ACP)
論文 Appendix F.3:

**制御アルゴリズム:**
```python
if autocorr < ε_ACP:
    λ_t = (1 - δ_ACP) * λ_t  # 減少
elif autocorr > prev_autocorr:
    λ_t = (1 + δ_ACP) * λ_t  # 増加
```

**ハイパーパラメータ:**
- `ε_ACP`: 0.02-0.1
- `δ_ACP`: 0.1-0.3
- `λ_min`: 0.001-0.00001

## 3. 問題特化インターフェース

### 3.1 Sudoku問題

#### 3.1.1 エンコーディング
**変数定義:**
- 9×9×9 二値変数: `x[i,j,k] ∈ {0,1}`
  - (i,j): セル位置
  - k: 数字 (1-9)
  - `x[i,j,k]=1` ⟺ セル(i,j)に数字kが入る

**制約のエネルギー関数化:**
```python
# 各セルに1つの数字
E_cell = -Σ_{i,j} (Σ_k x[i,j,k] - 1)²

# 行制約
E_row = -Σ_{i,k} (Σ_j x[i,j,k] - 1)²

# 列制約
E_col = -Σ_{j,k} (Σ_i x[i,j,k] - 1)²

# 3×3ブロック制約
E_block = -Σ_{b,k} (Σ_{(i,j)∈b} x[i,j,k] - 1)²

# 総エネルギー
E_total = α₁*E_cell + α₂*E_row + α₃*E_col + α₄*E_block
```

**ペナルティ重み:** α₁=α₂=α₃=α₄ (均等に設定)

#### 3.1.2 初期値設定
- 与えられたヒント: 対応する`x[i,j,k]=1`に固定
- 残りのセル: ランダム初期化

### 3.2 N-Queen問題

#### 3.2.1 エンコーディング
**変数定義:**
- N×N 二値変数: `x[i,j] ∈ {0,1}`
  - `x[i,j]=1` ⟺ (i,j)にクイーンを配置

**制約のエネルギー関数化:**
```python
# 各行に1つのクイーン
E_row = -Σ_i (Σ_j x[i,j] - 1)²

# 各列に1つのクイーン
E_col = -Σ_j (Σ_i x[i,j] - 1)²

# 対角線制約 (左上→右下)
E_diag1 = -Σ_d (Σ_{i+j=d} x[i,j] ≤ 1の違反ペナルティ)

# 対角線制約 (右上→左下)
E_diag2 = -Σ_d (Σ_{i-j=d} x[i,j] ≤ 1の違反ペナルティ)

E_total = β₁*E_row + β₂*E_col + β₃*E_diag1 + β₄*E_diag2
```

## 4. ハードウェアシミュレーション機能

### 4.1 RNGシミュレータ

**シグモイドバイアス付きベルヌーイサンプリング:**
```python
def rng_sample(bias_voltage, V_s=1.0, φ=0.5):
    """
    論文 Eq. (D6)
    bias_voltage: 制御電圧
    V_s: スケール電圧
    φ: オフセット
    """
    p = sigmoid(bias_voltage / V_s - φ)
    return bernoulli(p)
```

**性能パラメータ (論文 Fig. 4):**
- サンプリングレート: ~10 MHz
- エネルギー: ~350 aJ/bit
- 相関時間: ~100 ns

### 4.2 エネルギー消費トラッカー

論文 Eq. (D12-D17)に基づく:

```python
E_total = T * (E_samp + E_init + E_read)

E_samp = K * N * (E_rng + E_bias + E_clock + E_nb)

# E_rng: RNGエネルギー (~350 aJ)
# E_bias: バイアス回路 (Eq. D10)
# E_clock: クロック配信
# E_nb: 隣接通信 (Eq. D11)
```

**出力メトリクス:**
- サンプルあたりのエネルギー消費 [J/sample]
- GPU比較効率
- レイヤーごとの内訳

## 5. 実装技術仕様

### 5.1 プログラミング言語・フレームワーク

**推奨スタック:**
- **言語:** Python 3.10+
- **数値計算:** NumPy, SciPy
- **機械学習:** JAX (論文で使用)
  - 理由: 自動微分、XLA最適化、GPU対応
- **可視化:** Matplotlib, Seaborn
- **テスト:** pytest

### 5.2 データ構造

#### 5.2.1 Boltzmann Machine クラス
```python
@dataclass
class BoltzmannMachine:
    L: int                    # Grid size
    connectivity: str         # "G8", "G12", etc.
    J: sp.sparse.csr_matrix   # Coupling matrix
    h: np.ndarray            # Bias vector
    beta: float = 1.0        # Inverse temperature
    
    data_nodes: List[int]    # Indices of visible nodes
    latent_nodes: List[int]  # Indices of hidden nodes
```

#### 5.2.2 DTM クラス
```python
@dataclass
class DTMConfig:
    num_layers: int = 8
    grid_size: int = 70
    connectivity: str = "G12"
    K_train: int = 1000      # Training mixing steps
    K_infer: int = 250       # Inference mixing steps
    gamma_forward: float = 1.0
```

### 5.3 モジュール構成

```
dtm_simulator/
├── core/
│   ├── boltzmann_machine.py   # BM実装
│   ├── gibbs_sampler.py        # Gibbs sampling
│   ├── forward_process.py      # Forward process
│   ├── reverse_process.py      # Reverse process
│   └── dtm.py                  # DTMメインクラス
├── hardware/
│   ├── rng_simulator.py        # RNGシミュレータ
│   ├── energy_model.py         # エネルギーモデル
│   └── bias_circuit.py         # バイアス回路
├── problems/
│   ├── base.py                 # 抽象基底クラス
│   ├── sudoku.py              # Sudoku問題
│   ├── nqueen.py              # N-Queen問題
│   └── constraint_encoder.py   # 汎用制約エンコーダ
├── training/
│   ├── loss.py                # 損失関数
│   ├── acp.py                 # Adaptive Correlation Penalty
│   └── trainer.py             # トレーニングループ
├── utils/
│   ├── metrics.py             # 評価指標
│   ├── visualization.py       # 可視化
│   └── autocorrelation.py     # 自己相関計算
└── tests/
    ├── test_boltzmann.py
    ├── test_gibbs.py
    ├── test_sudoku.py
    └── test_nqueen.py
```

## 6. テスト仕様

### 6.1 ユニットテスト

#### 6.1.1 Boltzmann Machine テスト
```python
def test_energy_function():
    """エネルギー関数の計算が正しいか"""
    
def test_gibbs_sampling_stationary():
    """定常分布に収束するか"""
    
def test_connectivity_patterns():
    """接続パターン(G8-G24)が正しいか"""
```

#### 6.1.2 Gibbs Sampler テスト
```python
def test_conditional_update():
    """条件付き更新が論文 Eq. (11) に従うか"""
    
def test_chromatic_sampling():
    """2色並列サンプリングが動作するか"""
    
def test_mixing_time():
    """混合時間が合理的な範囲か"""
```

#### 6.1.3 Forward/Reverse Process テスト
```python
def test_forward_noise_schedule():
    """ノイズスケジュールが論文と一致するか"""
    
def test_reverse_denoising():
    """デノイジングが機能するか"""
```

### 6.2 統合テスト

#### 6.2.1 Sudoku求解テスト
```python
def test_sudoku_easy():
    """簡単なSudoku問題を解けるか"""
    # 制約充足率 > 95%
    
def test_sudoku_hard():
    """難しいSudoku問題でも動作するか"""
    # 制約充足率 > 80%
    
def test_sudoku_invalid():
    """解なし問題を適切に扱えるか"""
```

#### 6.2.2 N-Queen求解テスト
```python
def test_nqueen_small(N=8):
    """8-Queenを解けるか"""
    # 全制約充足を確認
    
def test_nqueen_scaling(N=[12, 16, 20]):
    """スケーラビリティ確認"""
```

### 6.3 性能テスト

```python
def test_energy_efficiency():
    """GPU比での効率が論文と整合するか"""
    # 目標: 10^3-10^4倍の効率改善
    
def test_convergence_speed():
    """収束速度が適切か"""
    # K=250での制約充足率を測定
```

## 7. ドキュメント要件

### 7.1 コードドキュメント

**Docstring規約:** Google Style

```python
def gibbs_step(self, x: np.ndarray, color: int) -> np.ndarray:
    """
    Gibbs samplingの1ステップを実行
    
    Args:
        x: 現在の状態ベクトル (N,) {-1, +1}
        color: 更新する色グループ (0 or 1)
        
    Returns:
        更新後の状態ベクトル (N,)
        
    Note:
        論文 Eq. (11) の条件付き更新則を使用
        P(x_i = +1 | x_{-i}) = σ(2β(Σ_j J_ij x_j + h_i))
    """
```

### 7.2 ユーザーマニュアル

**必須セクション:**
1. **インストール手順**
   - 依存パッケージ
   - セットアップスクリプト

2. **クイックスタート**
   ```python
   from dtm_simulator import DTM, SudokuProblem
   
   # Sudoku問題の読み込み
   problem = SudokuProblem.from_string("53..7....")
   
   # DTM求解器の作成
   dtm = DTM(config=DTMConfig(num_layers=8))
   
   # 求解
   solution = dtm.solve(problem, max_steps=5000)
   ```

3. **API リファレンス**
   - 全クラス・関数の詳細

4. **チュートリアル**
   - Sudoku求解の例
   - N-Queen求解の例
   - カスタム問題の定義方法

### 7.3 開発者ドキュメント

1. **アーキテクチャ設計書**
   - システム構成図
   - データフロー図
   - クラス図

2. **アルゴリズム詳細**
   - 各コンポーネントの数学的背景
   - 論文との対応関係

3. **拡張ガイド**
   - 新しい問題タイプの追加方法
   - 新しい接続パターンの追加方法

## 8. 評価指標

### 8.1 問題求解性能

**Sudoku:**
- 制約充足率: `satisfied_constraints / total_constraints`
- 解探索成功率: 完全解を見つけた割合
- 収束ステップ数: 解に到達するまでのサンプリング回数

**N-Queen:**
- 完全解発見率
- 平均違反数
- スケーリング特性 (N vs 求解時間)

### 8.2 ハードウェア効率性

```python
efficiency_metrics = {
    "energy_per_sample": float,      # [J/sample]
    "gpu_speedup": float,             # GPU比
    "samples_per_second": float,
    "energy_per_constraint": float,   # [J/constraint]
}
```

### 8.3 DTM特有指標

```python
dtm_metrics = {
    "mixing_time_per_layer": List[int],
    "autocorrelation": List[float],
    "layer_wise_fid": List[float],    # Fashion-MNISTとの比較用
    "total_correlation": List[float],
}
```

## 9. 実装フェーズ計画

### Phase 1: コア実装 (2-3週間)
- [ ] Boltzmann Machineクラス
- [ ] Gibbs Samplerクラス
- [ ] エネルギー関数
- [ ] 基本的なユニットテスト

### Phase 2: DTMエンジン (2-3週間)
- [ ] Forward/Reverse Process
- [ ] 多層EBMチェーン
- [ ] トレーニングループ
- [ ] ACP実装

### Phase 3: 問題インターフェース (1-2週間)
- [ ] Sudokuエンコーダ/デコーダ
- [ ] N-Queenエンコーダ/デコーダ
- [ ] 問題固有のテスト

### Phase 4: ハードウェアシミュレーション (1-2週間)
- [ ] RNGシミュレータ
- [ ] エネルギーモデル
- [ ] 性能プロファイラ

### Phase 5: 検証・最適化 (2週間)
- [ ] 統合テスト
- [ ] 性能ベンチマーク
- [ ] ドキュメント整備

## 10. 参考実装例

### 10.1 Boltzmann Machine の基本実装スケルトン

```python
import numpy as np
import scipy.sparse as sp
from typing import Tuple, List

class BoltzmannMachine:
    def __init__(self, L: int, connectivity: str = "G12", beta: float = 1.0):
        self.L = L
        self.N = L * L
        self.connectivity = connectivity
        self.beta = beta
        
        # 接続行列の初期化
        self.J = self._build_connectivity_matrix()
        self.h = np.zeros(self.N)
        
        # 色分け (chromatic Gibbs用)
        self.colors = self._assign_colors()
        
    def _build_connectivity_matrix(self) -> sp.csr_matrix:
        """論文 Table I に基づく接続行列を構築"""
        # 実装詳細は省略
        pass
        
    def _assign_colors(self) -> np.ndarray:
        """2色問題のための色割り当て (チェッカーボードパターン)"""
        colors = np.zeros((self.L, self.L), dtype=int)
        colors[1::2, ::2] = 1
        colors[::2, 1::2] = 1
        return colors.flatten()
        
    def energy(self, x: np.ndarray) -> float:
        """論文 Eq. (10) のエネルギー計算"""
        interaction = -self.beta * (x @ self.J @ x) / 2
        bias = -self.beta * (self.h @ x)
        return interaction + bias
        
    def conditional_prob(self, x: np.ndarray, i: int) -> float:
        """論文 Eq. (11) の条件付き確率"""
        neighbors_sum = self.J[i, :] @ x
        logit = 2 * self.beta * (neighbors_sum + self.h[i])
        return 1.0 / (1.0 + np.exp(-logit))
        
    def gibbs_step(self, x: np.ndarray, color: int) -> np.ndarray:
        """1色グループのGibbs更新"""
        x_new = x.copy()
        indices = np.where(self.colors == color)[0]
        
        for i in indices:
            p_i = self.conditional_prob(x_new, i)
            x_new[i] = 1 if np.random.rand() < p_i else -1
            
        return x_new
        
    def sample(self, x_init: np.ndarray, num_steps: int) -> np.ndarray:
        """Gibbs samplingでサンプル生成"""
        x = x_init.copy()
        
        for _ in range(num_steps):
            x = self.gibbs_step(x, color=0)
            x = self.gibbs_step(x, color=1)
            
        return x
```

この仕様書に基づいて、段階的に実装を進めることができます。各フェーズで必要なテストとドキュメントを整備しながら開発を進めることを推奨します。