# DTMハードウェアシミュレータ - 実装解説編

## 目次

1. [はじめに](#はじめに)
2. [アーキテクチャ概要](#アーキテクチャ概要)
3. [Boltzmann Machine実装](#boltzmann-machine実装)
4. [Gibbs Sampler実装](#gibbs-sampler実装)
5. [Forward/Reverse Process実装](#forwardreverse-process実装)
6. [DTMメインクラス実装](#dtmメインクラス実装)
7. [ハードウェアシミュレーション実装](#ハードウェアシミュレーション実装)
8. [問題インターフェース実装](#問題インターフェース実装)
9. [パフォーマンス最適化](#パフォーマンス最適化)
10. [テスト戦略](#テスト戦略)
11. [使用例とベストプラクティス](#使用例とベストプラクティス)

---

## はじめに

### 対象読者

このドキュメントは、以下の読者を想定しています：

- DTMシミュレータのコードを理解したい開発者
- 実装の詳細を知りたい研究者
- カスタマイズや拡張を行いたいユーザー

### 前提知識

- Python 3.10以上の基本的な知識
- NumPy/SciPyの使用経験
- [理論編](theory.md)の理解

### ファイル構造

```
dtm_simulator/
├── core/                           # コアコンポーネント
│   ├── boltzmann_machine.py       # BMの実装
│   ├── gibbs_sampler.py           # Gibbsサンプラー
│   ├── forward_process.py         # Forward過程
│   ├── reverse_process.py         # Reverse過程
│   └── dtm.py                     # DTMメインクラス
├── hardware/                       # HWシミュレーション
│   ├── rng_simulator.py           # RNG
│   ├── energy_model.py            # エネルギーモデル
│   └── bias_circuit.py            # バイアス回路
├── problems/                       # 問題インターフェース
│   ├── base.py                    # 抽象基底クラス
│   ├── sudoku.py                  # Sudoku問題
│   └── nqueen.py                  # N-Queen問題
└── tests/                         # テスト
    └── test_basic.py
```

---

## アーキテクチャ概要

### 設計原則

1. **モジュール性**: 各コンポーネントは独立して動作
2. **拡張性**: 新しい問題やアルゴリズムを簡単に追加可能
3. **効率性**: NumPyベクトル化による高速計算
4. **可読性**: ドキュメントと論文の対応を明確に

### データフロー

```
Problem Definition
     ↓
  DTM Setup
     ↓
Initial State → Gibbs Sampling → Energy Evaluation
     ↑               ↓
     └───── Iterative Updates ─────┘
     ↓
Solution Decode
```

### クラス図

```
ConstraintProblem (抽象基底)
    ↑
    ├── SudokuProblem
    └── NQueenProblem

BoltzmannMachine
    ↓
GibbsSampler (診断機能)
    ↓
ForwardProcess / ReverseProcess
    ↓
DTM (統合クラス)
```

---

## Boltzmann Machine実装

### クラス定義

**ファイル**: `dtm_simulator/core/boltzmann_machine.py`

```python
@dataclass
class BoltzmannMachine:
    L: int                    # グリッドサイズ
    connectivity: str = "G12" # 接続パターン
    beta: float = 1.0        # 逆温度
    seed: Optional[int] = None
```

### 接続行列の構築

**メソッド**: `_build_connectivity_matrix()`

```python
def _build_connectivity_matrix(self) -> sp.csr_matrix:
    """疎行列で接続を表現"""

    # 接続パターンの定義
    patterns = {
        "G8": [(0,1), (1,0), (0,-1), (-1,0)],
        "G12": [...],  # 8方向 + 4対角
        # ... 他のパターン
    }

    offsets = patterns[self.connectivity]
    rows, cols, data = [], [], []

    # 2Dグリッド上の接続を構築
    for i in range(self.L):
        for j in range(self.L):
            idx = i * self.L + j
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.L and 0 <= nj < self.L:
                    nidx = ni * self.L + nj
                    rows.append(idx)
                    cols.append(nidx)
                    data.append(self.rng.normal(0, 0.1))

    J = sp.csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
    # 対称化
    return (J + J.T) / 2
```

**ポイント**:
- `sp.csr_matrix`で疎行列（メモリ効率良い）
- 対称性を保証（J_ij = J_ji）
- O(N)のメモリ（全結合ならO(N²)）

### 色割り当て（Chromatic）

**メソッド**: `_assign_colors()`

```python
def _assign_colors(self) -> np.ndarray:
    """チェッカーボードパターン"""
    colors = np.zeros((self.L, self.L), dtype=int)
    colors[1::2, ::2] = 1   # 奇数行・偶数列
    colors[::2, 1::2] = 1   # 偶数行・奇数列
    return colors.flatten()
```

**結果**:
```
0 1 0 1
1 0 1 0
0 1 0 1
1 0 1 0
```

### エネルギー計算

**メソッド**: `energy(x)`

```python
def energy(self, x: np.ndarray) -> float:
    """論文 Eq. (10) の実装"""
    # 相互作用項: -β * x^T J x / 2
    interaction = -self.beta * (x @ self.J @ x) / 2

    # バイアス項: -β * h^T x
    bias = -self.beta * (self.h @ x)

    return interaction + bias
```

**計算複雑度**:
- 疎行列演算: O(E)（Eは辺の数）
- 全結合なら: O(N²)
- G12パターン: E ≈ 6N → O(N)

### 条件付き確率

**メソッド**: `conditional_prob(x, i)`

```python
def conditional_prob(self, x: np.ndarray, i: int) -> float:
    """論文 Eq. (11) の実装"""
    # 隣接ノードの寄与を計算
    neighbors_sum = self.J[i, :].toarray().flatten() @ x

    # ロジット: 2β(Σ_j J_ij x_j + h_i)
    logit = 2 * self.beta * (neighbors_sum + self.h[i])

    # シグモイド
    return 1.0 / (1.0 + np.exp(-logit))
```

**数値安定性**:
- `np.exp(-logit)`で大きな負の値を避ける
- 必要に応じて`np.clip`で範囲制限

### Gibbs更新

**メソッド**: `gibbs_step(x, color)`

```python
def gibbs_step(self, x: np.ndarray, color: int) -> np.ndarray:
    """1色グループの並列更新"""
    x_new = x.copy()
    indices = np.where(self.colors == color)[0]

    for i in indices:
        p_i = self.conditional_prob(x_new, i)
        x_new[i] = 1 if self.rng.random() < p_i else -1

    return x_new
```

**並列化の可能性**:
- 同じ色のノードは独立に更新可能
- 実際のハードウェアでは真の並列実行
- Pythonでは逐次実行（NumPyベクトル化で高速化可能）

---

## Gibbs Sampler実装

### 自己相関計算

**ファイル**: `dtm_simulator/core/gibbs_sampler.py`

**関数**: `compute_autocorrelation(samples, max_lag)`

```python
def compute_autocorrelation(samples: List[np.ndarray],
                           max_lag: int = 50) -> np.ndarray:
    """自己相関関数の計算"""
    samples_array = np.array(samples)
    mean = np.mean(samples_array, axis=0)
    var = np.var(samples_array, axis=0)

    autocorr = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            cov = np.mean(
                (samples_array[:-lag] - mean) *
                (samples_array[lag:] - mean),
                axis=0
            )
            autocorr[lag] = np.mean(cov / (var + 1e-10))

    return autocorr
```

**用途**:
- 混合時間の推定
- サンプリング効率の評価
- 収束診断

### 混合時間推定

**関数**: `estimate_mixing_time(autocorr, threshold)`

```python
def estimate_mixing_time(autocorr: np.ndarray,
                        threshold: float = 0.1) -> int:
    """自己相関が閾値以下になる最初の時刻"""
    for i, ac in enumerate(autocorr):
        if abs(ac) < threshold:
            return i
    return len(autocorr)
```

### 有効サンプルサイズ

**関数**: `effective_sample_size(n_samples, autocorr)`

```python
def effective_sample_size(n_samples: int,
                         autocorr: np.ndarray) -> float:
    """ESS = N / (1 + 2Σρ(k))"""
    autocorr_sum = np.sum(autocorr[1:])
    ess = n_samples / (1.0 + 2.0 * autocorr_sum)
    return max(1.0, ess)
```

**意味**:
- 相関を考慮した実効的なサンプル数
- ESS ≈ N なら良好な混合
- ESS << N なら混合が不十分

---

## Forward/Reverse Process実装

### Forward Process

**ファイル**: `dtm_simulator/core/forward_process.py`

#### ノイズスケジュールの計算

```python
def _compute_noise_schedule(self) -> np.ndarray:
    """論文 Eq. (A20) の実装"""
    t_values = np.arange(self.T + 1) / self.T
    gamma_Mt = self.gamma * self.M * t_values

    numerator = 1 + (self.M - 1) * np.exp(-gamma_Mt)
    denominator = 1 - np.exp(-gamma_Mt)

    gamma_schedule = np.where(
        t_values > 0,
        np.log(numerator / (denominator + 1e-10)),
        10.0  # t=0の特別処理
    )

    return gamma_schedule
```

**数値安定性**:
- `1e-10`で0除算を回避
- t=0での特別処理

#### ノイズ付加

```python
def add_noise(self, x: np.ndarray, t: int,
              rng: np.random.Generator = None) -> np.ndarray:
    """状態xにノイズを加える"""
    flip_prob = self.get_transition_prob(t)

    # ビット反転マスク
    flip_mask = rng.random(x.shape) < flip_prob
    x_noisy = x.copy()
    x_noisy[flip_mask] = -x_noisy[flip_mask]

    return x_noisy
```

### Reverse Process

**ファイル**: `dtm_simulator/core/reverse_process.py`

#### デノイジングステップ

```python
def denoise_step(self, x_t: np.ndarray, t: int,
                 rng: np.random.Generator = None) -> np.ndarray:
    """1ステップのデノイジング"""
    ebm = self.ebm_layers[t - 1]

    # x_tを条件とするバイアスを計算
    bias = self._compute_conditional_bias(x_t, t)
    ebm.set_bias(bias)

    # EBMでサンプリング
    x_prev = ebm.sample(x_t, num_steps=self.K)[0]

    return x_prev
```

#### 条件付きバイアス

```python
def _compute_conditional_bias(self, x_t: np.ndarray,
                              t: int) -> np.ndarray:
    """Forward energyを考慮したバイアス"""
    flip_prob = self.forward_process.get_transition_prob(t)

    # x_tに近い状態を優先
    bias = x_t * (1.0 - 2.0 * flip_prob)

    return bias
```

**意味**:
- flip_prob小 → x_tに近い状態を強く優先
- flip_prob大 → より柔軟な探索

---

## DTMメインクラス実装

### 設定クラス

**ファイル**: `dtm_simulator/core/dtm.py`

```python
@dataclass
class DTMConfig:
    num_layers: int = 8        # 拡散層数
    grid_size: int = 10        # BMグリッドサイズ
    connectivity: str = "G12"  # 接続パターン
    K_train: int = 1000        # 訓練時混合ステップ
    K_infer: int = 250         # 推論時混合ステップ
    gamma_forward: float = 1.0 # ノイズレート
    beta: float = 1.0          # 逆温度
    seed: Optional[int] = None
```

### 初期化

```python
def __init__(self, config: DTMConfig = None):
    self.config = config
    self.rng = np.random.default_rng(config.seed)

    # Forward過程の初期化
    self.forward_process = ForwardProcess(
        num_layers=config.num_layers,
        gamma=config.gamma_forward,
        M=2
    )

    # EBM層の作成
    self.ebm_layers = self._create_ebm_layers()

    # Reverse過程の初期化
    self.reverse_process = ReverseProcess(
        ebm_layers=self.ebm_layers,
        forward_process=self.forward_process,
        mixing_steps=config.K_infer
    )
```

### 問題求解

```python
def solve(self, problem, max_steps: int = 5000,
          verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """制約充足問題を解く"""
    N = problem.get_num_variables()

    # ランダム初期化
    x_init = self.rng.choice([-1, 1], size=N)

    # 問題特有のバイアスを設定
    for ebm in self.ebm_layers:
        problem_bias = problem.get_bias_vector(N)
        ebm.set_bias(problem_bias)

    # 反復サンプリング
    best_x = x_init.copy()
    best_energy = float('inf')
    energies = []

    x = x_init.copy()
    for step in range(max_steps):
        # Gibbs更新
        x = self.ebm_layers[-1].gibbs_step(x, color=0)
        x = self.ebm_layers[-1].gibbs_step(x, color=1)

        # エネルギー評価
        energy = problem.energy_function(x)
        energies.append(energy)

        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

        if verbose and step % 500 == 0:
            print(f"Step {step}: Energy = {energy:.4f}")

    return best_x, {"energies": energies,
                    "best_energy": best_energy}
```

**アルゴリズムの流れ**:
1. ランダム初期化
2. 問題制約をEBMバイアスに変換
3. Gibbsサンプリングで状態更新
4. エネルギー評価と最良解の保存
5. 収束まで繰り返し

---

## ハードウェアシミュレーション実装

### RNGシミュレータ

**ファイル**: `dtm_simulator/hardware/rng_simulator.py`

```python
class RNGSimulator:
    def sample(self, bias_voltage: float,
               rng: np.random.Generator = None) -> int:
        """論文 Eq. (D6) の実装"""
        # 確率計算
        p = self.sigmoid(bias_voltage / self.V_s - self.phi)

        # サンプリング
        bit = 1 if rng.random() < p else 0

        # 統計更新
        self.total_samples += 1
        self.total_energy += self.energy_per_bit

        return bit
```

**ベクトル版**（高速）:

```python
def sample_vector(self, bias_voltages: np.ndarray,
                 rng: np.random.Generator = None) -> np.ndarray:
    """並列サンプリング"""
    p = self.sigmoid(bias_voltages / self.V_s - self.phi)
    bits = (rng.random(len(bias_voltages)) < p).astype(int)

    self.total_samples += len(bits)
    self.total_energy += self.energy_per_bit * len(bits)

    return bits
```

### エネルギーモデル

**ファイル**: `dtm_simulator/hardware/energy_model.py`

```python
class EnergyModel:
    def compute_total_energy(self, T: int, K: int,
                            N: int) -> float:
        """論文 Eq. (D12) の実装"""
        E_samp = self.compute_sampling_energy(K, N)
        E_layer = E_samp + self.E_init + self.E_read
        return T * E_layer

    def compute_sampling_energy(self, K: int, N: int) -> float:
        """サンプリングエネルギー"""
        E_per_var = (self.E_rng + self.E_bias +
                    self.E_clock + self.E_nb)
        return K * N * E_per_var
```

**エネルギー内訳**:

```python
def compute_energy_breakdown(self, T: int, K: int,
                            N: int) -> Dict[str, float]:
    """詳細な内訳"""
    return {
        "rng": T * K * N * self.E_rng,
        "bias": T * K * N * self.E_bias,
        "clock": T * K * N * self.E_clock,
        "neighbor": T * K * N * self.E_nb,
        "init": T * self.E_init,
        "read": T * self.E_read,
        "total": self.compute_total_energy(T, K, N)
    }
```

### バイアス回路

**ファイル**: `dtm_simulator/hardware/bias_circuit.py`

```python
class BiasCircuit:
    def prob_to_voltage(self, p: float) -> float:
        """確率→電圧変換（逆シグモイド）"""
        p = np.clip(p, 1e-10, 1 - 1e-10)
        logit = np.log(p / (1 - p))
        voltage = self.V_s * (logit + self.phi)
        return voltage

    def voltage_to_prob(self, V: float) -> float:
        """電圧→確率変換（シグモイド）"""
        return 1.0 / (1.0 + np.exp(-(V / self.V_s - self.phi)))
```

**用途**:
- Gibbs更新の確率を電圧に変換
- ハードウェアRNGの制御
- エネルギー消費の推定

---

## 問題インターフェース実装

### 抽象基底クラス

**ファイル**: `dtm_simulator/problems/base.py`

```python
class ConstraintProblem(ABC):
    @abstractmethod
    def get_num_variables(self) -> int:
        """変数数を返す"""
        pass

    @abstractmethod
    def energy_function(self, x: np.ndarray) -> float:
        """エネルギー関数（制約違反度）"""
        pass

    @abstractmethod
    def decode_solution(self, x: np.ndarray) -> any:
        """バイナリ→問題の解に変換"""
        pass

    @abstractmethod
    def check_constraints(self, x: np.ndarray) -> Tuple[int, int]:
        """制約充足をチェック"""
        pass
```

### Sudoku実装

**ファイル**: `dtm_simulator/problems/sudoku.py`

#### エンコーディング

```python
def __init__(self, puzzle: np.ndarray, ...):
    # 9×9×9 = 729変数
    self.N = 9 * 9 * 9
    self.puzzle = puzzle.copy()
    self.given_mask = (puzzle > 0)

def _idx(self, i: int, j: int, k: int) -> int:
    """3Dインデックス→1Dインデックス"""
    return i * 81 + j * 9 + k
```

#### エネルギー関数

```python
def energy_function(self, x: np.ndarray) -> float:
    """制約をエネルギーに変換"""
    x_bin = self._spin_to_binary(x)
    x_3d = x_bin.reshape(9, 9, 9)

    energy = 0.0

    # セル制約: 各セルに1つの数字
    for i in range(9):
        for j in range(9):
            constraint = np.sum(x_3d[i, j, :]) - 1
            energy -= self.alpha_cell * constraint ** 2

    # 行制約
    for i in range(9):
        for k in range(9):
            constraint = np.sum(x_3d[i, :, k]) - 1
            energy -= self.alpha_row * constraint ** 2

    # 列制約
    for j in range(9):
        for k in range(9):
            constraint = np.sum(x_3d[:, j, k]) - 1
            energy -= self.alpha_col * constraint ** 2

    # ブロック制約
    for bi in range(3):
        for bj in range(3):
            for k in range(9):
                block = x_3d[bi*3:(bi+1)*3, bj*3:(bj+1)*3, k]
                constraint = np.sum(block) - 1
                energy -= self.alpha_block * constraint ** 2

    return energy
```

#### デコーディング

```python
def decode_solution(self, x: np.ndarray) -> np.ndarray:
    """729次元ベクトル→9×9グリッド"""
    x_bin = self._spin_to_binary(x)
    x_3d = x_bin.reshape(9, 9, 9)

    solution = np.zeros((9, 9), dtype=int)

    for i in range(9):
        for j in range(9):
            digits = x_3d[i, j, :]
            if np.sum(digits) > 0:
                k = np.argmax(digits)
                solution[i, j] = k + 1

    return solution
```

#### バイアス設定

```python
def get_bias_vector(self, N: int) -> np.ndarray:
    """与えられたヒントを強いバイアスで固定"""
    bias = np.zeros(N)

    for i in range(9):
        for j in range(9):
            if self.given_mask[i, j]:
                digit = self.puzzle[i, j]
                k = digit - 1
                idx = self._idx(i, j, k)
                bias[idx] = 10.0  # 強い正バイアス

                # 他の数字に負バイアス
                for k_other in range(9):
                    if k_other != k:
                        idx_other = self._idx(i, j, k_other)
                        bias[idx_other] = -10.0

    return bias
```

### N-Queen実装

**ファイル**: `dtm_simulator/problems/nqueen.py`

#### エネルギー関数

```python
def energy_function(self, x: np.ndarray) -> float:
    x_bin = self._spin_to_binary(x)
    board = x_bin.reshape(self.board_size, self.board_size)

    energy = 0.0

    # 行制約
    for i in range(self.board_size):
        constraint = np.sum(board[i, :]) - 1
        energy -= self.beta_row * constraint ** 2

    # 列制約
    for j in range(self.board_size):
        constraint = np.sum(board[:, j]) - 1
        energy -= self.beta_col * constraint ** 2

    # 対角線制約（≤1個のクイーン）
    for d in range(-(self.board_size-1), self.board_size):
        diag = np.diag(board, k=d)
        if len(diag) > 1:
            violation = max(0, np.sum(diag) - 1)
            energy -= self.beta_diag1 * violation ** 2

    # 反対角線制約
    board_flipped = np.fliplr(board)
    for d in range(-(self.board_size-1), self.board_size):
        diag = np.diag(board_flipped, k=d)
        if len(diag) > 1:
            violation = max(0, np.sum(diag) - 1)
            energy -= self.beta_diag2 * violation ** 2

    return energy
```

---

## パフォーマンス最適化

### NumPyベクトル化

**悪い例**（ループ）:
```python
# 遅い
result = []
for i in range(N):
    result.append(func(x[i]))
result = np.array(result)
```

**良い例**（ベクトル化）:
```python
# 速い
result = func(x)  # NumPyの組み込み関数を使う
```

### 疎行列の活用

```python
# 密行列（メモリ O(N²)）
J_dense = np.zeros((N, N))

# 疎行列（メモリ O(E)、Eは辺の数）
J_sparse = sp.csr_matrix((data, (rows, cols)), shape=(N, N))

# 計算も高速
x @ J_sparse @ x  # O(E) vs O(N²)
```

### メモリ効率

**コピーの最小化**:
```python
# コピーが必要な場合のみ
x_new = x.copy()

# インプレース操作
x[mask] = value  # コピーなし
```

**ビューの活用**:
```python
# コピーなし
x_3d = x.reshape(9, 9, 9)  # ビュー
block = x_3d[0:3, 0:3, :]  # ビュー
```

### 並列化の可能性

現在の実装は逐次実行ですが、以下の部分は並列化可能：

1. **Chromatic Gibbs**: 同じ色のノード
2. **エネルギー計算**: 各制約を独立に計算
3. **複数層のEBM**: 独立なら並列学習可能

```python
# 将来の並列化例（擬似コード）
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(update_node, i)
               for i in indices]
    results = [f.result() for f in futures]
```

---

## テスト戦略

### ユニットテスト

**ファイル**: `dtm_simulator/tests/test_basic.py`

#### Boltzmann Machineのテスト

```python
def test_boltzmann_machine_energy():
    """エネルギー計算が正しいか"""
    bm = BoltzmannMachine(L=3, connectivity="G8", seed=42)
    x = np.ones(9)
    energy = bm.energy(x)
    assert isinstance(energy, (float, np.floating))
    # エネルギーは有限値
    assert np.isfinite(energy)
```

#### Gibbs Samplingのテスト

```python
def test_gibbs_sampling():
    """Gibbsサンプリングが動作するか"""
    bm = BoltzmannMachine(L=4, connectivity="G8", seed=42)
    x_init = np.random.choice([-1, 1], size=16)
    x_final, _ = bm.sample(x_init, num_steps=10)

    # 出力形状が正しい
    assert x_final.shape == (16,)
    # 値が±1
    assert np.all(np.abs(x_final) == 1)
```

#### 問題のテスト

```python
def test_sudoku_problem():
    """Sudoku問題のエンコーディング"""
    puzzle_str = "530070000..." # 81文字
    problem = SudokuProblem.from_string(puzzle_str)

    # 変数数が正しい
    assert problem.get_num_variables() == 729

    # エネルギー計算が動作
    x = np.random.choice([-1, 1], size=729)
    energy = problem.energy_function(x)
    assert isinstance(energy, (float, np.floating))

    # デコードが動作
    solution = problem.decode_solution(x)
    assert solution.shape == (9, 9)
```

### 統合テスト

```python
def test_dtm_solve_nqueen():
    """DTMでN-Queen問題を解く"""
    problem = NQueenProblem(N=4)  # 小さい問題
    config = DTMConfig(
        num_layers=2,
        grid_size=4,
        K_infer=50,
        beta=1.0,
        seed=42
    )
    dtm = DTM(config)

    solution_x, info = dtm.solve(problem, max_steps=100)

    # 解が得られる
    assert solution_x.shape == (16,)
    # エネルギーが記録される
    assert len(info["energies"]) == 100
```

### テスト実行

```bash
# 全テスト
pytest dtm_simulator/tests/ -v

# カバレッジ付き
pytest dtm_simulator/tests/ --cov=dtm_simulator --cov-report=html

# 特定のテスト
pytest dtm_simulator/tests/test_basic.py::test_boltzmann_machine_energy -v
```

---

## 使用例とベストプラクティス

### 基本的な使用例

```python
from dtm_simulator.core.dtm import DTM, DTMConfig
from dtm_simulator.problems.sudoku import SudokuProblem

# 問題を定義
puzzle_str = "530070000600195000..."
problem = SudokuProblem.from_string(puzzle_str)

# DTMを作成
config = DTMConfig(
    num_layers=4,
    grid_size=27,  # sqrt(729) ≈ 27
    K_infer=150,
    beta=2.0,
    seed=42
)
dtm = DTM(config)

# 求解
solution_x, info = dtm.solve(problem, max_steps=3000, verbose=True)

# 結果を表示
solution = problem.decode_solution(solution_x)
print(problem.format_solution(solution))
print(f"Satisfaction rate: {problem.satisfaction_rate(solution_x):.2%}")
```

### パラメータチューニング

#### グリッドサイズの選択

```python
# 問題の変数数に応じて調整
N = problem.get_num_variables()
L = int(np.ceil(np.sqrt(N)))
config.grid_size = L
```

#### 逆温度βの調整

```python
# β大: 低エネルギー状態に集中（収束速いが局所解に陥りやすい）
config.beta = 2.0

# β小: 広く探索（収束遅いがグローバル探索）
config.beta = 0.5

# 焼きなまし的なアプローチ
for step in range(max_steps):
    beta = beta_min + (beta_max - beta_min) * (step / max_steps)
    dtm.config.beta = beta
```

#### 混合ステップ数

```python
# 小さい問題
config.K_infer = 100

# 中規模問題
config.K_infer = 250

# 大規模問題
config.K_infer = 500-1000

# 適応的な設定
if problem.get_num_variables() < 100:
    config.K_infer = 100
else:
    config.K_infer = 250
```

### エネルギー消費の測定

```python
from dtm_simulator.hardware.energy_model import EnergyModel

energy_model = EnergyModel()

# シミュレーション後
breakdown = energy_model.compute_energy_breakdown(
    T=config.num_layers,
    K=config.K_infer,
    N=problem.get_num_variables()
)

print(f"Total: {energy_model.format_energy(breakdown['total'])}")
print(f"RNG: {energy_model.format_energy(breakdown['rng'])}")
print(f"Bias: {energy_model.format_energy(breakdown['bias'])}")

# GPU比較
gpu_speedup = energy_model.compare_with_gpu(
    dtm_energy=breakdown['total'],
    problem_size=problem.get_num_variables()
)
print(f"Energy efficiency vs GPU: {gpu_speedup:.1f}×")
```

### カスタム問題の実装

```python
from dtm_simulator.problems.base import ConstraintProblem
import numpy as np

class MyProblem(ConstraintProblem):
    def __init__(self, ...):
        # 初期化
        self.N = ...  # 変数数

    def get_num_variables(self) -> int:
        return self.N

    def energy_function(self, x: np.ndarray) -> float:
        # 制約違反度を計算
        # 低いほど良い解
        energy = 0.0
        # ... 制約ごとにペナルティを加算
        return energy

    def decode_solution(self, x: np.ndarray) -> any:
        # バイナリベクトルを解に変換
        return solution

    def check_constraints(self, x: np.ndarray) -> Tuple[int, int]:
        # 充足した制約数と総制約数
        satisfied = 0
        total = 0
        # ... 各制約をチェック
        return satisfied, total

    def get_bias_vector(self, N: int) -> np.ndarray:
        # 初期バイアス（オプション）
        return np.zeros(N)
```

### デバッグのヒント

#### エネルギーの可視化

```python
import matplotlib.pyplot as plt

_, info = dtm.solve(problem, max_steps=1000)
energies = info["energies"]

plt.plot(energies)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("Energy Evolution")
plt.show()
```

#### 制約充足率の追跡

```python
satisfactions = []
for step in range(max_steps):
    x = dtm.step()  # 1ステップ更新
    sat_rate = problem.satisfaction_rate(x)
    satisfactions.append(sat_rate)

plt.plot(satisfactions)
plt.xlabel("Iteration")
plt.ylabel("Satisfaction Rate")
plt.show()
```

#### 自己相関の診断

```python
from dtm_simulator.core.gibbs_sampler import compute_autocorrelation

# サンプル列を記録
samples = []
for _ in range(100):
    x = bm.sample(x_init, num_steps=10)[0]
    samples.append(x)

# 自己相関を計算
autocorr = compute_autocorrelation(samples, max_lag=50)

plt.plot(autocorr)
plt.axhline(y=0.1, color='r', linestyle='--', label='Threshold')
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.legend()
plt.show()
```

### ベストプラクティス

1. **シード設定**: 再現性のため常にseedを設定
2. **段階的な最適化**: まず小さい問題で動作確認、次に大規模問題へ
3. **エネルギー監視**: エネルギーが減少しているか常にチェック
4. **早期停止**: 一定ステップでエネルギーが改善しなければ停止
5. **複数回実行**: 確率的アルゴリズムなので複数回試行して最良解を選択

```python
# 複数回実行の例
best_solution = None
best_energy = float('inf')

for trial in range(5):
    config.seed = 42 + trial
    dtm = DTM(config)
    solution_x, info = dtm.solve(problem, max_steps=1000)

    if info["best_energy"] < best_energy:
        best_energy = info["best_energy"]
        best_solution = solution_x

print(f"Best energy over 5 trials: {best_energy}")
```

---

## まとめ

このドキュメントでは、DTMハードウェアシミュレータの実装の詳細を解説しました：

1. **モジュール構造**: 各コンポーネントの役割と連携
2. **アルゴリズム実装**: 論文の数式をPythonコードに変換
3. **最適化技術**: NumPyベクトル化、疎行列、メモリ効率
4. **テスト戦略**: ユニットテストと統合テスト
5. **使用例**: 実践的なコード例とパラメータチューニング

さらに詳しい理論的背景については、[理論編](theory.md)を参照してください。

---

## 参考リソース

- [仕様書](../spec/specification.md): 完全な実装仕様
- [README](../README.md): クイックスタートガイド
- [デモプログラム](../demo.py): 実行可能な使用例
- [テストコード](../dtm_simulator/tests/): 動作確認用テスト

---

## 貢献

バグ報告、機能要望、改善提案は、GitHubのissueまでお願いします。プルリクエストも歓迎します！
