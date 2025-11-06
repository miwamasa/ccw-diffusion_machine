# DTM Hardware Simulator

Denoising Thermodynamic Models (DTM) のハードウェアシミュレータ実装

## 概要

このプロジェクトは、論文「An efficient probabilistic hardware architecture for diffusion-like models」で提案されたDTMのハードウェア動作を模擬するシミュレータです。Sudoku、N-queen問題などの制約充足問題の求解に対応しています。

## 主な機能

### コアコンポーネント
- **Boltzmann Machine**: エネルギーベースモデル、疎結合グラフ (G8, G12, G16, G20, G24)
- **Gibbs Sampler**: Chromatic Gibbs sampling、自己相関計算
- **Forward/Reverse Process**: ノイズ付加とデノイジング
- **DTM エンジン**: 多層EBMチェーン

### ハードウェアシミュレーション
- **RNGシミュレータ**: シグモイドバイアス付きベルヌーイサンプリング (論文 Eq. D6)
- **エネルギーモデル**: ハードウェアエネルギー消費推定 (論文 Eq. D12-D17)
- **バイアス回路**: 確率-電圧変換

### 問題インターフェース
- **Sudoku**: 9×9 Sudoku問題のエンコーダ/デコーダ
- **N-Queen**: N-Queen問題のエンコーダ/デコーダ
- **Potts Model**: グラフ彩色、クラスタリング、画像セグメンテーション
- **カスタム問題**: 抽象基底クラスによる拡張可能な設計

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/miwamasa/ccw-diffusion_machine.git
cd ccw-diffusion_machine

# 依存パッケージのインストール
pip install -r requirements.txt
```

## クイックスタート

### デモの実行

**メインデモ（N-Queen & Sudoku）:**
```bash
python demo.py
```

デモでは以下を実行します：
1. ハードウェアコンポーネント（RNG、Bias Circuit、BM）のシミュレーション
2. 8-Queen問題の求解
3. Sudoku問題の求解
4. エネルギー消費統計の表示

**Potts Modelデモ（グラフ彩色）:**
```bash
python examples/potts_model_demo.py
```

Potts Modelデモでは以下を実行します：
1. サイクルグラフの彩色
2. 2Dグリッドグラフの彩色
3. ランダムグラフの彩色
4. ハードウェアエネルギー分析

**拡散プロセスデモ（Forward/Reverse Process）:**
```bash
python examples/diffusion_process_demo.py
```

拡散プロセスデモでは以下を実行します：
1. Forward Process: クリーンなパターンにノイズを段階的に追加
2. Reverse Process: ノイズからクリーンなパターンへのデノイジング
3. パターン生成: ランダムノイズから構造化パターンの生成
4. 複数パターンの比較: チェッカーボード、縞模様、十字パターン

このデモは、DTMの**本来の用途**である生成モデルとしての動作を示します。

### テストの実行

```bash
pytest dtm_simulator/tests/test_basic.py -v
```

## 使用例

### Sudoku問題の求解

```python
from dtm_simulator.core.dtm import DTM, DTMConfig
from dtm_simulator.problems.sudoku import SudokuProblem

# Sudoku問題を読み込む
puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
problem = SudokuProblem.from_string(puzzle_str)

# DTM求解器を作成
config = DTMConfig(num_layers=4, grid_size=27, K_infer=150, beta=2.0)
dtm = DTM(config)

# 求解
solution_x, info = dtm.solve(problem, max_steps=3000)
solution = problem.decode_solution(solution_x)

# 結果を表示
print(problem.format_solution(solution))
print(f"Constraint satisfaction: {problem.satisfaction_rate(solution_x)*100:.1f}%")
```

### N-Queen問題の求解

```python
from dtm_simulator.core.dtm import DTM, DTMConfig
from dtm_simulator.problems.nqueen import NQueenProblem

# 8-Queen問題を作成
problem = NQueenProblem(N=8)

# DTM求解器を作成
config = DTMConfig(num_layers=4, grid_size=8, K_infer=100, beta=1.5)
dtm = DTM(config)

# 求解
solution_x, info = dtm.solve(problem, max_steps=2000)
board = problem.decode_solution(solution_x)

# 結果を表示
print(problem.format_solution(board))
print(f"Constraint satisfaction: {problem.satisfaction_rate(solution_x)*100:.1f}%")
```

### Potts Model（グラフ彩色）

```python
from dtm_simulator.core.dtm import DTM, DTMConfig
from dtm_simulator.problems.potts import PottsModel

# サイクルグラフの彩色問題を作成
problem = PottsModel.create_cycle_graph(num_nodes=6, num_colors=3)

# または2Dグリッドグラフ
# problem = PottsModel.create_grid_graph(rows=3, cols=3, num_colors=4)

# またはランダムグラフ
# problem = PottsModel.create_random_graph(num_nodes=10, num_colors=3, edge_probability=0.3)

# DTM求解器を作成
config = DTMConfig(num_layers=2, grid_size=5, K_infer=50, beta=1.0)
dtm = DTM(config)

# 求解
solution_x, info = dtm.solve(problem, max_steps=10000)
states = problem.decode_solution(solution_x)

# 結果を表示
print(problem.format_solution(states))
print(f"Constraint satisfaction: {problem.satisfaction_rate(solution_x)*100:.1f}%")
```

### エネルギー消費の計算

```python
from dtm_simulator.hardware.energy_model import EnergyModel

energy_model = EnergyModel()

# エネルギー消費の計算
breakdown = energy_model.compute_energy_breakdown(
    T=8,    # レイヤー数
    K=250,  # サンプリングステップ数
    N=100   # 変数数
)

print(f"Total energy: {energy_model.format_energy(breakdown['total'])}")
print(f"RNG: {energy_model.format_energy(breakdown['rng'])}")
print(f"Bias circuits: {energy_model.format_energy(breakdown['bias'])}")

# GPU比較
gpu_speedup = energy_model.compare_with_gpu(
    dtm_energy=breakdown['total'],
    problem_size=100
)
print(f"Energy efficiency vs GPU: {gpu_speedup:.1f}×")
```

## プロジェクト構造

```
dtm_simulator/
├── core/                    # コアコンポーネント
│   ├── boltzmann_machine.py
│   ├── gibbs_sampler.py
│   ├── forward_process.py
│   ├── reverse_process.py
│   └── dtm.py
├── hardware/                # ハードウェアシミュレーション
│   ├── rng_simulator.py
│   ├── energy_model.py
│   └── bias_circuit.py
├── problems/                # 問題インターフェース
│   ├── base.py
│   ├── sudoku.py
│   └── nqueen.py
├── training/                # トレーニングコンポーネント（今後実装予定）
├── utils/                   # ユーティリティ（今後実装予定）
└── tests/                   # テストスイート
    └── test_basic.py
```

## 詳細仕様

詳細な実装仕様については、以下のドキュメントを参照してください：

- [spec/specification.md](spec/specification.md) - 完全な仕様書

## パフォーマンス

デモ実行時の参考パフォーマンス：

- **8-Queen問題**: 約6-7秒 (2000ステップ)
- **Sudoku問題**: 約90-120秒 (3000ステップ)
- **推定エネルギー効率**: GPU比で約200-300倍

注: 実際の性能は問題の難易度、パラメータ設定、ハードウェア環境により変動します。

## 制限事項

現在の実装には以下の制限があります：

1. **学習機能**: DTMのトレーニング機能は未実装（推論のみ）
2. **最適化**: デモ用に簡略化された実装（完全な解を保証しません）
3. **スケーラビリティ**: 大規模問題ではサンプリングステップ数の増加が必要

## ライセンス

MIT License

## 参考文献

- 論文: "An efficient probabilistic hardware architecture for diffusion-like models"

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 連絡先

問題や質問がある場合は、GitHubのissueを作成してください。
