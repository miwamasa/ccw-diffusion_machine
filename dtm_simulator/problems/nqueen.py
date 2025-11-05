"""
N-Queen problem encoding for DTM.

Implements N-Queen problem as a constraint satisfaction problem.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple
from dtm_simulator.problems.base import ConstraintProblem


class NQueenProblem(ConstraintProblem):
    """
    N-Queen problem encoder.

    Encodes N-Queen using binary variables x[i,j] ∈ {0,1}
    where x[i,j]=1 means a queen is placed at position (i,j).

    For DTM, we convert to {-1, +1} representation.

    Args:
        N: Board size (N×N)
        penalty_weights: Weights for different constraints (row, col, diag1, diag2)
    """

    def __init__(self,
                 N: int = 8,
                 penalty_weights: Tuple[float, float, float, float] = (1.0, 1.0, 0.5, 0.5)):
        if N < 4:
            raise ValueError("N must be at least 4")

        self.board_size = N
        self.beta_row, self.beta_col, self.beta_diag1, self.beta_diag2 = penalty_weights

        # Total variables: N×N
        self.N = N * N

    def get_num_variables(self) -> int:
        """Return total number of binary variables."""
        return self.N

    def _idx(self, i: int, j: int) -> int:
        """Convert 2D index (i,j) to flat index."""
        return i * self.board_size + j

    def _decode_idx(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to 2D index (i,j)."""
        i = idx // self.board_size
        j = idx % self.board_size
        return i, j

    def _binary_to_spin(self, x_binary: np.ndarray) -> np.ndarray:
        """Convert {0,1} to {-1,+1}."""
        return 2 * x_binary - 1

    def _spin_to_binary(self, x_spin: np.ndarray) -> np.ndarray:
        """Convert {-1,+1} to {0,1}."""
        return ((x_spin + 1) / 2).astype(int)

    def energy_function(self, x: np.ndarray) -> float:
        """
        Compute energy using constraint violations.

        E_total = β₁*E_row + β₂*E_col + β₃*E_diag1 + β₄*E_diag2

        Args:
            x: State vector in {-1, +1}

        Returns:
            Energy value (lower is better)
        """
        # Convert to binary {0, 1}
        x_bin = self._spin_to_binary(x)
        board = x_bin.reshape(self.board_size, self.board_size)

        energy = 0.0

        # Row constraint: exactly one queen per row
        for i in range(self.board_size):
            constraint = np.sum(board[i, :]) - 1
            energy += self.beta_row * constraint ** 2

        # Column constraint: exactly one queen per column
        for j in range(self.board_size):
            constraint = np.sum(board[:, j]) - 1
            energy += self.beta_col * constraint ** 2

        # Diagonal constraints: at most one queen per diagonal
        # Main diagonals (top-left to bottom-right)
        for d in range(-(self.board_size - 1), self.board_size):
            diag = np.diag(board, k=d)
            if len(diag) > 1:
                violation = max(0, np.sum(diag) - 1)
                energy += self.beta_diag1 * violation ** 2

        # Anti-diagonals (top-right to bottom-left)
        board_flipped = np.fliplr(board)
        for d in range(-(self.board_size - 1), self.board_size):
            diag = np.diag(board_flipped, k=d)
            if len(diag) > 1:
                violation = max(0, np.sum(diag) - 1)
                energy += self.beta_diag2 * violation ** 2

        return energy

    def decode_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Decode binary variables to N×N board.

        Args:
            x: State vector in {-1, +1}

        Returns:
            N×N array with 1 for queens, 0 for empty
        """
        x_bin = self._spin_to_binary(x)
        return x_bin.reshape(self.board_size, self.board_size)

    def check_constraints(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Check constraint satisfaction.

        Args:
            x: State vector in {-1, +1}

        Returns:
            Tuple of (satisfied_constraints, total_constraints)
        """
        x_bin = self._spin_to_binary(x)
        board = x_bin.reshape(self.board_size, self.board_size)

        satisfied = 0
        total = 0

        # Row constraints (N total)
        for i in range(self.board_size):
            total += 1
            if np.sum(board[i, :]) == 1:
                satisfied += 1

        # Column constraints (N total)
        for j in range(self.board_size):
            total += 1
            if np.sum(board[:, j]) == 1:
                satisfied += 1

        # Diagonal constraints (2N-2 main diagonals)
        for d in range(-(self.board_size - 1), self.board_size):
            diag = np.diag(board, k=d)
            if len(diag) > 1:
                total += 1
                if np.sum(diag) <= 1:
                    satisfied += 1

        # Anti-diagonal constraints (2N-2 anti-diagonals)
        board_flipped = np.fliplr(board)
        for d in range(-(self.board_size - 1), self.board_size):
            diag = np.diag(board_flipped, k=d)
            if len(diag) > 1:
                total += 1
                if np.sum(diag) <= 1:
                    satisfied += 1

        return satisfied, total

    def get_bias_vector(self, N: int) -> np.ndarray:
        """
        Get bias vector that encourages valid queen placements.

        Args:
            N: Number of variables

        Returns:
            Bias vector
        """
        # Slight positive bias to encourage queen placement
        # (since we need exactly N queens on the board)
        return np.ones(N) * 0.1

    def format_solution(self, board: np.ndarray) -> str:
        """
        Format solution as a readable string.

        Args:
            board: N×N array

        Returns:
            Formatted string with Q for queens, . for empty
        """
        lines = []
        lines.append("+" + "-" * (self.board_size * 2 - 1) + "+")

        for i in range(self.board_size):
            row_str = "|"
            for j in range(self.board_size):
                row_str += "Q" if board[i, j] == 1 else "."
                if j < self.board_size - 1:
                    row_str += " "
            row_str += "|"
            lines.append(row_str)

        lines.append("+" + "-" * (self.board_size * 2 - 1) + "+")
        return "\n".join(lines)

    def count_violations(self, board: np.ndarray) -> dict:
        """
        Count violations for each constraint type.

        Args:
            board: N×N array

        Returns:
            Dictionary with violation counts
        """
        violations = {
            "row": 0,
            "col": 0,
            "diag1": 0,
            "diag2": 0
        }

        # Row violations
        for i in range(self.board_size):
            if np.sum(board[i, :]) != 1:
                violations["row"] += 1

        # Column violations
        for j in range(self.board_size):
            if np.sum(board[:, j]) != 1:
                violations["col"] += 1

        # Diagonal violations
        for d in range(-(self.board_size - 1), self.board_size):
            diag = np.diag(board, k=d)
            if len(diag) > 1 and np.sum(diag) > 1:
                violations["diag1"] += 1

        # Anti-diagonal violations
        board_flipped = np.fliplr(board)
        for d in range(-(self.board_size - 1), self.board_size):
            diag = np.diag(board_flipped, k=d)
            if len(diag) > 1 and np.sum(diag) > 1:
                violations["diag2"] += 1

        return violations
