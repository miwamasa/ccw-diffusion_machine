"""
Sudoku problem encoding for DTM.

Implements 9x9 Sudoku as a constraint satisfaction problem.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional
from dtm_simulator.problems.base import ConstraintProblem


class SudokuProblem(ConstraintProblem):
    """
    Sudoku problem encoder.

    Encodes 9×9 Sudoku using binary variables x[i,j,k] ∈ {0,1}
    where x[i,j,k]=1 means cell (i,j) contains digit k (1-9).

    For DTM, we convert to {-1, +1} representation.

    Args:
        puzzle: 9×9 array with given digits (0 for empty cells)
        penalty_weights: Weights for different constraints (cell, row, col, block)
    """

    def __init__(self,
                 puzzle: np.ndarray,
                 penalty_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)):
        if puzzle.shape != (9, 9):
            raise ValueError("Puzzle must be 9×9")

        self.puzzle = puzzle.copy()
        self.alpha_cell, self.alpha_row, self.alpha_col, self.alpha_block = penalty_weights

        # Total variables: 9×9×9 = 729
        self.N = 9 * 9 * 9

        # Create mask for given cells
        self.given_mask = (puzzle > 0)

    @classmethod
    def from_string(cls, puzzle_string: str):
        """
        Create Sudoku problem from string.

        Args:
            puzzle_string: String of 81 characters (0 for empty, 1-9 for givens)

        Returns:
            SudokuProblem instance
        """
        puzzle_string = puzzle_string.replace('.', '0').replace(' ', '')
        if len(puzzle_string) != 81:
            raise ValueError("Puzzle string must have 81 characters")

        puzzle = np.array([int(c) for c in puzzle_string]).reshape(9, 9)
        return cls(puzzle)

    def get_num_variables(self) -> int:
        """Return total number of binary variables."""
        return self.N

    def _idx(self, i: int, j: int, k: int) -> int:
        """Convert 3D index (i,j,k) to flat index."""
        return i * 81 + j * 9 + k

    def _decode_idx(self, idx: int) -> Tuple[int, int, int]:
        """Convert flat index to 3D index (i,j,k)."""
        i = idx // 81
        j = (idx % 81) // 9
        k = idx % 9
        return i, j, k

    def _binary_to_spin(self, x_binary: np.ndarray) -> np.ndarray:
        """Convert {0,1} to {-1,+1}."""
        return 2 * x_binary - 1

    def _spin_to_binary(self, x_spin: np.ndarray) -> np.ndarray:
        """Convert {-1,+1} to {0,1}."""
        return ((x_spin + 1) / 2).astype(int)

    def energy_function(self, x: np.ndarray) -> float:
        """
        Compute energy using constraint violations.

        E_total = α₁*E_cell + α₂*E_row + α₃*E_col + α₄*E_block

        Args:
            x: State vector in {-1, +1}

        Returns:
            Energy value (lower is better)
        """
        # Convert to binary {0, 1}
        x_bin = self._spin_to_binary(x)
        x_3d = x_bin.reshape(9, 9, 9)

        energy = 0.0

        # Cell constraint: each cell has exactly one digit
        for i in range(9):
            for j in range(9):
                constraint = np.sum(x_3d[i, j, :]) - 1
                energy += self.alpha_cell * constraint ** 2

        # Row constraint: each digit appears once per row
        for i in range(9):
            for k in range(9):
                constraint = np.sum(x_3d[i, :, k]) - 1
                energy += self.alpha_row * constraint ** 2

        # Column constraint: each digit appears once per column
        for j in range(9):
            for k in range(9):
                constraint = np.sum(x_3d[:, j, k]) - 1
                energy += self.alpha_col * constraint ** 2

        # Block constraint: each digit appears once per 3×3 block
        for bi in range(3):
            for bj in range(3):
                for k in range(9):
                    block = x_3d[bi*3:(bi+1)*3, bj*3:(bj+1)*3, k]
                    constraint = np.sum(block) - 1
                    energy += self.alpha_block * constraint ** 2

        return energy

    def decode_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Decode binary variables to 9×9 Sudoku grid.

        Args:
            x: State vector in {-1, +1}

        Returns:
            9×9 array with digits (0 if no clear assignment)
        """
        x_bin = self._spin_to_binary(x)
        x_3d = x_bin.reshape(9, 9, 9)

        solution = np.zeros((9, 9), dtype=int)

        for i in range(9):
            for j in range(9):
                # Find digit with highest activation
                digits = x_3d[i, j, :]
                if np.sum(digits) > 0:
                    k = np.argmax(digits)
                    solution[i, j] = k + 1  # Digits are 1-9

        return solution

    def check_constraints(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Check constraint satisfaction.

        Args:
            x: State vector in {-1, +1}

        Returns:
            Tuple of (satisfied_constraints, total_constraints)
        """
        x_bin = self._spin_to_binary(x)
        x_3d = x_bin.reshape(9, 9, 9)

        satisfied = 0
        total = 0

        # Cell constraints (81 total)
        for i in range(9):
            for j in range(9):
                total += 1
                if np.sum(x_3d[i, j, :]) == 1:
                    satisfied += 1

        # Row constraints (81 total: 9 rows × 9 digits)
        for i in range(9):
            for k in range(9):
                total += 1
                if np.sum(x_3d[i, :, k]) == 1:
                    satisfied += 1

        # Column constraints (81 total)
        for j in range(9):
            for k in range(9):
                total += 1
                if np.sum(x_3d[:, j, k]) == 1:
                    satisfied += 1

        # Block constraints (81 total: 9 blocks × 9 digits)
        for bi in range(3):
            for bj in range(3):
                for k in range(9):
                    total += 1
                    block = x_3d[bi*3:(bi+1)*3, bj*3:(bj+1)*3, k]
                    if np.sum(block) == 1:
                        satisfied += 1

        return satisfied, total

    def get_bias_vector(self, N: int) -> np.ndarray:
        """
        Get bias vector based on given cells.

        Args:
            N: Number of variables

        Returns:
            Bias vector that encourages given digits
        """
        bias = np.zeros(N)

        for i in range(9):
            for j in range(9):
                if self.given_mask[i, j]:
                    digit = self.puzzle[i, j]
                    k = digit - 1  # Convert to 0-indexed
                    idx = self._idx(i, j, k)
                    # Strong positive bias for given digits
                    bias[idx] = 10.0
                    # Negative bias for other digits in this cell
                    for k_other in range(9):
                        if k_other != k:
                            idx_other = self._idx(i, j, k_other)
                            bias[idx_other] = -10.0

        return bias

    def format_solution(self, solution: np.ndarray) -> str:
        """
        Format solution as a readable string.

        Args:
            solution: 9×9 array

        Returns:
            Formatted string
        """
        lines = []
        for i in range(9):
            if i % 3 == 0 and i > 0:
                lines.append("------+-------+------")
            row_str = ""
            for j in range(9):
                if j % 3 == 0 and j > 0:
                    row_str += "| "
                digit = solution[i, j]
                row_str += str(digit) if digit > 0 else "."
                row_str += " "
            lines.append(row_str)
        return "\n".join(lines)
