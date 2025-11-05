"""
Abstract base class for constraint satisfaction problems.
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional


class ConstraintProblem(ABC):
    """
    Abstract base class for constraint satisfaction problems.

    Defines the interface for encoding problems as energy functions
    for the DTM solver.
    """

    @abstractmethod
    def get_num_variables(self) -> int:
        """
        Get total number of binary variables.

        Returns:
            Number of variables
        """
        pass

    @abstractmethod
    def energy_function(self, x: np.ndarray) -> float:
        """
        Compute energy of a configuration.

        Lower energy indicates better constraint satisfaction.

        Args:
            x: Binary state vector

        Returns:
            Energy value
        """
        pass

    @abstractmethod
    def decode_solution(self, x: np.ndarray) -> any:
        """
        Decode binary variables into problem solution.

        Args:
            x: Binary state vector

        Returns:
            Problem-specific solution representation
        """
        pass

    @abstractmethod
    def check_constraints(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Check constraint satisfaction.

        Args:
            x: Binary state vector

        Returns:
            Tuple of (satisfied_constraints, total_constraints)
        """
        pass

    def get_coupling_matrix(self, N: int) -> Optional[sp.csr_matrix]:
        """
        Get coupling matrix J for the problem.

        Args:
            N: Number of variables

        Returns:
            Sparse coupling matrix or None
        """
        return None

    def get_bias_vector(self, N: int) -> np.ndarray:
        """
        Get bias vector h for the problem.

        Args:
            N: Number of variables

        Returns:
            Bias vector
        """
        return np.zeros(N)

    def satisfaction_rate(self, x: np.ndarray) -> float:
        """
        Compute constraint satisfaction rate.

        Args:
            x: Binary state vector

        Returns:
            Satisfaction rate (0 to 1)
        """
        satisfied, total = self.check_constraints(x)
        if total == 0:
            return 1.0
        return satisfied / total

    def is_valid_solution(self, x: np.ndarray) -> bool:
        """
        Check if configuration is a valid solution.

        Args:
            x: Binary state vector

        Returns:
            True if all constraints are satisfied
        """
        satisfied, total = self.check_constraints(x)
        return satisfied == total
