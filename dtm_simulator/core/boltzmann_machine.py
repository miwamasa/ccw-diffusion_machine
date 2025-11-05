"""
Boltzmann Machine implementation based on the DTM paper.

Implements energy-based model with sparse connectivity patterns
as described in Table I of the paper.
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class BoltzmannMachine:
    """
    Boltzmann Machine with 2D grid connectivity.

    Implements energy function from paper Eq. (10):
        E(x) = -β * (Σ(i≠j) xi * Jij * xj + Σi hi * xi)

    Args:
        L: Grid size (L×L)
        connectivity: Connection pattern ("G8", "G12", "G16", "G20", "G24")
        beta: Inverse temperature (default: 1.0)
        seed: Random seed for reproducibility
    """
    L: int
    connectivity: str = "G12"
    beta: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self):
        """Initialize Boltzmann Machine structure."""
        self.N = self.L * self.L
        self.rng = np.random.default_rng(self.seed)

        # Build connectivity matrix
        self.J = self._build_connectivity_matrix()
        self.h = np.zeros(self.N)

        # Color assignment for chromatic Gibbs sampling
        self.colors = self._assign_colors()

        # Track data and latent nodes (can be set externally)
        self.data_nodes = list(range(self.N))
        self.latent_nodes = []

    def _build_connectivity_matrix(self) -> sp.csr_matrix:
        """
        Build sparse connectivity matrix based on paper Table I.

        Connectivity patterns (in (dx, dy) format):
        - G8:  4-neighbor (±1, 0), (0, ±1)
        - G12: G8 + diagonals (±1, ±1)
        - G16: Extended connectivity
        - G20: Extended connectivity
        - G24: Extended connectivity

        Returns:
            Sparse CSR matrix of shape (N, N)
        """
        # Define connectivity patterns
        patterns = {
            "G8": [(0, 1), (1, 0), (0, -1), (-1, 0)],
            "G12": [(0, 1), (1, 0), (0, -1), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)],
            "G16": [(0, 1), (1, 0), (0, -1), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                    (2, 0), (0, 2), (-2, 0), (0, -2)],
            "G20": [(0, 1), (1, 0), (0, -1), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                    (2, 0), (0, 2), (-2, 0), (0, -2),
                    (2, 1), (1, 2), (-2, 1), (-1, 2)],
            "G24": [(0, 1), (1, 0), (0, -1), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1),
                    (2, 0), (0, 2), (-2, 0), (0, -2),
                    (2, 1), (1, 2), (-2, 1), (-1, 2),
                    (-2, -1), (-1, -2), (2, -1), (1, -2)],
        }

        if self.connectivity not in patterns:
            raise ValueError(f"Unknown connectivity: {self.connectivity}")

        offsets = patterns[self.connectivity]

        # Build sparse matrix
        rows, cols, data = [], [], []

        for i in range(self.L):
            for j in range(self.L):
                idx = i * self.L + j

                for di, dj in offsets:
                    ni, nj = i + di, j + dj

                    # Check bounds
                    if 0 <= ni < self.L and 0 <= nj < self.L:
                        nidx = ni * self.L + nj
                        rows.append(idx)
                        cols.append(nidx)
                        # Initialize with small random weights
                        data.append(self.rng.normal(0, 0.1))

        J = sp.csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        # Make symmetric
        J = (J + J.T) / 2

        return J

    def _assign_colors(self) -> np.ndarray:
        """
        Assign 2 colors for chromatic Gibbs sampling.
        Uses checkerboard pattern for 2D grid.

        Returns:
            Color assignment array of shape (N,)
        """
        colors = np.zeros((self.L, self.L), dtype=int)
        colors[1::2, ::2] = 1
        colors[::2, 1::2] = 1
        return colors.flatten()

    def energy(self, x: np.ndarray) -> float:
        """
        Calculate energy using paper Eq. (10).

        E(x) = -β * (Σ(i≠j) xi * Jij * xj + Σi hi * xi)

        Args:
            x: State vector of shape (N,) with values in {-1, +1}

        Returns:
            Energy value (scalar)

        Note:
            Lower energy indicates more stable states
        """
        # Interaction term: -β * Σ xi * Jij * xj / 2
        interaction = -self.beta * (x @ self.J @ x) / 2

        # Bias term: -β * Σ hi * xi
        bias = -self.beta * (self.h @ x)

        return interaction + bias

    def conditional_prob(self, x: np.ndarray, i: int) -> float:
        """
        Calculate conditional probability for node i using paper Eq. (11).

        P(xi = +1 | X[-i]) = σ(2β * (Σj Jij*xj + hi))

        where σ(x) = 1/(1+exp(-x)) is the sigmoid function

        Args:
            x: Current state vector (N,)
            i: Node index

        Returns:
            Probability that xi = +1
        """
        # Sum over neighbors: Σj Jij * xj
        neighbors_sum = self.J[i, :].toarray().flatten() @ x

        # Logit: 2β * (neighbors_sum + hi)
        logit = 2 * self.beta * (neighbors_sum + self.h[i])

        # Sigmoid function
        return 1.0 / (1.0 + np.exp(-logit))

    def gibbs_step(self, x: np.ndarray, color: int) -> np.ndarray:
        """
        Perform one Gibbs sampling step for a color group.

        Updates all nodes of the specified color in parallel.

        Args:
            x: Current state vector (N,) with values in {-1, +1}
            color: Color group to update (0 or 1)

        Returns:
            Updated state vector (N,)

        Note:
            Implements chromatic Gibbs sampling for parallel updates
        """
        x_new = x.copy()
        indices = np.where(self.colors == color)[0]

        for i in indices:
            p_i = self.conditional_prob(x_new, i)
            x_new[i] = 1 if self.rng.random() < p_i else -1

        return x_new

    def sample(self, x_init: np.ndarray, num_steps: int,
               record_trajectory: bool = False) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Generate samples using Gibbs sampling.

        Args:
            x_init: Initial state vector (N,)
            num_steps: Number of sampling iterations
            record_trajectory: Whether to record all intermediate states

        Returns:
            Final state and optionally trajectory of states

        Note:
            Each step alternates between updating color 0 and color 1
        """
        x = x_init.copy()
        trajectory = [x.copy()] if record_trajectory else None

        for _ in range(num_steps):
            x = self.gibbs_step(x, color=0)
            x = self.gibbs_step(x, color=1)

            if record_trajectory:
                trajectory.append(x.copy())

        if record_trajectory:
            return x, trajectory
        else:
            return x, None

    def set_bias(self, h: np.ndarray):
        """Set bias vector for the Boltzmann machine."""
        if h.shape[0] != self.N:
            raise ValueError(f"Bias shape {h.shape} doesn't match N={self.N}")
        self.h = h.copy()

    def set_coupling(self, J: sp.csr_matrix):
        """Set coupling matrix for the Boltzmann machine."""
        if J.shape != (self.N, self.N):
            raise ValueError(f"Coupling shape {J.shape} doesn't match ({self.N}, {self.N})")
        self.J = J.copy()
