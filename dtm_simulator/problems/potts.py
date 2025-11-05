"""
Potts Model implementation for DTM.

Potts model is an energy-based model for categorical variables,
commonly used for graph coloring, clustering, and image segmentation.

Based on the model from:
https://github.com/extropic-ai/thrml/blob/main/examples/00_probabilistic_computing.ipynb
"""

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, List
from dtm_simulator.problems.base import ConstraintProblem


class PottsModel(ConstraintProblem):
    """
    Potts Model for categorical variables.

    The Potts model assigns each node in a graph one of Q states,
    with energy penalizing neighboring nodes having the same state
    (for graph coloring) or different states (for clustering).

    Energy function:
        E(x) = -Σᵢ W¹[xᵢ] - Σ_{(i,j)∈E} W²[xᵢ, xⱼ]

    For graph coloring:
        W²[i, j] = -∞ if colors are same (hard constraint)
        W²[i, j] = 0 otherwise

    Args:
        num_nodes: Number of nodes in the graph
        num_states: Number of possible states per node (Q)
        edges: List of edges as (i, j) tuples
        interaction_type: "coloring" or "ferromagnetic"
        W1: Optional bias weights for each state (num_nodes, num_states)
        W2: Optional interaction weights (num_states, num_states)
    """

    def __init__(self,
                 num_nodes: int,
                 num_states: int,
                 edges: List[Tuple[int, int]],
                 interaction_type: str = "coloring",
                 W1: Optional[np.ndarray] = None,
                 W2: Optional[np.ndarray] = None):
        self.num_nodes = num_nodes
        self.num_states = num_states  # Q in standard notation
        self.edges = edges
        self.interaction_type = interaction_type

        # Total binary variables: N * Q (one-hot encoding)
        self.N = num_nodes * num_states

        # Bias weights (if not provided, use uniform)
        if W1 is None:
            self.W1 = np.zeros((num_nodes, num_states))
        else:
            self.W1 = W1

        # Interaction weights
        if W2 is None:
            self.W2 = self._default_interaction_matrix()
        else:
            self.W2 = W2

    def _default_interaction_matrix(self) -> np.ndarray:
        """
        Create default interaction matrix based on problem type.

        Returns:
            Interaction matrix of shape (num_states, num_states)
        """
        if self.interaction_type == "coloring":
            # Graph coloring: penalize same colors on adjacent nodes
            W2 = np.ones((self.num_states, self.num_states))
            np.fill_diagonal(W2, 0)  # Same color → high penalty
            return -W2  # Negative because we minimize energy

        elif self.interaction_type == "ferromagnetic":
            # Clustering: prefer same states on adjacent nodes
            W2 = np.eye(self.num_states)
            return W2

        else:
            raise ValueError(f"Unknown interaction type: {self.interaction_type}")

    def get_num_variables(self) -> int:
        """Return total number of binary variables."""
        return self.N

    def _idx(self, node: int, state: int) -> int:
        """Convert (node, state) to flat index."""
        return node * self.num_states + state

    def _decode_idx(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to (node, state)."""
        node = idx // self.num_states
        state = idx % self.num_states
        return node, state

    def _spin_to_onehot(self, x_spin: np.ndarray) -> np.ndarray:
        """
        Convert spin encoding {-1, +1}^N to one-hot encoding.

        Args:
            x_spin: Spin vector (N,)

        Returns:
            One-hot matrix (num_nodes, num_states)
        """
        x_bin = ((x_spin + 1) / 2).astype(int)
        x_matrix = x_bin.reshape(self.num_nodes, self.num_states)
        return x_matrix

    def _onehot_to_state(self, x_onehot: np.ndarray) -> np.ndarray:
        """
        Convert one-hot encoding to state assignment.

        Args:
            x_onehot: One-hot matrix (num_nodes, num_states)

        Returns:
            State vector (num_nodes,) with values in [0, num_states-1]
        """
        states = np.argmax(x_onehot, axis=1)
        return states

    def energy_function(self, x: np.ndarray) -> float:
        """
        Compute Potts model energy.

        E(x) = -Σᵢ W¹[xᵢ] - Σ_{(i,j)∈E} W²[xᵢ, xⱼ] + penalty for invalid one-hot

        Args:
            x: State vector in {-1, +1}

        Returns:
            Energy value (lower is better)
        """
        x_onehot = self._spin_to_onehot(x)
        energy = 0.0

        # One-hot constraint: each node must have exactly one active state
        for i in range(self.num_nodes):
            constraint = np.sum(x_onehot[i, :]) - 1
            energy += 10.0 * constraint ** 2  # Strong penalty for violation

        # Bias term: -Σᵢ W¹[xᵢ]
        for i in range(self.num_nodes):
            state = np.argmax(x_onehot[i, :])
            energy -= self.W1[i, state]

        # Interaction term: -Σ_{(i,j)∈E} W²[xᵢ, xⱼ]
        for i, j in self.edges:
            state_i = np.argmax(x_onehot[i, :])
            state_j = np.argmax(x_onehot[j, :])
            energy -= self.W2[state_i, state_j]

        return energy

    def decode_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Decode binary variables to state assignment.

        Args:
            x: State vector in {-1, +1}

        Returns:
            State assignment (num_nodes,)
        """
        x_onehot = self._spin_to_onehot(x)
        return self._onehot_to_state(x_onehot)

    def check_constraints(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Check constraint satisfaction.

        Args:
            x: State vector in {-1, +1}

        Returns:
            Tuple of (satisfied_constraints, total_constraints)
        """
        x_onehot = self._spin_to_onehot(x)
        states = self._onehot_to_state(x_onehot)

        satisfied = 0
        total = 0

        # One-hot constraints
        for i in range(self.num_nodes):
            total += 1
            if np.sum(x_onehot[i, :]) == 1:
                satisfied += 1

        # Edge constraints (for graph coloring)
        if self.interaction_type == "coloring":
            for i, j in self.edges:
                total += 1
                if states[i] != states[j]:
                    satisfied += 1

        return satisfied, total

    def get_bias_vector(self, N: int) -> np.ndarray:
        """
        Get bias vector for initialization.

        Encourages one-hot encoding.

        Args:
            N: Number of variables

        Returns:
            Bias vector
        """
        bias = np.zeros(N)

        # Slight bias to encourage at least one state per node
        for i in range(self.num_nodes):
            # Random state gets slight positive bias
            state = np.random.randint(0, self.num_states)
            idx = self._idx(i, state)
            bias[idx] = 0.5

        return bias

    def format_solution(self, states: np.ndarray) -> str:
        """
        Format solution as a readable string.

        Args:
            states: State assignment (num_nodes,)

        Returns:
            Formatted string
        """
        lines = []
        lines.append(f"Potts Model Solution ({self.num_nodes} nodes, {self.num_states} states)")
        lines.append("=" * 50)

        # Node assignments
        lines.append("\nNode assignments:")
        for i in range(min(self.num_nodes, 20)):  # Show first 20 nodes
            lines.append(f"  Node {i:2d}: State {states[i]}")

        if self.num_nodes > 20:
            lines.append(f"  ... ({self.num_nodes - 20} more nodes)")

        # Edge violations (for coloring)
        if self.interaction_type == "coloring":
            violations = []
            for i, j in self.edges:
                if states[i] == states[j]:
                    violations.append((i, j))

            lines.append(f"\nEdge violations: {len(violations)}/{len(self.edges)}")
            if violations and len(violations) <= 10:
                lines.append("Violating edges:")
                for i, j in violations:
                    lines.append(f"  ({i}, {j}): both have state {states[i]}")

        return "\n".join(lines)

    @classmethod
    def create_graph_coloring(cls, num_nodes: int, num_colors: int,
                             edges: List[Tuple[int, int]]):
        """
        Create a graph coloring problem instance.

        Args:
            num_nodes: Number of nodes
            num_colors: Number of colors available
            edges: List of edges

        Returns:
            PottsModel instance
        """
        return cls(
            num_nodes=num_nodes,
            num_states=num_colors,
            edges=edges,
            interaction_type="coloring"
        )

    @classmethod
    def create_random_graph(cls, num_nodes: int, num_colors: int,
                           edge_probability: float = 0.3, seed: int = None):
        """
        Create a random graph coloring problem.

        Args:
            num_nodes: Number of nodes
            num_colors: Number of colors
            edge_probability: Probability of edge between any two nodes
            seed: Random seed

        Returns:
            PottsModel instance
        """
        rng = np.random.default_rng(seed)

        # Generate random edges
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if rng.random() < edge_probability:
                    edges.append((i, j))

        return cls.create_graph_coloring(num_nodes, num_colors, edges)

    @classmethod
    def create_cycle_graph(cls, num_nodes: int, num_colors: int):
        """
        Create a cycle graph coloring problem.

        A cycle graph is a graph where nodes form a ring.

        Args:
            num_nodes: Number of nodes
            num_colors: Number of colors

        Returns:
            PottsModel instance
        """
        edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
        return cls.create_graph_coloring(num_nodes, num_colors, edges)

    @classmethod
    def create_grid_graph(cls, rows: int, cols: int, num_colors: int):
        """
        Create a 2D grid graph coloring problem.

        Args:
            rows: Number of rows
            cols: Number of columns
            num_colors: Number of colors

        Returns:
            PottsModel instance
        """
        num_nodes = rows * cols
        edges = []

        def idx(r, c):
            return r * cols + c

        # Add horizontal edges
        for r in range(rows):
            for c in range(cols - 1):
                edges.append((idx(r, c), idx(r, c + 1)))

        # Add vertical edges
        for r in range(rows - 1):
            for c in range(cols):
                edges.append((idx(r, c), idx(r + 1, c)))

        return cls.create_graph_coloring(num_nodes, num_colors, edges)
