"""
Maze Path Finding Problem for DTM

Encodes maze path-finding as a binary optimization problem.
This problem is expected to have GOOD compatibility with diffusion models
because of its smooth energy landscape and meaningful intermediate states.
"""

import numpy as np
from typing import List, Tuple, Optional
from dtm_simulator.problems.base import ConstraintProblem


class MazeProblem(ConstraintProblem):
    """
    Maze path-finding problem with smooth energy landscape.

    The maze is represented as a 2D grid where:
    - 0: Empty space (can walk)
    - 1: Wall (cannot walk)
    - S: Start position
    - G: Goal position

    Encoding: Each grid cell is a binary variable (1 = on path, -1 = not on path)

    Energy function design (SMOOTH for diffusion models):
    1. Connectivity reward: Encourages connected paths
    2. Wall penalty: Penalizes paths through walls
    3. Endpoint constraint: Must include start and goal
    4. Smoothness reward: Encourages smooth paths (fewer turns)
    5. Path length penalty: Prevents overly long paths
    """

    def __init__(
        self,
        maze: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        beta_connectivity: float = 5.0,
        beta_wall: float = 20.0,
        beta_endpoint: float = 50.0,
        beta_smoothness: float = 2.0,
        beta_length: float = 0.5,
    ):
        """
        Initialize maze problem.

        Args:
            maze: 2D array (0=empty, 1=wall)
            start: (row, col) start position
            goal: (row, col) goal position
            beta_connectivity: Weight for connectivity reward
            beta_wall: Weight for wall penalty
            beta_endpoint: Weight for endpoint constraint
            beta_smoothness: Weight for smoothness reward
            beta_length: Weight for path length penalty
        """
        self.maze = maze
        self.rows, self.cols = maze.shape
        self.start = start
        self.goal = goal
        self.N = self.rows * self.cols

        # Energy weights
        self.beta_connectivity = beta_connectivity
        self.beta_wall = beta_wall
        self.beta_endpoint = beta_endpoint
        self.beta_smoothness = beta_smoothness
        self.beta_length = beta_length

        # Precompute neighbor relationships
        self._build_adjacency()

    def _build_adjacency(self):
        """Build adjacency list for grid connectivity."""
        self.neighbors = {}
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                neighbors = []
                # 4-connected grid (up, down, left, right)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        neighbors.append(nr * self.cols + nc)
                self.neighbors[idx] = neighbors

    def get_num_variables(self) -> int:
        """Get total number of binary variables."""
        return self.N

    def _coord_to_idx(self, row: int, col: int) -> int:
        """Convert (row, col) to flat index."""
        return row * self.cols + col

    def _idx_to_coord(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to (row, col)."""
        return idx // self.cols, idx % self.cols

    def energy_function(self, x: np.ndarray) -> float:
        """
        Compute energy for maze path.

        This energy function is designed to be SMOOTH:
        - Small changes in path lead to small changes in energy
        - Encourages gradual improvement
        - Compatible with diffusion model's denoising process

        Args:
            x: Binary vector {-1, +1}^N representing path

        Returns:
            Energy value (lower is better)
        """
        # Convert spin {-1, +1} to binary {0, 1}
        path = (x + 1) / 2  # 1 = on path, 0 = not on path

        energy = 0.0

        # 1. CONNECTIVITY REWARD (SMOOTH!)
        # Reward adjacent cells that are both on the path
        connectivity = 0.0
        for idx in range(self.N):
            if path[idx] > 0.5:
                for neighbor_idx in self.neighbors[idx]:
                    if path[neighbor_idx] > 0.5:
                        connectivity += 1.0
        # Negative energy = reward (more connected = lower energy)
        energy -= self.beta_connectivity * connectivity

        # 2. WALL PENALTY (LOCAL constraint)
        # Penalize path cells that go through walls
        wall_violations = 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                idx = self._coord_to_idx(r, c)
                if self.maze[r, c] == 1:  # Wall
                    if path[idx] > 0.5:  # Path goes through wall
                        wall_violations += 1.0
        energy += self.beta_wall * wall_violations

        # 3. ENDPOINT CONSTRAINT (HARD but only 2 points)
        # Must include start and goal
        start_idx = self._coord_to_idx(*self.start)
        goal_idx = self._coord_to_idx(*self.goal)
        endpoint_penalty = 0.0
        if path[start_idx] < 0.5:
            endpoint_penalty += 1.0
        if path[goal_idx] < 0.5:
            endpoint_penalty += 1.0
        energy += self.beta_endpoint * endpoint_penalty

        # 4. SMOOTHNESS REWARD (SMOOTH!)
        # Reward straight paths, penalize turns
        # A cell is "smooth" if it has exactly 2 neighbors on the path
        smoothness = 0.0
        for idx in range(self.N):
            if path[idx] > 0.5:
                num_neighbors = sum(
                    1 for n_idx in self.neighbors[idx] if path[n_idx] > 0.5
                )
                if num_neighbors == 2:
                    smoothness += 1.0  # Straight segment
                elif num_neighbors == 1:
                    smoothness += 0.5  # Endpoint (acceptable)
        energy -= self.beta_smoothness * smoothness

        # 5. PATH LENGTH PENALTY (SMOOTH!)
        # Prefer shorter paths (but not too aggressive)
        path_length = np.sum(path)
        energy += self.beta_length * path_length

        return energy

    def bias_vector(self) -> np.ndarray:
        """
        Generate bias vector for problem-specific initialization.

        Bias towards:
        - Start and goal cells (strong positive bias)
        - Cells along straight line from start to goal (weak positive bias)
        - Wall cells (strong negative bias)

        Returns:
            Bias vector h ∈ R^N
        """
        h = np.zeros(self.N)

        # Strong bias for start and goal
        start_idx = self._coord_to_idx(*self.start)
        goal_idx = self._coord_to_idx(*self.goal)
        h[start_idx] = 10.0
        h[goal_idx] = 10.0

        # Bias along straight line from start to goal (Euclidean distance heuristic)
        for r in range(self.rows):
            for c in range(self.cols):
                idx = self._coord_to_idx(r, c)

                # Distance from start and goal
                dist_start = abs(r - self.start[0]) + abs(c - self.start[1])
                dist_goal = abs(r - self.goal[0]) + abs(c - self.goal[1])
                dist_total = abs(self.start[0] - self.goal[0]) + abs(
                    self.start[1] - self.goal[1]
                )

                # Bias cells that are between start and goal
                if dist_start + dist_goal <= dist_total + 4:
                    h[idx] += 2.0 / (1.0 + min(dist_start, dist_goal))

                # Strong negative bias for walls
                if self.maze[r, c] == 1:
                    h[idx] = -20.0

        return h

    def decode_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Convert binary solution to path grid.

        Args:
            x: Binary vector {-1, +1}^N

        Returns:
            2D array representing path
        """
        path = (x + 1) / 2  # Convert to {0, 1}
        return path.reshape(self.rows, self.cols)

    def check_constraints(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Check constraint satisfaction.

        Constraints:
        1. Start and goal on path (2 constraints)
        2. No wall violations (count of wall cells)
        3. Path connectivity (reasonable path length)

        Args:
            x: Binary solution vector

        Returns:
            Tuple of (satisfied_constraints, total_constraints)
        """
        path = self.decode_solution(x)

        satisfied = 0
        total = 0

        # 1. Endpoint constraints (2 constraints)
        total += 2
        if path[self.start] > 0.5:
            satisfied += 1
        if path[self.goal] > 0.5:
            satisfied += 1

        # 2. Wall constraints (one per wall cell)
        wall_cells = np.sum(self.maze == 1)
        total += wall_cells
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze[r, c] == 1:
                    if path[r, c] < 0.5:  # Not on path (correct)
                        satisfied += 1

        # 3. Connectivity constraints (check if path is reasonably connected)
        # Use flood fill from start
        visited = self._flood_fill(path, self.start)
        if self.goal in visited:
            # Path connects start to goal
            satisfied += 1
        total += 1

        return satisfied, total

    def satisfaction_rate(self, x: np.ndarray) -> float:
        """
        Compute constraint satisfaction rate.

        Args:
            x: Binary solution vector

        Returns:
            Satisfaction rate ∈ [0, 1]
        """
        satisfied, total = self.check_constraints(x)
        return satisfied / total if total > 0 else 0.0

    def _flood_fill(
        self, path: np.ndarray, start: Tuple[int, int]
    ) -> set[Tuple[int, int]]:
        """
        Flood fill from start to find connected component.

        Args:
            path: 2D binary array
            start: Starting position

        Returns:
            Set of reachable positions
        """
        visited = set()
        stack = [start]

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                continue
            if path[r, c] < 0.5:  # Not on path
                continue

            visited.add((r, c))

            # Add neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((r + dr, c + dc))

        return visited

    def format_solution(self, path: np.ndarray) -> str:
        """
        Format solution as ASCII art.

        Args:
            path: 2D binary array (solution)

        Returns:
            Formatted string
        """
        lines = []
        lines.append(f"Maze Solution ({self.rows}×{self.cols})")
        lines.append("=" * 50)
        lines.append("")

        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if (r, c) == self.start:
                    row_str += " S "
                elif (r, c) == self.goal:
                    row_str += " G "
                elif self.maze[r, c] == 1:
                    row_str += " # "
                elif path[r, c] > 0.5:
                    row_str += " * "
                else:
                    row_str += " . "
            lines.append(row_str)

        lines.append("")
        lines.append("Legend:")
        lines.append("  S: Start")
        lines.append("  G: Goal")
        lines.append("  #: Wall")
        lines.append("  *: Path")
        lines.append("  .: Empty")

        return "\n".join(lines)

    @classmethod
    def create_simple_maze(cls, size: int = 10) -> "MazeProblem":
        """
        Create a simple maze with scattered walls.

        Args:
            size: Grid size (size × size)

        Returns:
            MazeProblem instance
        """
        maze = np.zeros((size, size), dtype=int)

        # Add some walls
        np.random.seed(42)
        for _ in range(size * 2):
            r, c = np.random.randint(0, size, 2)
            if (r, c) != (0, 0) and (r, c) != (size - 1, size - 1):
                maze[r, c] = 1

        start = (0, 0)
        goal = (size - 1, size - 1)

        return cls(maze, start, goal)

    @classmethod
    def create_corridor_maze(cls, rows: int = 15, cols: int = 15) -> "MazeProblem":
        """
        Create a maze with corridors and obstacles.

        Args:
            rows: Number of rows
            cols: Number of columns

        Returns:
            MazeProblem instance
        """
        maze = np.zeros((rows, cols), dtype=int)

        # Add horizontal walls
        for c in range(cols // 3, 2 * cols // 3):
            maze[rows // 3, c] = 1
            maze[2 * rows // 3, c] = 1

        # Add vertical walls
        for r in range(rows // 4, 3 * rows // 4):
            if r != rows // 2:  # Leave gap
                maze[r, cols // 3] = 1

        start = (0, 0)
        goal = (rows - 1, cols - 1)

        return cls(maze, start, goal)

    @classmethod
    def create_spiral_maze(cls, size: int = 15) -> "MazeProblem":
        """
        Create a spiral maze pattern.

        Args:
            size: Grid size (size × size)

        Returns:
            MazeProblem instance
        """
        maze = np.zeros((size, size), dtype=int)

        # Create spiral walls
        for layer in range(size // 4):
            # Top wall
            for c in range(layer, size - layer):
                if c < size - layer - 1:
                    maze[layer * 2 + 2, c] = 1

            # Right wall
            for r in range(layer * 2 + 2, size - layer * 2 - 2):
                maze[r, size - layer * 2 - 3] = 1

            # Bottom wall
            for c in range(layer * 2 + 2, size - layer * 2 - 2):
                maze[size - layer * 2 - 3, c] = 1

        start = (0, 0)
        goal = (size - 1, size - 1)

        return cls(maze, start, goal)
