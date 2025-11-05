"""
DTM (Denoising Thermodynamic Models) main class.

Integrates forward process, reverse process, and multi-layer EBM chain.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from dtm_simulator.core.boltzmann_machine import BoltzmannMachine
from dtm_simulator.core.forward_process import ForwardProcess
from dtm_simulator.core.reverse_process import ReverseProcess


@dataclass
class DTMConfig:
    """
    Configuration for DTM model.

    Args:
        num_layers: Number of diffusion layers (T)
        grid_size: Size of Boltzmann machine grid (L×L)
        connectivity: Connection pattern ("G8", "G12", etc.)
        K_train: Number of mixing steps during training
        K_infer: Number of mixing steps during inference
        gamma_forward: Forward process noise rate
        beta: Inverse temperature for EBMs
    """
    num_layers: int = 8
    grid_size: int = 10
    connectivity: str = "G12"
    K_train: int = 1000
    K_infer: int = 250
    gamma_forward: float = 1.0
    beta: float = 1.0
    seed: Optional[int] = None


class DTM:
    """
    Denoising Thermodynamic Model.

    Implements the full DTM architecture with multi-layer EBM chain
    for solving constraint satisfaction problems.

    Args:
        config: DTM configuration
    """

    def __init__(self, config: DTMConfig = None):
        if config is None:
            config = DTMConfig()

        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Initialize forward process
        self.forward_process = ForwardProcess(
            num_layers=config.num_layers,
            gamma=config.gamma_forward,
            M=2  # Binary variables
        )

        # Initialize EBM layers
        self.ebm_layers = self._create_ebm_layers()

        # Initialize reverse process
        self.reverse_process = ReverseProcess(
            ebm_layers=self.ebm_layers,
            forward_process=self.forward_process,
            mixing_steps=config.K_infer
        )

    def _create_ebm_layers(self) -> List[BoltzmannMachine]:
        """
        Create Boltzmann machine for each layer.

        Returns:
            List of BoltzmannMachine instances
        """
        ebm_layers = []

        for t in range(self.config.num_layers):
            bm = BoltzmannMachine(
                L=self.config.grid_size,
                connectivity=self.config.connectivity,
                beta=self.config.beta,
                seed=self.rng.integers(0, 2**31) if self.config.seed else None
            )
            ebm_layers.append(bm)

        return ebm_layers

    def solve(self, problem, max_steps: int = 5000,
              verbose: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Solve a constraint satisfaction problem.

        Args:
            problem: Problem instance with energy_function method
            max_steps: Maximum number of sampling steps
            verbose: Whether to print progress

        Returns:
            Tuple of (solution, info_dict)
        """
        # Get problem dimensions
        N = problem.get_num_variables()

        # Adjust grid size if needed
        if N != self.config.grid_size ** 2:
            # Resize or use problem-specific handling
            pass

        # Initialize with random state
        x_init = self.rng.choice([-1, 1], size=N)

        # Set problem-specific biases on EBM layers
        for ebm in self.ebm_layers:
            problem_bias = problem.get_bias_vector(N)
            ebm.set_bias(problem_bias)

        # Sample using the last layer EBM with problem constraints
        best_x = x_init.copy()
        best_energy = float('inf')
        energies = []

        x = x_init.copy()

        for step in range(max_steps):
            # Gibbs sampling step
            x = self.ebm_layers[-1].gibbs_step(x, color=0)
            x = self.ebm_layers[-1].gibbs_step(x, color=1)

            # Evaluate problem energy
            energy = problem.energy_function(x)
            energies.append(energy)

            if energy < best_energy:
                best_energy = energy
                best_x = x.copy()

            if verbose and step % 500 == 0:
                print(f"Step {step}: Energy = {energy:.4f}, Best = {best_energy:.4f}")

        info = {
            "energies": energies,
            "best_energy": best_energy,
            "final_energy": energies[-1],
            "num_steps": max_steps
        }

        return best_x, info

    def denoise(self, x_noisy: np.ndarray,
                noise_level: int = None) -> np.ndarray:
        """
        Denoise a noisy state.

        Args:
            x_noisy: Noisy input state
            noise_level: Noise level (1 to T), if None assumes maximum

        Returns:
            Denoised state
        """
        if noise_level is None:
            noise_level = self.config.num_layers

        # Run reverse process
        trajectory = self.reverse_process.reverse_trajectory(
            x_noisy,
            rng=self.rng,
            record_trajectory=False
        )

        return trajectory[-1]

    def generate_sample(self, num_steps: int = None) -> np.ndarray:
        """
        Generate a new sample from noise.

        Args:
            num_steps: Number of denoising steps (default: num_layers)

        Returns:
            Generated sample
        """
        if num_steps is None:
            num_steps = self.config.num_layers

        # Start from random noise
        N = self.config.grid_size ** 2
        x_noise = self.rng.choice([-1, 1], size=N)

        # Denoise
        return self.denoise(x_noise)

    def evaluate_energy(self, x: np.ndarray, layer: int = -1) -> float:
        """
        Evaluate energy of a state using specified EBM layer.

        Args:
            x: State vector
            layer: Layer index (default: -1 for last layer)

        Returns:
            Energy value
        """
        return self.ebm_layers[layer].energy(x)

    def set_problem_constraints(self, problem):
        """
        Set problem-specific constraints on EBM layers.

        Args:
            problem: Problem instance with constraint information
        """
        N = problem.get_num_variables()

        # Get coupling and bias from problem
        J = problem.get_coupling_matrix(N)
        h = problem.get_bias_vector(N)

        # Set on all layers (or just the last layer for inference)
        for ebm in self.ebm_layers:
            if J is not None:
                ebm.set_coupling(J)
            if h is not None:
                ebm.set_bias(h)
