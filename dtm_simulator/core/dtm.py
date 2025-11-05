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
              verbose: bool = False, use_annealing: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Solve a constraint satisfaction problem using Metropolis-Hastings sampling.

        Args:
            problem: Problem instance with energy_function method
            max_steps: Maximum number of sampling steps
            verbose: Whether to print progress
            use_annealing: Whether to use simulated annealing

        Returns:
            Tuple of (solution, info_dict)
        """
        # Get problem dimensions
        N = problem.get_num_variables()

        # Initialize with problem-specific initialization
        x_init = self._initialize_from_problem(problem, N)

        # Metropolis-Hastings sampling with simulated annealing
        best_x = x_init.copy()
        best_energy = problem.energy_function(x_init)
        energies = []
        acceptance_rate = []

        x = x_init.copy()
        current_energy = best_energy

        for step in range(max_steps):
            # Temperature schedule (simulated annealing)
            if use_annealing:
                temperature = self._get_temperature(step, max_steps)
            else:
                temperature = 1.0

            # Propose a change (flip a random bit)
            x_new = x.copy()
            flip_idx = self.rng.integers(0, N)
            x_new[flip_idx] = -x_new[flip_idx]

            # Evaluate new energy
            new_energy = problem.energy_function(x_new)

            # Metropolis-Hastings acceptance criterion
            delta_E = new_energy - current_energy
            if delta_E < 0 or self.rng.random() < np.exp(-delta_E / temperature):
                x = x_new
                current_energy = new_energy
                accepted = True
            else:
                accepted = False

            acceptance_rate.append(1.0 if accepted else 0.0)
            energies.append(current_energy)

            # Track best solution
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()

            if verbose and step % 500 == 0:
                recent_accept = np.mean(acceptance_rate[-100:]) if len(acceptance_rate) >= 100 else np.mean(acceptance_rate)
                sat_rate = problem.satisfaction_rate(best_x)
                print(f"Step {step}: Energy = {current_energy:.4f}, Best = {best_energy:.4f}, "
                      f"Temp = {temperature:.4f}, Accept = {recent_accept:.2%}, Sat = {sat_rate:.1%}")

        info = {
            "energies": energies,
            "best_energy": best_energy,
            "final_energy": energies[-1],
            "num_steps": max_steps,
            "acceptance_rate": np.mean(acceptance_rate)
        }

        return best_x, info

    def _initialize_from_problem(self, problem, N: int) -> np.ndarray:
        """
        Initialize state with problem-specific hints.

        Args:
            problem: Problem instance
            N: Number of variables

        Returns:
            Initial state vector
        """
        # Start with random state
        x = self.rng.choice([-1, 1], size=N)

        # Apply problem bias to guide initialization
        bias = problem.get_bias_vector(N)

        # Set variables with strong bias
        strong_bias_mask = np.abs(bias) > 5.0
        x[strong_bias_mask] = np.sign(bias[strong_bias_mask])

        return x

    def _get_temperature(self, step: int, max_steps: int) -> float:
        """
        Get temperature for simulated annealing.

        Uses exponential cooling schedule.

        Args:
            step: Current step
            max_steps: Total steps

        Returns:
            Temperature value
        """
        # Initial and final temperatures
        T_init = 10.0
        T_final = 0.01

        # Exponential cooling
        progress = step / max_steps
        temperature = T_init * (T_final / T_init) ** progress

        return temperature

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
