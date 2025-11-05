"""
Forward process implementation for DTM.

Implements discrete Markov jump process as described in paper Appendix A.1.b
"""

import numpy as np
from typing import Tuple


class ForwardProcess:
    """
    Forward noising process for discrete variables.

    Implements noise schedule from paper Eq. (A20):
        Γ(t) = ln((1 + (M-1)*exp(-γMt)) / (1 - exp(-γMt)))

    Args:
        num_layers: Number of diffusion layers (T)
        gamma: Noise rate parameter
        M: Number of categories (M=2 for binary)
    """

    def __init__(self, num_layers: int = 8, gamma: float = 1.0, M: int = 2):
        self.T = num_layers
        self.gamma = gamma
        self.M = M

        # Precompute noise schedule
        self.noise_schedule = self._compute_noise_schedule()

    def _compute_noise_schedule(self) -> np.ndarray:
        """
        Compute Γ(t) for all time steps according to paper Eq. (A20).

        Returns:
            Array of shape (T+1,) with Γ values for t=0,1,...,T
        """
        t_values = np.arange(self.T + 1) / self.T  # Normalize to [0, 1]

        # Γ(t) = ln((1 + (M-1)*exp(-γMt)) / (1 - exp(-γMt)))
        gamma_Mt = self.gamma * self.M * t_values

        numerator = 1 + (self.M - 1) * np.exp(-gamma_Mt)
        denominator = 1 - np.exp(-gamma_Mt)

        # Handle t=0 case (should be infinity, but we use large value)
        gamma_schedule = np.where(
            t_values > 0,
            np.log(numerator / (denominator + 1e-10)),
            10.0  # Large value for t=0
        )

        return gamma_schedule

    def get_transition_prob(self, t: int) -> float:
        """
        Get flip probability at time step t.

        For binary variables, this is the probability of flipping the bit.

        Args:
            t: Time step (0 to T)

        Returns:
            Flip probability
        """
        if t >= self.T:
            return 0.5  # Maximum noise

        # Compute flip probability from Γ
        gamma_t = self.noise_schedule[t]
        flip_prob = 1.0 / (1.0 + np.exp(gamma_t))

        return flip_prob

    def add_noise(self, x: np.ndarray, t: int, rng: np.random.Generator = None) -> np.ndarray:
        """
        Add noise to state x at time step t.

        For binary variables in {-1, +1}, randomly flip bits with probability p(t).

        Args:
            x: Clean state vector (N,) with values in {-1, +1}
            t: Time step (1 to T)
            rng: Random number generator

        Returns:
            Noisy state vector (N,)
        """
        if rng is None:
            rng = np.random.default_rng()

        if t == 0:
            return x.copy()

        flip_prob = self.get_transition_prob(t)

        # Randomly flip bits
        flip_mask = rng.random(x.shape) < flip_prob
        x_noisy = x.copy()
        x_noisy[flip_mask] = -x_noisy[flip_mask]

        return x_noisy

    def forward_trajectory(self, x_0: np.ndarray,
                          rng: np.random.Generator = None) -> Tuple[list, list]:
        """
        Generate full forward trajectory from clean to noisy.

        Args:
            x_0: Clean initial state (N,)
            rng: Random number generator

        Returns:
            Tuple of (states, noise_levels)
            - states: List of states at each time step
            - noise_levels: List of flip probabilities at each step
        """
        if rng is None:
            rng = np.random.default_rng()

        states = [x_0.copy()]
        noise_levels = [0.0]

        x = x_0.copy()
        for t in range(1, self.T + 1):
            x = self.add_noise(x, t, rng)
            states.append(x.copy())
            noise_levels.append(self.get_transition_prob(t))

        return states, noise_levels

    def get_forward_energy(self, x_prev: np.ndarray, x_curr: np.ndarray, t: int) -> float:
        """
        Compute forward process energy E^f as in paper Eq. (C1).

        E^f(x^{t-1}, x^t) encourages consistency between adjacent time steps.

        Args:
            x_prev: State at t-1
            x_curr: State at t
            t: Current time step

        Returns:
            Forward energy value
        """
        # Simple consistency energy: penalize differences
        diff = np.sum(x_prev != x_curr)

        # Scale by noise level
        flip_prob = self.get_transition_prob(t)

        # Energy increases with more unexpected flips
        expected_flips = flip_prob * len(x_prev)
        energy = (diff - expected_flips) ** 2

        return energy
