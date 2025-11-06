"""
Reverse process implementation for DTM.

Implements denoising process as described in paper Eq. (7-8)
"""

import numpy as np
from typing import Optional
from dtm_simulator.core.boltzmann_machine import BoltzmannMachine
from dtm_simulator.core.forward_process import ForwardProcess


class ReverseProcess:
    """
    Reverse denoising process for DTM.

    Implements conditional distribution from paper Eq. (7-8):
        P_θ(x^{t-1} | x^t) ∝ Σ_{z^{t-1}} exp(-E^f_{t-1}(x^{t-1}, x^t)
                                              - E^θ_{t-1}(x^{t-1}, z^{t-1}))

    Args:
        ebm_layers: List of Boltzmann machines for each layer
        forward_process: Forward process for computing E^f
        mixing_steps: Number of Gibbs steps per denoising step
    """

    def __init__(self,
                 ebm_layers: list,
                 forward_process: ForwardProcess,
                 mixing_steps: int = 250):
        self.ebm_layers = ebm_layers
        self.forward_process = forward_process
        self.K = mixing_steps
        self.T = len(ebm_layers)

    def denoise_step(self, x_t: np.ndarray, t: int,
                     rng: np.random.Generator = None) -> np.ndarray:
        """
        Perform one denoising step from x^t to x^{t-1}.

        Args:
            x_t: Noisy state at time t (N,)
            t: Current time step (1 to T)
            rng: Random number generator

        Returns:
            Denoised state at time t-1 (N,)
        """
        if rng is None:
            rng = np.random.default_rng()

        if t <= 0 or t > self.T:
            return x_t.copy()

        # Get EBM for this layer
        ebm = self.ebm_layers[t - 1]

        # Save original bias (preserve learned parameters)
        original_bias = ebm.h.copy()

        # Compute conditional bias based on x_t to incorporate forward energy
        # ADD this to the learned bias rather than REPLACING it
        conditional_bias = self._compute_conditional_bias(x_t, t)
        ebm.h = original_bias + conditional_bias

        # Sample from EBM to get x_{t-1}
        x_prev = ebm.sample(x_t, num_steps=self.K)[0]

        # Restore original bias for next use
        ebm.h = original_bias

        return x_prev

    def _compute_conditional_bias(self, x_t: np.ndarray, t: int) -> np.ndarray:
        """
        Compute bias term for conditioning on x_t.

        This incorporates the forward energy E^f.

        Args:
            x_t: State at time t
            t: Time step

        Returns:
            Bias vector for EBM
        """
        # Simple version: bias towards x_t
        # In full implementation, this would use learned parameters
        flip_prob = self.forward_process.get_transition_prob(t)

        # Bias encourages staying close to x_t
        bias = x_t * (1.0 - 2.0 * flip_prob)

        return bias

    def reverse_trajectory(self, x_T: np.ndarray,
                          rng: np.random.Generator = None,
                          record_trajectory: bool = True) -> list:
        """
        Generate full reverse trajectory from noisy to clean.

        Args:
            x_T: Noisy initial state (N,)
            rng: Random number generator
            record_trajectory: Whether to record all intermediate states

        Returns:
            List of states from x_T to x_0
        """
        if rng is None:
            rng = np.random.default_rng()

        states = [x_T.copy()] if record_trajectory else []
        x = x_T.copy()

        for t in range(self.T, 0, -1):
            x = self.denoise_step(x, t, rng)
            if record_trajectory:
                states.append(x.copy())

        if not record_trajectory:
            states = [x]

        return states

    def sample_clean(self, x_T: np.ndarray,
                    rng: np.random.Generator = None) -> np.ndarray:
        """
        Sample clean state from noisy input.

        Args:
            x_T: Noisy state (N,)
            rng: Random number generator

        Returns:
            Clean state estimate (N,)
        """
        trajectory = self.reverse_trajectory(x_T, rng, record_trajectory=False)
        return trajectory[-1]
