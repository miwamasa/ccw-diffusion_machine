"""
RNG (Random Number Generator) simulator.

Implements sigmoid-biased Bernoulli sampling as described in paper Eq. (D6)
"""

import numpy as np


class RNGSimulator:
    """
    Hardware RNG simulator with sigmoid bias.

    Implements paper Eq. (D6):
        p = σ(bias_voltage / V_s - φ)

    Args:
        V_s: Scale voltage (default: 1.0)
        phi: Offset parameter (default: 0.5)
        sampling_rate: Sampling rate in Hz (default: 10 MHz)
        energy_per_bit: Energy consumption in Joules per bit (default: 350 aJ)
    """

    def __init__(self,
                 V_s: float = 1.0,
                 phi: float = 0.5,
                 sampling_rate: float = 10e6,
                 energy_per_bit: float = 350e-18):
        self.V_s = V_s
        self.phi = phi
        self.sampling_rate = sampling_rate
        self.energy_per_bit = energy_per_bit

        self.total_samples = 0
        self.total_energy = 0.0

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function with numerical stability."""
        return 1.0 / (1.0 + np.exp(-x))

    def sample(self, bias_voltage: float,
               rng: np.random.Generator = None) -> int:
        """
        Sample a single bit with given bias voltage.

        Args:
            bias_voltage: Control voltage
            rng: Random number generator

        Returns:
            Sampled bit (0 or 1)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Compute probability
        p = self.sigmoid(bias_voltage / self.V_s - self.phi)

        # Sample
        bit = 1 if rng.random() < p else 0

        # Update statistics
        self.total_samples += 1
        self.total_energy += self.energy_per_bit

        return bit

    def sample_vector(self, bias_voltages: np.ndarray,
                     rng: np.random.Generator = None) -> np.ndarray:
        """
        Sample multiple bits in parallel.

        Args:
            bias_voltages: Array of bias voltages
            rng: Random number generator

        Returns:
            Array of sampled bits (0 or 1)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Compute probabilities
        p = self.sigmoid(bias_voltages / self.V_s - self.phi)

        # Sample
        bits = (rng.random(len(bias_voltages)) < p).astype(int)

        # Update statistics
        self.total_samples += len(bits)
        self.total_energy += self.energy_per_bit * len(bits)

        return bits

    def get_energy_consumption(self) -> float:
        """
        Get total energy consumption.

        Returns:
            Total energy in Joules
        """
        return self.total_energy

    def get_average_energy_per_sample(self) -> float:
        """
        Get average energy per sample.

        Returns:
            Average energy in Joules
        """
        if self.total_samples == 0:
            return 0.0
        return self.total_energy / self.total_samples

    def reset_statistics(self):
        """Reset energy and sample counters."""
        self.total_samples = 0
        self.total_energy = 0.0
