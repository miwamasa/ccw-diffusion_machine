"""
Bias circuit simulator.

Implements bias voltage generation for controlling RNG sampling.
"""

import numpy as np


class BiasCircuit:
    """
    Bias circuit for controlling sampling probabilities.

    Converts desired probabilities to bias voltages for RNG.

    Args:
        V_s: Scale voltage (should match RNG)
        phi: Offset parameter (should match RNG)
    """

    def __init__(self, V_s: float = 1.0, phi: float = 0.5):
        self.V_s = V_s
        self.phi = phi

    def prob_to_voltage(self, p: float) -> float:
        """
        Convert probability to bias voltage.

        Inverts sigmoid: V = V_s * (log(p/(1-p)) + φ)

        Args:
            p: Desired probability (0 to 1)

        Returns:
            Bias voltage
        """
        # Clip to avoid numerical issues
        p = np.clip(p, 1e-10, 1 - 1e-10)

        # Inverse sigmoid
        logit = np.log(p / (1 - p))
        voltage = self.V_s * (logit + self.phi)

        return voltage

    def voltage_to_prob(self, V: float) -> float:
        """
        Convert bias voltage to probability.

        Args:
            V: Bias voltage

        Returns:
            Probability (0 to 1)
        """
        p = 1.0 / (1.0 + np.exp(-(V / self.V_s - self.phi)))
        return p

    def compute_voltages(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Convert array of probabilities to voltages.

        Args:
            probabilities: Array of probabilities

        Returns:
            Array of bias voltages
        """
        return np.array([self.prob_to_voltage(p) for p in probabilities])
