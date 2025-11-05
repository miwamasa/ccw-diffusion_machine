"""
Energy consumption model for DTM hardware.

Implements energy model from paper Eq. (D12-D17)
"""

import numpy as np
from typing import Dict


class EnergyModel:
    """
    Hardware energy consumption model.

    Implements paper Eq. (D12):
        E_total = T * (E_samp + E_init + E_read)

    where E_samp = K * N * (E_rng + E_bias + E_clock + E_nb)

    Args:
        E_rng: RNG energy per bit (default: 350 aJ)
        E_bias: Bias circuit energy per bit (default: 100 aJ)
        E_clock: Clock distribution energy per bit (default: 50 aJ)
        E_nb: Neighbor communication energy per bit (default: 200 aJ)
        E_init: Initialization energy per layer (default: 1 pJ)
        E_read: Readout energy per layer (default: 1 pJ)
    """

    def __init__(self,
                 E_rng: float = 350e-18,
                 E_bias: float = 100e-18,
                 E_clock: float = 50e-18,
                 E_nb: float = 200e-18,
                 E_init: float = 1e-12,
                 E_read: float = 1e-12):
        self.E_rng = E_rng
        self.E_bias = E_bias
        self.E_clock = E_clock
        self.E_nb = E_nb
        self.E_init = E_init
        self.E_read = E_read

    def compute_sampling_energy(self, K: int, N: int) -> float:
        """
        Compute sampling energy for one layer.

        E_samp = K * N * (E_rng + E_bias + E_clock + E_nb)

        Args:
            K: Number of Gibbs iterations
            N: Number of variables

        Returns:
            Sampling energy in Joules
        """
        E_per_var = self.E_rng + self.E_bias + self.E_clock + self.E_nb
        return K * N * E_per_var

    def compute_layer_energy(self, K: int, N: int) -> float:
        """
        Compute total energy for one layer.

        Args:
            K: Number of Gibbs iterations
            N: Number of variables

        Returns:
            Layer energy in Joules
        """
        E_samp = self.compute_sampling_energy(K, N)
        return E_samp + self.E_init + self.E_read

    def compute_total_energy(self, T: int, K: int, N: int) -> float:
        """
        Compute total energy for all layers.

        Args:
            T: Number of layers
            K: Number of Gibbs iterations per layer
            N: Number of variables

        Returns:
            Total energy in Joules
        """
        E_layer = self.compute_layer_energy(K, N)
        return T * E_layer

    def compute_energy_breakdown(self, T: int, K: int, N: int) -> Dict[str, float]:
        """
        Compute detailed energy breakdown.

        Args:
            T: Number of layers
            K: Number of Gibbs iterations per layer
            N: Number of variables

        Returns:
            Dictionary with energy breakdown
        """
        E_samp = self.compute_sampling_energy(K, N)

        breakdown = {
            "rng": T * K * N * self.E_rng,
            "bias": T * K * N * self.E_bias,
            "clock": T * K * N * self.E_clock,
            "neighbor": T * K * N * self.E_nb,
            "init": T * self.E_init,
            "read": T * self.E_read,
            "sampling": T * E_samp,
            "total": self.compute_total_energy(T, K, N)
        }

        return breakdown

    def compare_with_gpu(self, dtm_energy: float,
                        problem_size: int,
                        gpu_energy_per_op: float = 100e-12) -> float:
        """
        Compare DTM energy efficiency with GPU.

        Args:
            dtm_energy: DTM energy in Joules
            problem_size: Problem size (number of variables)
            gpu_energy_per_op: GPU energy per operation (default: 100 pJ)

        Returns:
            Efficiency ratio (GPU energy / DTM energy)
        """
        # Estimate GPU energy (rough approximation)
        # Assume GPU needs similar number of operations
        gpu_energy = problem_size * gpu_energy_per_op

        return gpu_energy / dtm_energy

    def energy_per_constraint(self, total_energy: float,
                             num_constraints: int) -> float:
        """
        Compute energy per constraint.

        Args:
            total_energy: Total energy in Joules
            num_constraints: Number of constraints in problem

        Returns:
            Energy per constraint in Joules
        """
        if num_constraints == 0:
            return 0.0
        return total_energy / num_constraints

    def format_energy(self, energy: float) -> str:
        """
        Format energy value with appropriate units.

        Args:
            energy: Energy in Joules

        Returns:
            Formatted string with units
        """
        if energy >= 1:
            return f"{energy:.2f} J"
        elif energy >= 1e-3:
            return f"{energy*1e3:.2f} mJ"
        elif energy >= 1e-6:
            return f"{energy*1e6:.2f} μJ"
        elif energy >= 1e-9:
            return f"{energy*1e9:.2f} nJ"
        elif energy >= 1e-12:
            return f"{energy*1e12:.2f} pJ"
        elif energy >= 1e-15:
            return f"{energy*1e15:.2f} fJ"
        else:
            return f"{energy*1e18:.2f} aJ"
