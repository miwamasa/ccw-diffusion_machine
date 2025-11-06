"""
Training module for Boltzmann Machine using Contrastive Divergence.

Implements the Contrastive Divergence (CD) algorithm for learning
the parameters (J, h) of energy-based models.
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Optional
from dtm_simulator.core.boltzmann_machine import BoltzmannMachine


class BoltzmannMachineTrainer:
    """
    Trainer for Boltzmann Machine using Contrastive Divergence.

    The CD algorithm learns the coupling matrix J and bias vector h
    from training data by minimizing the negative log-likelihood.

    Args:
        bm: BoltzmannMachine instance to train
        learning_rate: Learning rate for parameter updates
        cd_steps: Number of Gibbs steps for negative phase (CD-k)
        l2_reg: L2 regularization strength
    """

    def __init__(
        self,
        bm: BoltzmannMachine,
        learning_rate: float = 0.01,
        cd_steps: int = 1,
        l2_reg: float = 0.0001,
    ):
        self.bm = bm
        self.learning_rate = learning_rate
        self.cd_steps = cd_steps
        self.l2_reg = l2_reg

        # Track training statistics
        self.training_history = {
            "reconstruction_error": [],
            "log_likelihood": [],
        }

    def compute_statistics(self, x: np.ndarray) -> Tuple[np.ndarray, sp.csr_matrix]:
        """
        Compute sufficient statistics (correlations) for a state.

        Args:
            x: Binary state vector (N,)

        Returns:
            Tuple of (bias_stats, coupling_stats)
            - bias_stats: E[xi] = xi
            - coupling_stats: E[xi xj] = xi * xj (outer product)
        """
        # Bias statistics: E[xi]
        bias_stats = x.copy()

        # Coupling statistics: E[xi xj]
        # We only compute for existing connections in J
        rows, cols = self.bm.J.nonzero()
        data = []
        for i, j in zip(rows, cols):
            if i < j:  # Only upper triangle to avoid double counting
                data.append(x[i] * x[j])

        coupling_stats = sp.csr_matrix(
            (data, (rows[:len(data)], cols[:len(data)])),
            shape=self.bm.J.shape
        )

        return bias_stats, coupling_stats

    def positive_phase(self, data_batch: List[np.ndarray]) -> Tuple[np.ndarray, sp.csr_matrix]:
        """
        Compute positive phase statistics from data.

        Args:
            data_batch: List of data samples

        Returns:
            Average statistics over the batch
        """
        n_samples = len(data_batch)

        # Accumulate statistics
        bias_sum = np.zeros(self.bm.N)
        coupling_sum_data = []

        for x in data_batch:
            bias_stats, coupling_stats = self.compute_statistics(x)
            bias_sum += bias_stats
            coupling_sum_data.append(coupling_stats)

        # Average
        bias_avg = bias_sum / n_samples

        # Average coupling statistics
        coupling_avg = sum(coupling_sum_data) / n_samples

        return bias_avg, coupling_avg

    def negative_phase(self, data_batch: List[np.ndarray]) -> Tuple[np.ndarray, sp.csr_matrix]:
        """
        Compute negative phase statistics using CD-k.

        Args:
            data_batch: List of data samples (used as starting points)

        Returns:
            Average statistics over model samples
        """
        n_samples = len(data_batch)

        # Accumulate statistics
        bias_sum = np.zeros(self.bm.N)
        coupling_sum_model = []

        for x_data in data_batch:
            # Run k steps of Gibbs sampling starting from data
            x_model = x_data.copy()
            for _ in range(self.cd_steps):
                x_model = self.bm.gibbs_step(x_model, color=0)
                x_model = self.bm.gibbs_step(x_model, color=1)

            # Compute statistics from model sample
            bias_stats, coupling_stats = self.compute_statistics(x_model)
            bias_sum += bias_stats
            coupling_sum_model.append(coupling_stats)

        # Average
        bias_avg = bias_sum / n_samples
        coupling_avg = sum(coupling_sum_model) / n_samples

        return bias_avg, coupling_avg

    def train_batch(self, data_batch: List[np.ndarray]) -> float:
        """
        Train on a single batch using Contrastive Divergence.

        Args:
            data_batch: List of training samples

        Returns:
            Reconstruction error for this batch
        """
        # Positive phase (data statistics)
        bias_pos, coupling_pos = self.positive_phase(data_batch)

        # Negative phase (model statistics)
        bias_neg, coupling_neg = self.negative_phase(data_batch)

        # Compute gradients
        grad_h = bias_pos - bias_neg
        grad_J = coupling_pos - coupling_neg

        # Update parameters with learning rate and regularization
        self.bm.h += self.learning_rate * (grad_h - self.l2_reg * self.bm.h)

        # Update J (only for existing connections)
        rows, cols = self.bm.J.nonzero()
        for idx, (i, j) in enumerate(zip(rows, cols)):
            if i < j:  # Only upper triangle
                gradient = grad_J[i, j] if grad_J[i, j] != 0 else 0
                current_value = self.bm.J[i, j]

                # Update with regularization
                new_value = current_value + self.learning_rate * (
                    gradient - self.l2_reg * current_value
                )

                # Symmetric update
                self.bm.J[i, j] = new_value
                self.bm.J[j, i] = new_value

        # Compute reconstruction error
        recon_error = 0.0
        for x_data in data_batch:
            x_recon, _ = self.bm.sample(x_data, num_steps=self.cd_steps)
            recon_error += np.mean(x_data != x_recon)

        return recon_error / len(data_batch)

    def train(
        self,
        training_data: List[np.ndarray],
        num_epochs: int = 10,
        batch_size: int = 10,
        verbose: bool = True,
    ) -> dict:
        """
        Train the Boltzmann Machine on training data.

        Args:
            training_data: List of training samples
            num_epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        n_samples = len(training_data)
        n_batches = (n_samples + batch_size - 1) // batch_size

        if verbose:
            print(f"Training Boltzmann Machine:")
            print(f"  Samples: {n_samples}")
            print(f"  Epochs: {num_epochs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Learning rate: {self.learning_rate}")
            print(f"  CD steps: {self.cd_steps}")
            print()

        for epoch in range(num_epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            shuffled_data = [training_data[i] for i in indices]

            epoch_recon_error = 0.0

            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch = shuffled_data[start_idx:end_idx]

                # Train on batch
                recon_error = self.train_batch(batch)
                epoch_recon_error += recon_error

            # Average reconstruction error
            avg_recon_error = epoch_recon_error / n_batches

            # Store statistics
            self.training_history["reconstruction_error"].append(avg_recon_error)

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Reconstruction Error = {avg_recon_error:.4f}")

        if verbose:
            print("\nTraining complete!")

        return self.training_history

    def evaluate(self, test_data: List[np.ndarray]) -> float:
        """
        Evaluate reconstruction error on test data.

        Args:
            test_data: List of test samples

        Returns:
            Average reconstruction error
        """
        total_error = 0.0
        for x in test_data:
            x_recon, _ = self.bm.sample(x, num_steps=self.cd_steps)
            total_error += np.mean(x != x_recon)

        return total_error / len(test_data)


def generate_training_data(
    pattern: np.ndarray,
    num_samples: int = 100,
    noise_level: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Generate training data by adding noise to a pattern.

    Args:
        pattern: Clean pattern to use as template
        num_samples: Number of samples to generate
        noise_level: Probability of flipping each bit
        rng: Random number generator

    Returns:
        List of noisy samples
    """
    if rng is None:
        rng = np.random.default_rng()

    training_data = []
    for _ in range(num_samples):
        sample = pattern.copy()

        # Add noise by flipping bits
        flip_mask = rng.random(len(sample)) < noise_level
        sample[flip_mask] = -sample[flip_mask]

        training_data.append(sample)

    return training_data


def generate_multi_pattern_data(
    patterns: List[np.ndarray],
    samples_per_pattern: int = 50,
    noise_level: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Generate training data from multiple patterns.

    Args:
        patterns: List of clean patterns
        samples_per_pattern: Number of samples per pattern
        noise_level: Noise level for each sample
        rng: Random number generator

    Returns:
        List of training samples
    """
    if rng is None:
        rng = np.random.default_rng()

    all_data = []
    for pattern in patterns:
        pattern_data = generate_training_data(
            pattern, samples_per_pattern, noise_level, rng
        )
        all_data.extend(pattern_data)

    # Shuffle
    indices = rng.permutation(len(all_data))
    shuffled_data = [all_data[i] for i in indices]

    return shuffled_data
