"""
Gibbs Sampler utilities and diagnostics.

Provides helper functions for monitoring mixing time and convergence.
"""

import numpy as np
from typing import List, Tuple


def compute_autocorrelation(samples: List[np.ndarray], max_lag: int = 50) -> np.ndarray:
    """
    Compute autocorrelation function of samples.

    Args:
        samples: List of state vectors
        max_lag: Maximum lag to compute

    Returns:
        Autocorrelation values for each lag
    """
    n_samples = len(samples)
    if n_samples < max_lag:
        max_lag = n_samples - 1

    # Convert to array
    samples_array = np.array(samples)  # (n_samples, N)

    # Mean and variance
    mean = np.mean(samples_array, axis=0)
    var = np.var(samples_array, axis=0)

    autocorr = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            # Compute autocorrelation at this lag
            cov = np.mean(
                (samples_array[:-lag] - mean) * (samples_array[lag:] - mean),
                axis=0
            )
            autocorr[lag] = np.mean(cov / (var + 1e-10))

    return autocorr


def estimate_mixing_time(autocorr: np.ndarray, threshold: float = 0.1) -> int:
    """
    Estimate mixing time from autocorrelation.

    Mixing time is the first lag where autocorrelation drops below threshold.

    Args:
        autocorr: Autocorrelation values
        threshold: Threshold value (default: 0.1)

    Returns:
        Estimated mixing time (number of steps)
    """
    for i, ac in enumerate(autocorr):
        if abs(ac) < threshold:
            return i

    return len(autocorr)


def effective_sample_size(n_samples: int, autocorr: np.ndarray) -> float:
    """
    Compute effective sample size considering autocorrelation.

    ESS = N / (1 + 2 * Σ ρ(k))

    Args:
        n_samples: Number of samples
        autocorr: Autocorrelation values

    Returns:
        Effective sample size
    """
    # Sum autocorrelations (excluding lag 0)
    autocorr_sum = np.sum(autocorr[1:])
    ess = n_samples / (1.0 + 2.0 * autocorr_sum)
    return max(1.0, ess)


def check_convergence(energies: List[float], window_size: int = 100,
                      tolerance: float = 0.01) -> bool:
    """
    Check if energy has converged.

    Args:
        energies: List of energy values over iterations
        window_size: Size of sliding window
        tolerance: Relative tolerance for convergence

    Returns:
        True if converged
    """
    if len(energies) < window_size * 2:
        return False

    # Compare recent window with previous window
    recent = np.array(energies[-window_size:])
    previous = np.array(energies[-2*window_size:-window_size])

    mean_recent = np.mean(recent)
    mean_previous = np.mean(previous)

    if abs(mean_previous) < 1e-10:
        return abs(mean_recent - mean_previous) < tolerance
    else:
        rel_change = abs(mean_recent - mean_previous) / abs(mean_previous)
        return rel_change < tolerance


def sample_with_diagnostics(bm, x_init: np.ndarray, num_steps: int,
                             diagnostic_interval: int = 10) -> Tuple[np.ndarray, dict]:
    """
    Sample with diagnostic information.

    Args:
        bm: BoltzmannMachine instance
        x_init: Initial state
        num_steps: Number of sampling steps
        diagnostic_interval: Interval for recording diagnostics

    Returns:
        Final state and dictionary of diagnostics
    """
    x = x_init.copy()
    energies = []
    states = []

    for step in range(num_steps):
        x = bm.gibbs_step(x, color=0)
        x = bm.gibbs_step(x, color=1)

        if step % diagnostic_interval == 0:
            energies.append(bm.energy(x))
            states.append(x.copy())

    # Compute diagnostics
    autocorr = compute_autocorrelation(states) if len(states) > 1 else np.array([1.0])
    mixing_time = estimate_mixing_time(autocorr)
    ess = effective_sample_size(len(states), autocorr)
    converged = check_convergence(energies)

    diagnostics = {
        "energies": energies,
        "autocorrelation": autocorr,
        "mixing_time": mixing_time,
        "effective_sample_size": ess,
        "converged": converged,
        "final_energy": energies[-1] if energies else None,
    }

    return x, diagnostics
