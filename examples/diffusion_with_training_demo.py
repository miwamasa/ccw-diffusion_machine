#!/usr/bin/env python
"""
Diffusion Process with EBM Training Demo

This demo compares diffusion model performance:
1. WITHOUT training: EBM with random/biased parameters
2. WITH training: EBM trained on data using Contrastive Divergence

Goal: Demonstrate that EBM training significantly improves denoising quality.
"""

import sys
import time
import numpy as np
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dtm_simulator.core.boltzmann_machine import BoltzmannMachine
from dtm_simulator.core.forward_process import ForwardProcess
from dtm_simulator.core.reverse_process import ReverseProcess
from dtm_simulator.training.ebm_trainer import (
    BoltzmannMachineTrainer,
    generate_training_data,
    generate_multi_pattern_data,
)


def create_pattern_checkerboard(L: int) -> np.ndarray:
    """Create a checkerboard pattern."""
    pattern = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            pattern[i, j] = 1 if (i + j) % 2 == 0 else -1
    return pattern.flatten()


def create_pattern_stripes_horizontal(L: int) -> np.ndarray:
    """Create horizontal stripes pattern."""
    pattern = np.zeros((L, L))
    for i in range(L):
        pattern[i, :] = 1 if i % 2 == 0 else -1
    return pattern.flatten()


def create_pattern_stripes_vertical(L: int) -> np.ndarray:
    """Create vertical stripes pattern."""
    pattern = np.zeros((L, L))
    for j in range(L):
        pattern[:, j] = 1 if j % 2 == 0 else -1
    return pattern.flatten()


def visualize_state(x: np.ndarray, L: int, title: str = ""):
    """Visualize a binary state as ASCII art."""
    if title:
        print(f"\n{title}:")

    grid = x.reshape((L, L))
    for i in range(L):
        row = ""
        for j in range(L):
            row += "██" if grid[i, j] > 0 else "  "
        print(row)


def compute_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute similarity between two binary states."""
    matches = np.sum(x1 == x2)
    return matches / len(x1)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_without_training():
    """Demo: Denoising WITHOUT EBM training (baseline)."""
    print_section("Baseline: Denoising WITHOUT EBM Training")

    L = 8
    T = 4  # Fewer layers for faster demo

    print(f"\nSetup:")
    print(f"  Grid size: {L}×{L}")
    print(f"  Diffusion layers: {T}")
    print(f"  EBM: Untrained (random parameters + weak bias)")
    print()

    # Target pattern
    x_target = create_pattern_stripes_horizontal(L)
    visualize_state(x_target, L, "Target Pattern (Horizontal Stripes)")

    # Add noise
    forward = ForwardProcess(num_layers=T, gamma=1.0, M=2)
    rng = np.random.default_rng(42)
    x_noisy = forward.add_noise(x_target, t=T, rng=rng)
    visualize_state(x_noisy, L, f"Noisy Input (t={T})")
    initial_similarity = compute_similarity(x_noisy, x_target)
    print(f"Initial similarity: {initial_similarity:.1%}")

    # Create UNTRAINED EBM layers
    print("\nCreating UNTRAINED EBM layers...")
    ebm_layers = []
    for t in range(T):
        ebm = BoltzmannMachine(L=L, connectivity="G12", beta=1.0, seed=42+t)
        # Only weak bias, no training
        ebm.set_bias(x_target * 0.5)
        ebm_layers.append(ebm)

    # Denoise
    print("Denoising with untrained EBM...")
    reverse = ReverseProcess(ebm_layers, forward, mixing_steps=50)
    start_time = time.time()
    x_denoised = reverse.sample_clean(x_noisy, rng)
    elapsed = time.time() - start_time

    visualize_state(x_denoised, L, "Denoised Output (UNTRAINED)")
    final_similarity = compute_similarity(x_denoised, x_target)

    print(f"\nResults (UNTRAINED):")
    print(f"  Initial similarity: {initial_similarity:.1%}")
    print(f"  Final similarity: {final_similarity:.1%}")
    print(f"  Improvement: {(final_similarity - initial_similarity)*100:+.1f} percentage points")
    print(f"  Time: {elapsed:.2f}s")

    return {
        "initial": initial_similarity,
        "final": final_similarity,
        "improvement": final_similarity - initial_similarity,
        "time": elapsed,
    }


def demo_with_training():
    """Demo: Denoising WITH EBM training."""
    print_section("Improved: Denoising WITH EBM Training")

    L = 8
    T = 4

    print(f"\nSetup:")
    print(f"  Grid size: {L}×{L}")
    print(f"  Diffusion layers: {T}")
    print(f"  EBM: TRAINED on pattern data")
    print()

    # Target pattern
    x_target = create_pattern_stripes_horizontal(L)
    visualize_state(x_target, L, "Target Pattern (Horizontal Stripes)")

    # Generate training data
    print("\nGenerating training data...")
    print("  Creating noisy samples of target pattern...")
    rng = np.random.default_rng(123)
    training_data = generate_training_data(
        x_target,
        num_samples=200,
        noise_level=0.15,
        rng=rng
    )
    print(f"  Generated {len(training_data)} training samples")

    # Add noise to test sample
    forward = ForwardProcess(num_layers=T, gamma=1.0, M=2)
    test_rng = np.random.default_rng(42)  # Same seed as baseline
    x_noisy = forward.add_noise(x_target, t=T, rng=test_rng)
    visualize_state(x_noisy, L, f"Noisy Input (t={T})")
    initial_similarity = compute_similarity(x_noisy, x_target)
    print(f"Initial similarity: {initial_similarity:.1%}")

    # Create and TRAIN EBM layers
    print("\nTraining EBM layers...")
    ebm_layers = []
    for t in range(T):
        print(f"\n--- Layer {t+1}/{T} ---")
        ebm = BoltzmannMachine(L=L, connectivity="G12", beta=1.0, seed=100+t)

        # Train this layer
        trainer = BoltzmannMachineTrainer(
            ebm,
            learning_rate=0.05,
            cd_steps=1,
            l2_reg=0.001
        )

        trainer.train(
            training_data,
            num_epochs=20,
            batch_size=20,
            verbose=True
        )

        ebm_layers.append(ebm)

    # Denoise with TRAINED EBM
    print("\nDenoising with TRAINED EBM...")
    reverse = ReverseProcess(ebm_layers, forward, mixing_steps=50)
    start_time = time.time()
    x_denoised = reverse.sample_clean(x_noisy, test_rng)
    elapsed = time.time() - start_time

    visualize_state(x_denoised, L, "Denoised Output (TRAINED)")
    final_similarity = compute_similarity(x_denoised, x_target)

    print(f"\nResults (TRAINED):")
    print(f"  Initial similarity: {initial_similarity:.1%}")
    print(f"  Final similarity: {final_similarity:.1%}")
    print(f"  Improvement: {(final_similarity - initial_similarity)*100:+.1f} percentage points")
    print(f"  Time: {elapsed:.2f}s (excluding training)")

    return {
        "initial": initial_similarity,
        "final": final_similarity,
        "improvement": final_similarity - initial_similarity,
        "time": elapsed,
    }


def demo_multi_pattern_training():
    """Demo: Training on multiple patterns."""
    print_section("Advanced: Multi-Pattern Training")

    L = 8
    T = 4

    print(f"\nSetup:")
    print(f"  Grid size: {L}×{L}")
    print(f"  Diffusion layers: {T}")
    print(f"  Training: Multiple patterns (checkerboard + stripes)")
    print()

    # Multiple target patterns
    patterns = [
        create_pattern_checkerboard(L),
        create_pattern_stripes_horizontal(L),
        create_pattern_stripes_vertical(L),
    ]

    print("Target Patterns:")
    visualize_state(patterns[0], L, "1. Checkerboard")
    visualize_state(patterns[1], L, "2. Horizontal Stripes")
    visualize_state(patterns[2], L, "3. Vertical Stripes")

    # Generate multi-pattern training data
    print("\nGenerating multi-pattern training data...")
    rng = np.random.default_rng(456)
    training_data = generate_multi_pattern_data(
        patterns,
        samples_per_pattern=100,
        noise_level=0.15,
        rng=rng
    )
    print(f"  Generated {len(training_data)} training samples")

    # Test on horizontal stripes
    x_target = patterns[1]  # Horizontal stripes
    forward = ForwardProcess(num_layers=T, gamma=1.0, M=2)
    test_rng = np.random.default_rng(42)
    x_noisy = forward.add_noise(x_target, t=T, rng=test_rng)

    visualize_state(x_noisy, L, f"Test Input: Noisy Horizontal Stripes (t={T})")
    initial_similarity = compute_similarity(x_noisy, x_target)
    print(f"Initial similarity: {initial_similarity:.1%}")

    # Train EBM on multi-pattern data
    print("\nTraining EBM on multi-pattern data...")
    ebm_layers = []
    for t in range(T):
        print(f"\n--- Layer {t+1}/{T} ---")
        ebm = BoltzmannMachine(L=L, connectivity="G12", beta=1.0, seed=200+t)

        trainer = BoltzmannMachineTrainer(
            ebm,
            learning_rate=0.05,
            cd_steps=1,
            l2_reg=0.001
        )

        trainer.train(
            training_data,
            num_epochs=15,
            batch_size=30,
            verbose=True
        )

        ebm_layers.append(ebm)

    # Denoise
    print("\nDenoising with multi-pattern trained EBM...")
    reverse = ReverseProcess(ebm_layers, forward, mixing_steps=50)
    start_time = time.time()
    x_denoised = reverse.sample_clean(x_noisy, test_rng)
    elapsed = time.time() - start_time

    visualize_state(x_denoised, L, "Denoised Output (Multi-Pattern Trained)")
    final_similarity = compute_similarity(x_denoised, x_target)

    print(f"\nResults (Multi-Pattern Trained):")
    print(f"  Initial similarity: {initial_similarity:.1%}")
    print(f"  Final similarity: {final_similarity:.1%}")
    print(f"  Improvement: {(final_similarity - initial_similarity)*100:+.1f} percentage points")
    print(f"  Time: {elapsed:.2f}s")

    return {
        "initial": initial_similarity,
        "final": final_similarity,
        "improvement": final_similarity - initial_similarity,
        "time": elapsed,
    }


def main():
    """Run all demos and compare results."""
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  DIFFUSION WITH EBM TRAINING - DTM SIMULATOR".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Comparing Performance: Untrained vs Trained EBM".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    print("This demo demonstrates the impact of EBM training on denoising quality.")
    print()
    print("Experiments:")
    print("  1. Baseline: Untrained EBM (random parameters + weak bias)")
    print("  2. Single-pattern training: EBM trained on target pattern")
    print("  3. Multi-pattern training: EBM trained on multiple patterns")
    print()

    # Run demos
    results_untrained = demo_without_training()
    results_trained = demo_with_training()
    results_multi = demo_multi_pattern_training()

    # Comparison
    print_section("Final Comparison")

    print("\nPerformance Summary:")
    print("-" * 70)
    print(f"{'Method':<30s} {'Initial':<10s} {'Final':<10s} {'Improvement':<12s} {'Time':<8s}")
    print("-" * 70)

    methods = [
        ("Untrained EBM", results_untrained),
        ("Single-Pattern Trained", results_trained),
        ("Multi-Pattern Trained", results_multi),
    ]

    for name, results in methods:
        print(f"{name:<30s} "
              f"{results['initial']*100:>6.1f}%   "
              f"{results['final']*100:>6.1f}%   "
              f"{results['improvement']*100:>+7.1f}pp     "
              f"{results['time']:>5.2f}s")

    print("-" * 70)
    print()

    # Analysis
    improvement_trained = results_trained['final'] - results_untrained['final']
    improvement_multi = results_multi['final'] - results_untrained['final']

    print("Key Findings:")
    if improvement_trained > 0.05:
        print(f"  ✓ Training SIGNIFICANTLY improves denoising (+{improvement_trained*100:.1f}pp)")
    elif improvement_trained > 0:
        print(f"  ○ Training moderately improves denoising (+{improvement_trained*100:.1f}pp)")
    else:
        print(f"  × Training did not improve denoising ({improvement_trained*100:+.1f}pp)")

    if results_trained['final'] > 0.8:
        print(f"  ✓ Trained model achieves excellent denoising ({results_trained['final']*100:.1f}%)")
    elif results_trained['final'] > 0.6:
        print(f"  ○ Trained model achieves good denoising ({results_trained['final']*100:.1f}%)")
    else:
        print(f"  △ Trained model has limited denoising ({results_trained['final']*100:.1f}%)")

    print()
    print("Conclusion:")
    print("  EBM training using Contrastive Divergence enables the model to")
    print("  learn the structure of patterns, leading to improved denoising.")
    print()
    print("  The trained EBM captures correlations in the data and uses them")
    print("  to guide the reverse diffusion process toward realistic samples.")


if __name__ == "__main__":
    main()
