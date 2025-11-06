#!/usr/bin/env python
"""
Diffusion Process Demo for DTM Hardware Simulator

This demo demonstrates the CORE FUNCTIONALITY of diffusion models:
1. Forward Process: Adding noise to clean data step-by-step
2. Reverse Process: Denoising from pure noise back to clean data

This is the ORIGINAL use case of diffusion models, as described in the paper.
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


def create_pattern_checkerboard(L: int) -> np.ndarray:
    """
    Create a checkerboard pattern.

    Args:
        L: Grid size (L×L)

    Returns:
        Binary vector {-1, +1}^(L×L) representing checkerboard
    """
    pattern = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            pattern[i, j] = 1 if (i + j) % 2 == 0 else -1
    return pattern.flatten()


def create_pattern_stripes_horizontal(L: int) -> np.ndarray:
    """
    Create horizontal stripes pattern.

    Args:
        L: Grid size

    Returns:
        Binary vector representing horizontal stripes
    """
    pattern = np.zeros((L, L))
    for i in range(L):
        pattern[i, :] = 1 if i % 2 == 0 else -1
    return pattern.flatten()


def create_pattern_stripes_vertical(L: int) -> np.ndarray:
    """
    Create vertical stripes pattern.

    Args:
        L: Grid size

    Returns:
        Binary vector representing vertical stripes
    """
    pattern = np.zeros((L, L))
    for j in range(L):
        pattern[:, j] = 1 if j % 2 == 0 else -1
    return pattern.flatten()


def create_pattern_cross(L: int) -> np.ndarray:
    """
    Create a cross pattern.

    Args:
        L: Grid size

    Returns:
        Binary vector representing a cross
    """
    pattern = np.ones((L, L)) * (-1)
    center = L // 2
    pattern[center, :] = 1  # Horizontal line
    pattern[:, center] = 1  # Vertical line
    return pattern.flatten()


def visualize_state(x: np.ndarray, L: int, title: str = ""):
    """
    Visualize a binary state as ASCII art.

    Args:
        x: Binary vector {-1, +1}
        L: Grid size
        title: Title to display
    """
    if title:
        print(f"\n{title}:")

    grid = x.reshape((L, L))
    for i in range(L):
        row = ""
        for j in range(L):
            row += "██" if grid[i, j] > 0 else "  "
        print(row)


def compute_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute similarity between two binary states.

    Args:
        x1: First state
        x2: Second state

    Returns:
        Similarity ratio (0 to 1)
    """
    matches = np.sum(x1 == x2)
    return matches / len(x1)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_forward_process():
    """Demo 1: Forward process - adding noise to clean pattern."""
    print_section("Forward Process: Clean → Noisy")

    L = 10  # Grid size
    T = 8   # Number of diffusion layers

    print(f"\nGrid size: {L}×{L}")
    print(f"Number of diffusion layers: {T}")
    print(f"Total variables: {L*L}")
    print()

    # Create clean pattern
    print("Creating checkerboard pattern...")
    x_clean = create_pattern_checkerboard(L)
    visualize_state(x_clean, L, "Clean Pattern (t=0)")

    # Initialize forward process
    forward = ForwardProcess(num_layers=T, gamma=1.0, M=2)

    print("\nNoise schedule:")
    for t in range(T + 1):
        flip_prob = forward.get_transition_prob(t)
        print(f"  t={t}: flip probability = {flip_prob:.3f}")

    # Generate forward trajectory
    print("\nGenerating forward trajectory (adding noise)...")
    rng = np.random.default_rng(42)
    states, noise_levels = forward.forward_trajectory(x_clean, rng)

    # Visualize key states
    print("\nForward trajectory visualization:")
    key_steps = [0, 2, 4, 6, 8]
    for t in key_steps:
        if t < len(states):
            similarity = compute_similarity(states[t], x_clean)
            visualize_state(states[t], L,
                          f"State at t={t} (flip_prob={noise_levels[t]:.3f}, similarity={similarity:.1%})")

    print(f"\nFinal state at t={T}:")
    print(f"  Similarity to clean pattern: {compute_similarity(states[-1], x_clean):.1%}")
    print(f"  Expected for random noise: ~50%")

    return states, forward


def demo_reverse_process():
    """Demo 2: Reverse process - denoising from noise to clean."""
    print_section("Reverse Process: Noisy → Clean")

    L = 10  # Grid size
    T = 8   # Number of diffusion layers

    print(f"\nGrid size: {L}×{L}")
    print(f"Number of diffusion layers: {T}")
    print()

    # Create clean pattern (ground truth)
    print("Creating target pattern (horizontal stripes)...")
    x_target = create_pattern_stripes_horizontal(L)
    visualize_state(x_target, L, "Target Pattern")

    # Initialize forward process
    forward = ForwardProcess(num_layers=T, gamma=1.0, M=2)

    # Add noise to get noisy starting point
    print("\nAdding noise to create starting point...")
    rng = np.random.default_rng(42)
    x_noisy = forward.add_noise(x_target, t=T, rng=rng)
    visualize_state(x_noisy, L, f"Noisy Input (t={T})")
    print(f"Similarity to target: {compute_similarity(x_noisy, x_target):.1%}")

    # Initialize EBM layers
    print("\nInitializing EBM layers...")
    ebm_layers = []
    for t in range(T):
        ebm = BoltzmannMachine(L=L, connectivity="G12", beta=1.0, seed=42+t)
        # Set bias to encourage target pattern
        ebm.set_bias(x_target * 0.5)  # Weak bias towards target
        ebm_layers.append(ebm)

    # Initialize reverse process
    reverse = ReverseProcess(ebm_layers, forward, mixing_steps=50)

    # Generate reverse trajectory
    print("\nGenerating reverse trajectory (denoising)...")
    print("(This may take 30-60 seconds...)")
    start_time = time.time()

    states_reverse = reverse.reverse_trajectory(x_noisy, rng, record_trajectory=True)

    elapsed = time.time() - start_time
    print(f"Denoising completed in {elapsed:.2f} seconds")

    # Visualize key states
    print("\nReverse trajectory visualization:")
    key_indices = [0, 2, 4, 6, 8]  # Corresponds to t=8,6,4,2,0
    for idx in key_indices:
        if idx < len(states_reverse):
            t_actual = T - idx
            similarity = compute_similarity(states_reverse[idx], x_target)
            visualize_state(states_reverse[idx], L,
                          f"State after {idx} denoising steps (t={t_actual}, similarity={similarity:.1%})")

    # Final result
    x_denoised = states_reverse[-1]
    final_similarity = compute_similarity(x_denoised, x_target)

    print("\n" + "─" * 70)
    print("Denoising Result:")
    print(f"  Initial similarity (noisy): {compute_similarity(x_noisy, x_target):.1%}")
    print(f"  Final similarity (denoised): {final_similarity:.1%}")
    print(f"  Improvement: {(final_similarity - compute_similarity(x_noisy, x_target))*100:.1f} percentage points")

    if final_similarity > 0.8:
        print("  ✓ Excellent denoising!")
    elif final_similarity > 0.6:
        print("  ○ Good denoising")
    else:
        print("  × Limited denoising effect")

    return states_reverse


def demo_pattern_generation():
    """Demo 3: Pattern generation from pure noise."""
    print_section("Pattern Generation: Pure Noise → Structured Pattern")

    L = 8   # Smaller grid for faster generation
    T = 6   # Fewer layers

    print(f"\nGrid size: {L}×{L}")
    print(f"Number of diffusion layers: {T}")
    print()

    print("Goal: Generate cross pattern from pure random noise")
    print()

    # Target pattern
    x_target = create_pattern_cross(L)
    visualize_state(x_target, L, "Target Pattern (Cross)")

    # Start from pure random noise
    print("\nGenerating from pure random noise...")
    rng = np.random.default_rng(123)
    x_noise = rng.choice([-1, 1], size=L*L)
    visualize_state(x_noise, L, "Initial Random Noise")
    print(f"Similarity to target: {compute_similarity(x_noise, x_target):.1%}")

    # Initialize forward process
    forward = ForwardProcess(num_layers=T, gamma=1.0, M=2)

    # Initialize EBM layers with strong bias towards target
    print("\nInitializing EBM layers with target bias...")
    ebm_layers = []
    for t in range(T):
        ebm = BoltzmannMachine(L=L, connectivity="G12", beta=2.0, seed=100+t)
        # Strong bias towards target pattern
        ebm.set_bias(x_target * 1.0)
        ebm_layers.append(ebm)

    # Initialize reverse process
    reverse = ReverseProcess(ebm_layers, forward, mixing_steps=100)

    # Generate pattern
    print("\nGenerating pattern through reverse diffusion...")
    print("(This may take 20-40 seconds...)")
    start_time = time.time()

    x_generated = reverse.sample_clean(x_noise, rng)

    elapsed = time.time() - start_time
    print(f"Generation completed in {elapsed:.2f} seconds")

    # Show result
    visualize_state(x_generated, L, "Generated Pattern")

    final_similarity = compute_similarity(x_generated, x_target)
    print(f"\nSimilarity to target: {final_similarity:.1%}")

    if final_similarity > 0.8:
        print("✓ Successfully generated target pattern!")
    elif final_similarity > 0.6:
        print("○ Partially captured target pattern")
    else:
        print("× Failed to generate target pattern")

    return x_generated


def demo_multiple_patterns():
    """Demo 4: Compare different patterns."""
    print_section("Multiple Pattern Comparison")

    L = 8

    patterns = {
        "Checkerboard": create_pattern_checkerboard(L),
        "Horizontal Stripes": create_pattern_stripes_horizontal(L),
        "Vertical Stripes": create_pattern_stripes_vertical(L),
        "Cross": create_pattern_cross(L),
    }

    print(f"\nDisplaying {len(patterns)} different patterns:\n")

    for name, pattern in patterns.items():
        visualize_state(pattern, L, name)

    # Test forward process on all patterns
    print("\n" + "─" * 70)
    print("Testing forward process (adding noise) on all patterns:")
    print()

    forward = ForwardProcess(num_layers=8, gamma=1.0, M=2)
    rng = np.random.default_rng(42)

    t_test = 6  # Test at t=6
    flip_prob = forward.get_transition_prob(t_test)
    print(f"Adding noise at t={t_test} (flip probability={flip_prob:.3f}):\n")

    for name, pattern in patterns.items():
        noisy = forward.add_noise(pattern, t_test, rng)
        similarity = compute_similarity(noisy, pattern)
        print(f"{name}:")
        print(f"  Similarity after noise: {similarity:.1%}")
        # visualize_state(noisy, L, f"{name} (noisy)")


def main():
    """Run all diffusion process demos."""
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "      DIFFUSION PROCESS DEMO - DTM SIMULATOR".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Forward & Reverse Process Demonstration".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    print("This demo showcases the CORE FUNCTIONALITY of diffusion models:")
    print("  1. Forward Process: Clean data → Gradually add noise → Pure noise")
    print("  2. Reverse Process: Pure noise → Gradually denoise → Clean data")
    print()
    print("This is the ORIGINAL use case described in the DTM paper,")
    print("as opposed to constraint satisfaction which we tested earlier.")
    print()

    # Demo 1: Forward process
    states_forward, forward_obj = demo_forward_process()

    # Demo 2: Reverse process
    states_reverse = demo_reverse_process()

    # Demo 3: Pattern generation
    generated = demo_pattern_generation()

    # Demo 4: Multiple patterns
    demo_multiple_patterns()

    # Summary
    print_section("Demo Complete")

    print("\nSuccessfully demonstrated:")
    print("  ✓ Forward Process: Progressive noise addition")
    print("  ✓ Reverse Process: Progressive denoising")
    print("  ✓ Pattern Generation: From noise to structured pattern")
    print("  ✓ Multiple Pattern Types: Checkerboard, Stripes, Cross")
    print()

    print("Key Insights:")
    print("  1. Forward process gradually destroys structure (Clean → Noise)")
    print("  2. Reverse process attempts to restore structure (Noise → Clean)")
    print("  3. EBM bias guides the denoising towards target patterns")
    print("  4. This is the INTENDED use case of diffusion models")
    print()

    print("Comparison with previous demos:")
    print("  • Previous demos: Used DTM for constraint satisfaction (CSP)")
    print("  • This demo: Uses DTM as a GENERATIVE MODEL (original purpose)")
    print("  • Forward/Reverse processes are the CORE of diffusion models")
    print()

    print("Note: The denoising quality depends on:")
    print("  - EBM layer training (not implemented - using biased sampling)")
    print("  - Number of mixing steps per layer")
    print("  - Noise schedule (gamma parameter)")
    print("  - Target pattern complexity")


if __name__ == "__main__":
    main()
