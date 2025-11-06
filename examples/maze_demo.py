#!/usr/bin/env python
"""
Maze Path Finding Demo for DTM Hardware Simulator

This demo tests DTM's compatibility with path-finding problems,
which are expected to have MUCH BETTER performance than CSP problems
due to their smooth energy landscape.
"""

import sys
import time
import numpy as np
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dtm_simulator.core.dtm import DTM, DTMConfig
from dtm_simulator.problems.maze import MazeProblem
from dtm_simulator.hardware.energy_model import EnergyModel


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_simple_maze():
    """Demo 1: Simple maze with scattered obstacles."""
    print_section("Simple Maze (10×10 with scattered walls)")

    # Create problem
    problem = MazeProblem.create_simple_maze(size=10)

    print(f"Maze size: {problem.rows}×{problem.cols}")
    print(f"Start: {problem.start}")
    print(f"Goal: {problem.goal}")
    print(f"Number of walls: {np.sum(problem.maze == 1)}")
    print()

    # Show initial maze
    print("Initial maze:")
    for r in range(problem.rows):
        row_str = ""
        for c in range(problem.cols):
            if (r, c) == problem.start:
                row_str += " S "
            elif (r, c) == problem.goal:
                row_str += " G "
            elif problem.maze[r, c] == 1:
                row_str += " # "
            else:
                row_str += " . "
        print(row_str)
    print()

    # Solve with DTM
    print("Solving with DTM...")
    print("(This may take 15-30 seconds...)")
    print()

    config = DTMConfig(
        num_layers=2,
        grid_size=10,
        K_infer=100,
        beta=1.0
    )
    dtm = DTM(config)

    start_time = time.time()
    solution_x, info = dtm.solve(problem, max_steps=10000, use_annealing=True)
    solve_time = time.time() - start_time

    # Display solution
    path = problem.decode_solution(solution_x)
    print(problem.format_solution(path))
    print()

    # Evaluation
    satisfaction = problem.satisfaction_rate(solution_x)
    energy = problem.energy_function(solution_x)

    print(f"Constraint satisfaction: {satisfaction*100:.1f}%")
    print(f"Solving time: {solve_time:.2f} seconds")
    print(f"Best energy: {energy:.4f}")
    print()

    if satisfaction > 0.95:
        print("✓ Excellent solution!")
    elif satisfaction > 0.80:
        print("○ Good solution (minor issues)")
    else:
        print("× Solution needs improvement")

    return satisfaction, solve_time


def demo_corridor_maze():
    """Demo 2: Corridor maze with structured obstacles."""
    print_section("Corridor Maze (15×15 with structured walls)")

    # Create problem
    problem = MazeProblem.create_corridor_maze(rows=15, cols=15)

    print(f"Maze size: {problem.rows}×{problem.cols}")
    print(f"Start: {problem.start}")
    print(f"Goal: {problem.goal}")
    print(f"Number of walls: {np.sum(problem.maze == 1)}")
    print()

    # Show initial maze
    print("Initial maze:")
    for r in range(problem.rows):
        row_str = ""
        for c in range(problem.cols):
            if (r, c) == problem.start:
                row_str += "S"
            elif (r, c) == problem.goal:
                row_str += "G"
            elif problem.maze[r, c] == 1:
                row_str += "#"
            else:
                row_str += "."
        print(row_str)
    print()

    # Solve with DTM
    print("Solving with DTM...")
    print("(This may take 30-60 seconds...)")
    print()

    config = DTMConfig(
        num_layers=2,
        grid_size=15,
        K_infer=100,
        beta=1.0
    )
    dtm = DTM(config)

    start_time = time.time()
    solution_x, info = dtm.solve(problem, max_steps=20000, use_annealing=True)
    solve_time = time.time() - start_time

    # Display solution
    path = problem.decode_solution(solution_x)
    print("Solution:")
    for r in range(problem.rows):
        row_str = ""
        for c in range(problem.cols):
            if (r, c) == problem.start:
                row_str += "S"
            elif (r, c) == problem.goal:
                row_str += "G"
            elif problem.maze[r, c] == 1:
                row_str += "#"
            elif path[r, c] > 0.5:
                row_str += "*"
            else:
                row_str += "."
        print(row_str)
    print()

    # Evaluation
    satisfaction = problem.satisfaction_rate(solution_x)
    energy = problem.energy_function(solution_x)

    print(f"Constraint satisfaction: {satisfaction*100:.1f}%")
    print(f"Solving time: {solve_time:.2f} seconds")
    print(f"Best energy: {energy:.4f}")
    print()

    if satisfaction > 0.95:
        print("✓ Excellent solution!")
    elif satisfaction > 0.80:
        print("○ Good solution (minor issues)")
    else:
        print("× Solution needs improvement")

    return satisfaction, solve_time


def demo_spiral_maze():
    """Demo 3: Spiral maze pattern."""
    print_section("Spiral Maze (15×15 with spiral walls)")

    # Create problem
    problem = MazeProblem.create_spiral_maze(size=15)

    print(f"Maze size: {problem.rows}×{problem.cols}")
    print(f"Start: {problem.start}")
    print(f"Goal: {problem.goal}")
    print(f"Number of walls: {np.sum(problem.maze == 1)}")
    print()

    # Show initial maze
    print("Initial maze:")
    for r in range(problem.rows):
        row_str = ""
        for c in range(problem.cols):
            if (r, c) == problem.start:
                row_str += "S"
            elif (r, c) == problem.goal:
                row_str += "G"
            elif problem.maze[r, c] == 1:
                row_str += "#"
            else:
                row_str += "."
        print(row_str)
    print()

    # Solve with DTM
    print("Solving with DTM...")
    print("(This may take 30-60 seconds...)")
    print()

    config = DTMConfig(
        num_layers=2,
        grid_size=15,
        K_infer=100,
        beta=1.0
    )
    dtm = DTM(config)

    start_time = time.time()
    solution_x, info = dtm.solve(problem, max_steps=20000, use_annealing=True)
    solve_time = time.time() - start_time

    # Display solution
    path = problem.decode_solution(solution_x)
    print("Solution:")
    for r in range(problem.rows):
        row_str = ""
        for c in range(problem.cols):
            if (r, c) == problem.start:
                row_str += "S"
            elif (r, c) == problem.goal:
                row_str += "G"
            elif problem.maze[r, c] == 1:
                row_str += "#"
            elif path[r, c] > 0.5:
                row_str += "*"
            else:
                row_str += "."
        print(row_str)
    print()

    # Evaluation
    satisfaction = problem.satisfaction_rate(solution_x)
    energy = problem.energy_function(solution_x)

    print(f"Constraint satisfaction: {satisfaction*100:.1f}%")
    print(f"Solving time: {solve_time:.2f} seconds")
    print(f"Best energy: {energy:.4f}")
    print()

    if satisfaction > 0.95:
        print("✓ Excellent solution!")
    elif satisfaction > 0.80:
        print("○ Good solution (minor issues)")
    else:
        print("× Solution needs improvement")

    return satisfaction, solve_time


def comparison_analysis(results):
    """Analyze and compare results."""
    print_section("Performance Comparison Analysis")

    print("Maze Problem Results:")
    print("-" * 70)
    for name, satisfaction, time_taken in results:
        print(f"  {name:25s}: {satisfaction*100:5.1f}% satisfaction ({time_taken:.2f}s)")
    print()

    avg_satisfaction = np.mean([s for _, s, _ in results])
    print(f"Average satisfaction: {avg_satisfaction*100:.1f}%")
    print()

    # Compare with CSP problems
    print("Comparison with CSP Problems (from previous demos):")
    print("-" * 70)
    print("  N-Queen (8×8):              97.6% satisfaction")
    print("  Sudoku (9×9):               96.3% satisfaction")
    print("  Potts Model (cycle):        66.7% satisfaction")
    print("  Potts Model (grid):         66.7% satisfaction")
    print()

    print("Analysis:")
    if avg_satisfaction > 0.95:
        print("✓ Maze problems show EXCELLENT compatibility with DTM!")
        print("  → Smooth energy landscape enables effective optimization")
        print("  → Path-finding is well-suited for diffusion models")
    elif avg_satisfaction > 0.85:
        print("○ Maze problems show GOOD compatibility with DTM")
        print("  → Better than some CSP problems (Potts Model)")
    else:
        print("× Maze problems need improvement")
        print("  → May need better energy function tuning")

    print()
    print("Key insights:")
    print("  1. Path-finding problems have smooth energy landscapes")
    print("  2. Connectivity rewards create continuous optimization")
    print("  3. Intermediate states (partial paths) are meaningful")
    print("  4. Local constraints (walls) are easier than global constraints (Sudoku)")


def main():
    """Run all maze demos."""
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "         MAZE PATH-FINDING DEMO - DTM SIMULATOR".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Testing DTM Compatibility with Path-Finding Problems".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    print("Hypothesis: Maze problems should have BETTER performance than CSP")
    print("Reason: Smooth energy landscape and meaningful intermediate states")
    print()

    # Run demos
    results = []

    sat1, time1 = demo_simple_maze()
    results.append(("Simple Maze (10×10)", sat1, time1))

    sat2, time2 = demo_corridor_maze()
    results.append(("Corridor Maze (15×15)", sat2, time2))

    sat3, time3 = demo_spiral_maze()
    results.append(("Spiral Maze (15×15)", sat3, time3))

    # Analysis
    comparison_analysis(results)

    # Summary
    print_section("Demo Complete")
    print("Successfully demonstrated maze path-finding with DTM:")
    print("  ✓ Simple maze (scattered walls)")
    print("  ✓ Corridor maze (structured obstacles)")
    print("  ✓ Spiral maze (complex pattern)")
    print("  ✓ Performance comparison analysis")
    print()
    print("The maze problems test DTM's capability for problems with:")
    print("  - Smooth energy landscapes")
    print("  - Local connectivity constraints")
    print("  - Meaningful intermediate states")
    print()
    print("This demonstrates DTM's strength in continuous optimization tasks.")


if __name__ == "__main__":
    main()
