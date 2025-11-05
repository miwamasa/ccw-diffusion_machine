#!/usr/bin/env python
"""
Potts Model Demo for DTM Hardware Simulator

Demonstrates graph coloring using the Potts model, inspired by:
https://github.com/extropic-ai/thrml/blob/main/examples/00_probabilistic_computing.ipynb

The Potts model is an energy-based model for categorical variables,
commonly used in statistical physics and machine learning.
"""

import sys
import time
import numpy as np
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dtm_simulator.core.dtm import DTM, DTMConfig
from dtm_simulator.problems.potts import PottsModel
from dtm_simulator.hardware.energy_model import EnergyModel


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_cycle_graph():
    """
    Demo: Graph coloring on a cycle graph.

    A cycle graph is a ring where each node is connected to its neighbors.
    Classic example: can be colored with 2 colors if even number of nodes,
    3 colors if odd number of nodes.
    """
    print_section("Cycle Graph Coloring (6 nodes, 3 colors)")

    # Create cycle graph: 0-1-2-3-4-5-0
    num_nodes = 6
    num_colors = 3

    print(f"\nProblem: Color a cycle of {num_nodes} nodes using {num_colors} colors")
    print("Constraint: Adjacent nodes must have different colors")

    problem = PottsModel.create_cycle_graph(num_nodes, num_colors)

    print(f"\nGraph structure:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {len(problem.edges)}")
    print(f"  Edge list: {problem.edges[:10]}")  # Show first 10 edges

    # Create DTM solver
    print("\nInitializing DTM solver...")
    config = DTMConfig(
        num_layers=2,
        grid_size=int(np.ceil(np.sqrt(problem.get_num_variables()))),
        K_infer=50,
        beta=1.0,
        seed=42
    )
    dtm = DTM(config)

    # Solve
    print("\nSolving graph coloring problem...")
    print("(This may take 10-20 seconds...)")
    start_time = time.time()

    best_x, info = dtm.solve(problem, max_steps=10000, verbose=False)

    solve_time = time.time() - start_time

    # Decode solution
    states = problem.decode_solution(best_x)

    print("\nSolution found!")
    print(problem.format_solution(states))

    # Check constraints
    satisfied, total = problem.check_constraints(best_x)
    print(f"\nConstraint satisfaction: {satisfied}/{total} ({satisfied/total*100:.1f}%)")
    print(f"Solving time: {solve_time:.2f} seconds")
    print(f"Best energy: {info['best_energy']:.4f}")

    # Visualize coloring
    print("\nColor assignments:")
    for i in range(num_nodes):
        next_node = (i + 1) % num_nodes
        arrow = "✓" if states[i] != states[next_node] else "✗"
        print(f"  Node {i} (color {states[i]}) -> Node {next_node} (color {states[next_node]}) {arrow}")

    if satisfied == total:
        print("\n✓ Perfect graph coloring found!")
    else:
        print(f"\n× Solution has {total - satisfied} constraint violations")

    return states, info


def demo_grid_graph():
    """
    Demo: Graph coloring on a 2D grid.

    Grid graphs are planar graphs that can be 4-colored
    (four color theorem).
    """
    print_section("2D Grid Graph Coloring (3×3 grid, 4 colors)")

    rows, cols = 3, 3
    num_colors = 4

    print(f"\nProblem: Color a {rows}×{cols} grid using {num_colors} colors")
    print("Constraint: Adjacent nodes (up/down/left/right) must have different colors")

    problem = PottsModel.create_grid_graph(rows, cols, num_colors)

    print(f"\nGraph structure:")
    print(f"  Nodes: {rows * cols}")
    print(f"  Edges: {len(problem.edges)}")

    # Create DTM solver
    print("\nInitializing DTM solver...")
    config = DTMConfig(
        num_layers=2,
        grid_size=int(np.ceil(np.sqrt(problem.get_num_variables()))),
        K_infer=50,
        beta=1.0,
        seed=42
    )
    dtm = DTM(config)

    # Solve
    print("\nSolving graph coloring problem...")
    print("(This may take 15-30 seconds...)")
    start_time = time.time()

    best_x, info = dtm.solve(problem, max_steps=15000, verbose=False)

    solve_time = time.time() - start_time

    # Decode solution
    states = problem.decode_solution(best_x)

    print("\nSolution found!")

    # Visualize as grid
    print("\nGrid coloring (colors represented as numbers):")
    grid = states.reshape(rows, cols)
    for i in range(rows):
        row_str = "  "
        for j in range(cols):
            row_str += f"[{grid[i, j]}] "
        print(row_str)

    # Check constraints
    satisfied, total = problem.check_constraints(best_x)
    print(f"\nConstraint satisfaction: {satisfied}/{total} ({satisfied/total*100:.1f}%)")
    print(f"Solving time: {solve_time:.2f} seconds")
    print(f"Best energy: {info['best_energy']:.4f}")

    # Check specific edges
    print("\nEdge constraint verification:")
    violations = 0
    for i, j in problem.edges[:10]:  # Show first 10 edges
        ri, ci = i // cols, i % cols
        rj, cj = j // cols, j % cols
        match = "same" if states[i] == states[j] else "different"
        status = "✗" if states[i] == states[j] else "✓"
        print(f"  ({ri},{ci})-({rj},{cj}): colors {states[i]}, {states[j]} ({match}) {status}")
        if states[i] == states[j]:
            violations += 1

    if satisfied == total:
        print("\n✓ Perfect graph coloring found!")
    else:
        print(f"\n× Solution has {total - satisfied} constraint violations")

    return states, info


def demo_random_graph():
    """
    Demo: Graph coloring on a random graph.

    Random graphs provide challenging coloring problems.
    """
    print_section("Random Graph Coloring (8 nodes, 3 colors)")

    num_nodes = 8
    num_colors = 3
    edge_prob = 0.4

    print(f"\nProblem: Color a random graph with {num_nodes} nodes using {num_colors} colors")
    print(f"Edge probability: {edge_prob}")

    problem = PottsModel.create_random_graph(num_nodes, num_colors, edge_prob, seed=42)

    print(f"\nGraph structure:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {len(problem.edges)}")
    print(f"  Edge density: {len(problem.edges) / (num_nodes * (num_nodes - 1) / 2):.2%}")

    # Create DTM solver
    print("\nInitializing DTM solver...")
    config = DTMConfig(
        num_layers=2,
        grid_size=int(np.ceil(np.sqrt(problem.get_num_variables()))),
        K_infer=50,
        beta=1.0,
        seed=43
    )
    dtm = DTM(config)

    # Solve
    print("\nSolving graph coloring problem...")
    print("(This may take 15-30 seconds...)")
    start_time = time.time()

    best_x, info = dtm.solve(problem, max_steps=12000, verbose=False)

    solve_time = time.time() - start_time

    # Decode solution
    states = problem.decode_solution(best_x)

    print("\nSolution found!")
    print(problem.format_solution(states))

    # Check constraints
    satisfied, total = problem.check_constraints(best_x)
    print(f"\nConstraint satisfaction: {satisfied}/{total} ({satisfied/total*100:.1f}%)")
    print(f"Solving time: {solve_time:.2f} seconds")
    print(f"Best energy: {info['best_energy']:.4f}")

    if satisfied == total:
        print("\n✓ Perfect graph coloring found!")
    else:
        print(f"\n× Solution has {total - satisfied} constraint violations")

    return states, info


def demo_hardware_energy():
    """Show hardware energy consumption for Potts model."""
    print_section("Hardware Energy Analysis")

    # Example problem
    num_nodes = 10
    num_colors = 3
    problem = PottsModel.create_cycle_graph(num_nodes, num_colors)

    print(f"\nAnalyzing energy for Potts model:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Colors: {num_colors}")
    print(f"  Binary variables: {problem.get_num_variables()}")

    # Energy model
    energy_model = EnergyModel()

    config = DTMConfig(num_layers=4, K_infer=250)

    breakdown = energy_model.compute_energy_breakdown(
        T=config.num_layers,
        K=config.K_infer,
        N=problem.get_num_variables()
    )

    print(f"\nEstimated hardware energy consumption:")
    print(f"  Total energy: {energy_model.format_energy(breakdown['total'])}")
    print(f"  RNG: {energy_model.format_energy(breakdown['rng'])}")
    print(f"  Bias circuits: {energy_model.format_energy(breakdown['bias'])}")
    print(f"  Clock: {energy_model.format_energy(breakdown['clock'])}")
    print(f"  Communication: {energy_model.format_energy(breakdown['neighbor'])}")

    # GPU comparison
    gpu_speedup = energy_model.compare_with_gpu(
        dtm_energy=breakdown['total'],
        problem_size=problem.get_num_variables()
    )
    print(f"\nEnergy efficiency vs GPU: {gpu_speedup:.1f}×")


def main():
    """Main demo function."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + " " * 18 + "POTTS MODEL DEMO - DTM SIMULATOR" + " " * 18 + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Graph Coloring with Energy-Based Probabilistic Computing  " + " " * 4 + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    print("\nThe Potts model is a categorical energy-based model from statistical")
    print("physics, widely used for graph coloring, image segmentation, and clustering.")
    print("\nInspired by: https://github.com/extropic-ai/thrml")

    try:
        # Demo 1: Cycle graph
        demo_cycle_graph()

        # Demo 2: Grid graph
        demo_grid_graph()

        # Demo 3: Random graph
        demo_random_graph()

        # Demo 4: Hardware energy
        demo_hardware_energy()

        # Summary
        print_section("Demo Complete")
        print("\nSuccessfully demonstrated Potts model with DTM:")
        print("  ✓ Cycle graph coloring")
        print("  ✓ 2D grid graph coloring")
        print("  ✓ Random graph coloring")
        print("  ✓ Hardware energy analysis")
        print("\nThe Potts model shows DTM's versatility for categorical problems.")
        print("Applications: graph coloring, clustering, image segmentation, etc.")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
