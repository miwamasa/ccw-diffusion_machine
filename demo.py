"""
DTM Hardware Simulator Demo

Demonstrates solving constraint satisfaction problems using DTM:
1. N-Queen problem (8-Queen)
2. Sudoku puzzle (Easy)

Shows hardware energy consumption statistics.
"""

import numpy as np
import sys
import time

# Add package to path
sys.path.insert(0, '/home/user/ccw-diffusion_machine')

from dtm_simulator.core.boltzmann_machine import BoltzmannMachine
from dtm_simulator.core.dtm import DTM, DTMConfig
from dtm_simulator.problems.nqueen import NQueenProblem
from dtm_simulator.problems.sudoku import SudokuProblem
from dtm_simulator.hardware.energy_model import EnergyModel


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_nqueen():
    """Demonstrate N-Queen problem solving."""
    print_section("8-Queen Problem Demo")

    # Create 8-Queen problem
    print("\nSetting up 8-Queen problem...")
    problem = NQueenProblem(N=8)

    print(f"Board size: 8×8")
    print(f"Number of variables: {problem.get_num_variables()}")

    # Create DTM solver (using smaller grid for faster demo)
    print("\nInitializing DTM solver...")
    config = DTMConfig(
        num_layers=4,
        grid_size=8,  # Match problem size
        connectivity="G8",
        K_infer=100,  # Fewer steps for demo
        beta=1.5,
        seed=42
    )
    dtm = DTM(config)

    # Solve
    print("\nSolving 8-Queen problem...")
    print("(This may take 30-60 seconds...)")
    start_time = time.time()

    best_x, info = dtm.solve(problem, max_steps=20000, verbose=True)

    solve_time = time.time() - start_time

    # Decode and display solution
    board = problem.decode_solution(best_x)
    print("\nSolution found!")
    print(problem.format_solution(board))

    # Check constraints
    satisfied, total = problem.check_constraints(best_x)
    satisfaction_rate = satisfied / total if total > 0 else 0

    print(f"\nConstraint satisfaction: {satisfied}/{total} ({satisfaction_rate*100:.1f}%)")
    print(f"Solving time: {solve_time:.2f} seconds")
    print(f"Final energy: {info['best_energy']:.4f}")

    # Violation breakdown
    violations = problem.count_violations(board)
    print("\nViolation breakdown:")
    print(f"  Row violations: {violations['row']}")
    print(f"  Column violations: {violations['col']}")
    print(f"  Diagonal violations: {violations['diag1']}")
    print(f"  Anti-diagonal violations: {violations['diag2']}")

    # Energy statistics
    energy_model = EnergyModel()
    total_energy = energy_model.compute_total_energy(
        T=config.num_layers,
        K=config.K_infer,
        N=problem.get_num_variables()
    )

    print(f"\nEstimated hardware energy consumption:")
    print(f"  Total energy: {energy_model.format_energy(total_energy)}")
    print(f"  Energy per constraint: {energy_model.format_energy(total_energy / total)}")

    return board, info


def demo_sudoku():
    """Demonstrate Sudoku problem solving."""
    print_section("Sudoku Problem Demo")

    # Easy Sudoku puzzle
    puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"

    print("\nInitial Sudoku puzzle:")
    problem = SudokuProblem.from_string(puzzle_str)

    # Display initial puzzle
    puzzle_display = problem.puzzle.copy()
    print(problem.format_solution(puzzle_display))

    print(f"\nNumber of variables: {problem.get_num_variables()}")
    print(f"Given clues: {np.sum(problem.given_mask)}")

    # Create DTM solver (smaller configuration for demo)
    print("\nInitializing DTM solver...")
    config = DTMConfig(
        num_layers=4,
        grid_size=27,  # Adjust for Sudoku (729 = 27^2)
        connectivity="G12",
        K_infer=150,
        beta=2.0,
        seed=42
    )
    dtm = DTM(config)

    # Solve
    print("\nSolving Sudoku puzzle...")
    print("(This may take 60-120 seconds...)")
    start_time = time.time()

    best_x, info = dtm.solve(problem, max_steps=50000, verbose=True)

    solve_time = time.time() - start_time

    # Decode and display solution
    solution = problem.decode_solution(best_x)
    print("\nSolution found!")
    print(problem.format_solution(solution))

    # Check constraints
    satisfied, total = problem.check_constraints(best_x)
    satisfaction_rate = satisfied / total if total > 0 else 0

    print(f"\nConstraint satisfaction: {satisfied}/{total} ({satisfaction_rate*100:.1f}%)")
    print(f"Solving time: {solve_time:.2f} seconds")
    print(f"Final energy: {info['best_energy']:.4f}")

    # Energy statistics
    energy_model = EnergyModel()
    total_energy = energy_model.compute_total_energy(
        T=config.num_layers,
        K=config.K_infer,
        N=problem.get_num_variables()
    )

    breakdown = energy_model.compute_energy_breakdown(
        T=config.num_layers,
        K=config.K_infer,
        N=problem.get_num_variables()
    )

    print(f"\nEstimated hardware energy consumption:")
    print(f"  Total energy: {energy_model.format_energy(total_energy)}")
    print(f"  Energy per constraint: {energy_model.format_energy(total_energy / total)}")
    print(f"\nEnergy breakdown:")
    print(f"  RNG: {energy_model.format_energy(breakdown['rng'])}")
    print(f"  Bias circuits: {energy_model.format_energy(breakdown['bias'])}")
    print(f"  Clock distribution: {energy_model.format_energy(breakdown['clock'])}")
    print(f"  Neighbor communication: {energy_model.format_energy(breakdown['neighbor'])}")

    # GPU comparison (rough estimate)
    gpu_speedup = energy_model.compare_with_gpu(
        dtm_energy=total_energy,
        problem_size=problem.get_num_variables(),
        gpu_energy_per_op=100e-12
    )
    print(f"\nEstimated energy efficiency vs GPU: {gpu_speedup:.1f}×")

    return solution, info


def demo_hardware_components():
    """Demonstrate hardware component simulations."""
    print_section("Hardware Component Simulation Demo")

    from dtm_simulator.hardware.rng_simulator import RNGSimulator
    from dtm_simulator.hardware.bias_circuit import BiasCircuit

    # RNG simulation
    print("\n1. RNG (Random Number Generator) Simulation")
    rng_sim = RNGSimulator(V_s=1.0, phi=0.5)

    print("   Sampling 1000 bits with different bias voltages...")
    for voltage in [0.0, 0.5, 1.0, 1.5]:
        bits = []
        for _ in range(1000):
            bit = rng_sim.sample(voltage)
            bits.append(bit)

        probability = np.mean(bits)
        print(f"   Voltage={voltage:.2f}V → P(1)={probability:.3f}")

    energy_formatter = EnergyModel()
    print(f"\n   Total energy consumed: {energy_formatter.format_energy(rng_sim.get_energy_consumption())}")

    # Bias circuit simulation
    print("\n2. Bias Circuit Simulation")
    bias_circuit = BiasCircuit(V_s=1.0, phi=0.5)

    print("   Converting probabilities to voltages:")
    for prob in [0.1, 0.5, 0.9]:
        voltage = bias_circuit.prob_to_voltage(prob)
        print(f"   P={prob:.1f} → V={voltage:.3f}V")

    # Boltzmann machine demo
    print("\n3. Boltzmann Machine Simulation")
    print("   Creating 5×5 Boltzmann machine with G12 connectivity...")

    bm = BoltzmannMachine(L=5, connectivity="G12", beta=1.0, seed=42)
    print(f"   Number of nodes: {bm.N}")
    print(f"   Number of connections: {bm.J.nnz}")

    # Sample
    print("   Running 50 Gibbs sampling steps...")
    x_init = np.random.choice([-1, 1], size=bm.N)
    x_final, _ = bm.sample(x_init, num_steps=50)

    print(f"   Initial energy: {bm.energy(x_init):.4f}")
    print(f"   Final energy: {bm.energy(x_final):.4f}")


def main():
    """Main demo function."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + " " * 15 + "DTM HARDWARE SIMULATOR DEMO" + " " * 25 + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Denoising Thermodynamic Models for Constraint Satisfaction  " + " " * 2 + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    try:
        # Demo 1: Hardware components
        demo_hardware_components()

        # Demo 2: 8-Queen problem
        demo_nqueen()

        # Demo 3: Sudoku puzzle
        demo_sudoku()

        # Summary
        print_section("Demo Complete")
        print("\nThe DTM simulator has successfully demonstrated:")
        print("  ✓ Hardware component simulation (RNG, Bias Circuit, BM)")
        print("  ✓ N-Queen problem solving")
        print("  ✓ Sudoku puzzle solving")
        print("  ✓ Energy consumption tracking")
        print("\nFor more information, see:")
        print("  - spec/specification.md for detailed documentation")
        print("  - dtm_simulator/tests/ for unit tests")
        print("\nThank you for using DTM Hardware Simulator!")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
