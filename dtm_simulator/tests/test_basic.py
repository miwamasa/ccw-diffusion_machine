"""
Basic tests for DTM simulator components.
"""

import pytest
import numpy as np
from dtm_simulator.core.boltzmann_machine import BoltzmannMachine
from dtm_simulator.hardware.rng_simulator import RNGSimulator
from dtm_simulator.hardware.energy_model import EnergyModel
from dtm_simulator.problems.sudoku import SudokuProblem
from dtm_simulator.problems.nqueen import NQueenProblem


def test_boltzmann_machine_creation():
    """Test Boltzmann machine can be created."""
    bm = BoltzmannMachine(L=5, connectivity="G8", beta=1.0, seed=42)
    assert bm.N == 25
    assert bm.L == 5
    assert bm.J.shape == (25, 25)


def test_boltzmann_machine_energy():
    """Test energy calculation."""
    bm = BoltzmannMachine(L=3, connectivity="G8", beta=1.0, seed=42)
    x = np.ones(9)  # All +1
    energy = bm.energy(x)
    assert isinstance(energy, (float, np.floating))


def test_gibbs_sampling():
    """Test Gibbs sampling runs."""
    bm = BoltzmannMachine(L=4, connectivity="G8", beta=1.0, seed=42)
    x_init = np.random.choice([-1, 1], size=16)
    x_final, _ = bm.sample(x_init, num_steps=10)
    assert x_final.shape == (16,)
    assert np.all(np.abs(x_final) == 1)  # All values are ±1


def test_rng_simulator():
    """Test RNG simulator."""
    rng_sim = RNGSimulator()
    bit = rng_sim.sample(0.5)
    assert bit in [0, 1]

    # Test vector sampling
    bits = rng_sim.sample_vector(np.array([0.0, 0.5, 1.0]))
    assert len(bits) == 3
    assert all(b in [0, 1] for b in bits)


def test_energy_model():
    """Test energy model calculations."""
    energy_model = EnergyModel()

    E_samp = energy_model.compute_sampling_energy(K=100, N=100)
    assert E_samp > 0

    E_total = energy_model.compute_total_energy(T=8, K=100, N=100)
    assert E_total > E_samp

    breakdown = energy_model.compute_energy_breakdown(T=8, K=100, N=100)
    assert "total" in breakdown
    assert breakdown["total"] > 0


def test_sudoku_problem():
    """Test Sudoku problem encoding."""
    # Simple 9x9 Sudoku (easy)
    puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    problem = SudokuProblem.from_string(puzzle_str)

    assert problem.get_num_variables() == 729  # 9x9x9

    # Test with random state
    x = np.random.choice([-1, 1], size=729)
    energy = problem.energy_function(x)
    assert isinstance(energy, (float, np.floating))

    # Decode solution
    solution = problem.decode_solution(x)
    assert solution.shape == (9, 9)

    # Check constraints
    satisfied, total = problem.check_constraints(x)
    assert total == 324  # 81*4 constraints


def test_nqueen_problem():
    """Test N-Queen problem encoding."""
    problem = NQueenProblem(N=8)

    assert problem.get_num_variables() == 64  # 8x8

    # Test with random state
    x = np.random.choice([-1, 1], size=64)
    energy = problem.energy_function(x)
    assert isinstance(energy, (float, np.floating))

    # Decode solution
    board = problem.decode_solution(x)
    assert board.shape == (8, 8)

    # Check constraints
    satisfied, total = problem.check_constraints(x)
    assert total > 0


def test_sudoku_format():
    """Test Sudoku solution formatting."""
    puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    problem = SudokuProblem.from_string(puzzle_str)

    # Create a simple solution (not necessarily valid)
    solution = np.random.randint(0, 10, size=(9, 9))
    formatted = problem.format_solution(solution)
    assert isinstance(formatted, str)
    assert len(formatted) > 0


def test_nqueen_format():
    """Test N-Queen solution formatting."""
    problem = NQueenProblem(N=8)

    # Create a simple board
    board = np.zeros((8, 8))
    board[0, 0] = 1  # One queen at (0,0)

    formatted = problem.format_solution(board)
    assert isinstance(formatted, str)
    assert "Q" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
