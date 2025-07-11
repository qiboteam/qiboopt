import networkx as nx
import numpy as np
import pytest
from qibo.hamiltonians import SymbolicHamiltonian

from qiboopt.combinatorial_classes.combinatorial_classes import (
    TSP,
    Mis,
    _calculate_two_to_one,
    _tsp_mixer,
    _tsp_phaser,
)
from qiboopt.opt_class.opt_class import (
    QUBO,
    linear_problem,
)


def test__calculate_two_to_one():
    num_cities = 3
    result = _calculate_two_to_one(num_cities)
    expected = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert np.array_equal(
        result, expected
    ), "calculate_two_to_one did not return the expected result"


def test__tsp_phaser():
    distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]])
    hamiltonian = _tsp_phaser(distance_matrix)
    assert isinstance(
        hamiltonian, SymbolicHamiltonian
    ), "tsp_phaser did not return a SymbolicHamiltonian"
    assert (
        hamiltonian.terms is not None
    ), "tsp_phaser returned a Hamiltonian with no terms"


def test__tsp_mixer():
    num_cities = 3
    hamiltonian = _tsp_mixer(num_cities)
    assert isinstance(
        hamiltonian, SymbolicHamiltonian
    ), "tsp_mixer did not return a SymbolicHamiltonian"
    assert (
        hamiltonian.terms is not None
    ), "tsp_mixer returned a Hamiltonian with no terms"


def test_tsp_class():
    distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]])
    tsp = TSP(distance_matrix)
    assert tsp.num_cities == 3, "TSP class did not set the number of cities correctly"
    assert np.array_equal(
        tsp.distance_matrix, distance_matrix
    ), "TSP class did not set the distance matrix correctly"

    obj_hamil, mixer = tsp.hamiltonians()
    assert isinstance(
        obj_hamil, SymbolicHamiltonian
    ), "TSP.hamiltonians did not return a SymbolicHamiltonian for the objective Hamiltonian"
    assert isinstance(
        mixer, SymbolicHamiltonian
    ), "TSP.hamiltonians did not return a SymbolicHamiltonian for the mixer"

    ordering = [0, 1, 2]
    initial_state = tsp.prepare_initial_state(ordering)
    assert (
        initial_state is not None
    ), "TSP.prepare_initial_state did not return a valid state"


def run_tests():
    # Test Setup
    num_cities = 4
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]
    two_to_one = lambda u, j: u * num_cities + j  # Example two_to_one mapping
    tsp = TSP(num_cities, distance_matrix, two_to_one)  # Assuming TSP class exists

    # Test 1: Basic functionality with a moderate penalty value
    penalty = 1.0
    qp = tsp.penalty_method(penalty)
    assert isinstance(qp, QUBO), "Test 1 Failed: Returned object is not a QUBO."
    assert len(qp.q_dict) > 0, "Test 1 Failed: QUBO dictionary is empty."

    # Test 2: Zero penalty
    penalty = 0.0
    qp = tsp.penalty_method(penalty)
    for key, value in qp.q_dict.items():
        expected_value = distance_matrix[key[0] // num_cities][key[1] // num_cities]
        assert (
            value == expected_value
        ), f"Test 2 Failed: Expected {expected_value} but got {value} for key {key}."

    # Test 3: High penalty
    penalty = 1000.0
    qp = tsp.penalty_method(penalty)
    for key, value in qp.q_dict.items():
        assert (
            abs(value) >= 1000
        ), f"Test 3 Failed: Value {value} is less than expected penalty."

    # Test 4: Single city (edge case)
    tsp.num_cities = 1
    tsp.distance_matrix = [[0]]
    qp = tsp.penalty_method(penalty=1.0)
    assert (
        len(qp.q_dict) == 0
    ), "Test 4 Failed: QUBO dictionary should be empty for a single city."

    # Test 5: Two cities (small problem)
    tsp.num_cities = 2
    tsp.distance_matrix = [[0, 10], [10, 0]]
    qp = tsp.penalty_method(penalty=1.0)
    assert (
        len(qp.q_dict) > 0
    ), "Test 5 Failed: QUBO dictionary should not be empty for two cities."


@pytest.fixture
def setup_tsp():
    num_cities = 3
    distance_matrix = [[0, 10, 15], [10, 0, 20], [15, 20, 0]]
    two_to_one = lambda u, j: u * num_cities + j
    return num_cities, distance_matrix, two_to_one


def test_qubo_objective_function(setup_tsp):
    num_cities, distance_matrix, two_to_one = setup_tsp

    # Initialize q_dict and compute objective function part
    q_dict = {}
    for u in range(num_cities):
        for v in range(num_cities):
            if v != u:
                for j in range(num_cities):
                    q_dict[two_to_one(u, j), two_to_one(v, (j + 1) % num_cities)] = (
                        distance_matrix[u][v]
                    )

    # Check that the q_dict is correctly populated
    expected_q_dict = {
        (0, 4): 10,
        (1, 5): 10,
        (2, 3): 10,
        (0, 7): 15,
        (1, 8): 15,
        (2, 6): 15,
        (3, 1): 10,
        (4, 2): 10,
        (5, 0): 10,
        (3, 7): 20,
        (4, 8): 20,
        (5, 6): 20,
        (6, 1): 15,
        (7, 2): 15,
        (8, 0): 15,
        (6, 4): 20,
        (7, 5): 20,
        (8, 3): 20,
    }
    assert q_dict == expected_q_dict, f"Expected {expected_q_dict}, but got {q_dict}"


def test_row_constraints(setup_tsp):
    num_cities, _, two_to_one = setup_tsp
    penalty = 10
    qp = QUBO(0, {})

    for v in range(num_cities):
        row_constraint = np.array([0 for _ in range(num_cities**2)])
        for j in range(num_cities):
            row_constraint[two_to_one(v, j)] = 1
        lp = linear_problem(row_constraint, -1)
        tmp_qp = lp.square()
        tmp_qp *= penalty
        qp += tmp_qp

    # Check that the row constraints are correctly applied to the QUBO
    expected_row_constraints = {
        (0, 0): -10,
        (0, 1): 10,
        (0, 2): 10,
        (0, 3): 0,
        (0, 4): 0,
        (0, 5): 0,
        (0, 6): 0,
        (0, 7): 0,
        (0, 8): 0,
        (1, 0): 10,
        (1, 1): -10,
        (1, 2): 10,
        (1, 3): 0,
        (1, 4): 0,
        (1, 5): 0,
        (1, 6): 0,
        (1, 7): 0,
        (1, 8): 0,
        (2, 0): 10,
        (2, 1): 10,
        (2, 2): -10,
        (2, 3): 0,
        (2, 4): 0,
        (2, 5): 0,
        (2, 6): 0,
        (2, 7): 0,
        (2, 8): 0,
        (3, 0): 0,
        (3, 1): 0,
        (3, 2): 0,
        (3, 3): -10,
        (3, 4): 10,
        (3, 5): 10,
        (3, 6): 0,
        (3, 7): 0,
        (3, 8): 0,
        (4, 0): 0,
        (4, 1): 0,
        (4, 2): 0,
        (4, 3): 10,
        (4, 4): -10,
        (4, 5): 10,
        (4, 6): 0,
        (4, 7): 0,
        (4, 8): 0,
        (5, 0): 0,
        (5, 1): 0,
        (5, 2): 0,
        (5, 3): 10,
        (5, 4): 10,
        (5, 5): -10,
        (5, 6): 0,
        (5, 7): 0,
        (5, 8): 0,
        (6, 0): 0,
        (6, 1): 0,
        (6, 2): 0,
        (6, 3): 0,
        (6, 4): 0,
        (6, 5): 0,
        (6, 6): -10,
        (6, 7): 10,
        (6, 8): 10,
        (7, 0): 0,
        (7, 1): 0,
        (7, 2): 0,
        (7, 3): 0,
        (7, 4): 0,
        (7, 5): 0,
        (7, 6): 10,
        (7, 7): -10,
        (7, 8): 10,
        (8, 0): 0,
        (8, 1): 0,
        (8, 2): 0,
        (8, 3): 0,
        (8, 4): 0,
        (8, 5): 0,
        (8, 6): 10,
        (8, 7): 10,
        (8, 8): -10,
    }
    assert (
        qp.Qdict == expected_row_constraints
    ), f"Expected {expected_row_constraints}, but got {qp.Qdict}"


def test_column_constraints(setup_tsp):
    num_cities, _, two_to_one = setup_tsp
    penalty = 10
    qp = QUBO(0, {})

    for j in range(num_cities):
        col_constraint = np.array([0 for _ in range(num_cities**2)])
        for v in range(num_cities):
            col_constraint[two_to_one(v, j)] = 1
        lp = linear_problem(col_constraint, -1)
        tmp_qp = lp.square()
        tmp_qp *= penalty
        qp += tmp_qp

    # Check that the column constraints are correctly applied to the QUBO
    expected_col_constraints = {
        (0, 0): -10,
        (0, 1): 0,
        (0, 2): 0,
        (0, 3): 10,
        (0, 4): 0,
        (0, 5): 0,
        (0, 6): 10,
        (0, 7): 0,
        (0, 8): 0,
        (1, 0): 0,
        (1, 1): -10,
        (1, 2): 0,
        (1, 3): 0,
        (1, 4): 10,
        (1, 5): 0,
        (1, 6): 0,
        (1, 7): 10,
        (1, 8): 0,
        (2, 0): 0,
        (2, 1): 0,
        (2, 2): -10,
        (2, 3): 0,
        (2, 4): 0,
        (2, 5): 10,
        (2, 6): 0,
        (2, 7): 0,
        (2, 8): 10,
        (3, 0): 10,
        (3, 1): 0,
        (3, 2): 0,
        (3, 3): -10,
        (3, 4): 0,
        (3, 5): 0,
        (3, 6): 10,
        (3, 7): 0,
        (3, 8): 0,
        (4, 0): 0,
        (4, 1): 10,
        (4, 2): 0,
        (4, 3): 0,
        (4, 4): -10,
        (4, 5): 0,
        (4, 6): 0,
        (4, 7): 10,
        (4, 8): 0,
        (5, 0): 0,
        (5, 1): 0,
        (5, 2): 10,
        (5, 3): 0,
        (5, 4): 0,
        (5, 5): -10,
        (5, 6): 0,
        (5, 7): 0,
        (5, 8): 10,
        (6, 0): 10,
        (6, 1): 0,
        (6, 2): 0,
        (6, 3): 10,
        (6, 4): 0,
        (6, 5): 0,
        (6, 6): -10,
        (6, 7): 0,
        (6, 8): 0,
        (7, 0): 0,
        (7, 1): 10,
        (7, 2): 0,
        (7, 3): 0,
        (7, 4): 10,
        (7, 5): 0,
        (7, 6): 0,
        (7, 7): -10,
        (7, 8): 0,
        (8, 0): 0,
        (8, 1): 0,
        (8, 2): 10,
        (8, 3): 0,
        (8, 4): 0,
        (8, 5): 10,
        (8, 6): 0,
        (8, 7): 0,
        (8, 8): -10,
    }
    assert (
        qp.Qdict == expected_col_constraints
    ), f"Expected {expected_col_constraints}, but got {qp.Qdict}"


def test_tsp_penalty():
    distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]])
    tsp = TSP(distance_matrix)
    qp = tsp.penalty_method(10)
    expected_q_dict = {
        (0, 4): 0.9,
        (1, 5): 0.9,
        (2, 3): 0.9,
        (0, 7): 0.8,
        (1, 8): 0.8,
        (2, 6): 0.8,
        (3, 1): 0.4,
        (4, 2): 0.4,
        (5, 0): 0.4,
        (3, 7): 0.1,
        (4, 8): 0.1,
        (5, 6): 0.1,
        (6, 1): 0.0,
        (7, 2): 0.0,
        (8, 0): 0.0,
        (6, 4): 0.7,
        (7, 5): 0.7,
        (8, 3): 0.7,
        (0, 0): -20,
        (0, 1): 10,
        (0, 2): 10,
        (0, 3): 10,
        (0, 5): 0,
        (0, 6): 10,
        (0, 8): 0,
        (1, 0): 10,
        (1, 1): -20,
        (1, 2): 10,
        (1, 3): 0,
        (1, 4): 10,
        (1, 6): 0,
        (1, 7): 10,
        (2, 0): 10,
        (2, 1): 10,
        (2, 2): -20,
        (2, 4): 0,
        (2, 5): 10,
        (2, 7): 0,
        (2, 8): 10,
        (3, 0): 10,
        (3, 2): 0,
        (3, 3): -20,
        (3, 4): 10,
        (3, 5): 10,
        (3, 6): 10,
        (3, 8): 0,
        (4, 0): 0,
        (4, 1): 10,
        (4, 3): 10,
        (4, 4): -20,
        (4, 5): 10,
        (4, 6): 0,
        (4, 7): 10,
        (5, 1): 0,
        (5, 2): 10,
        (5, 3): 10,
        (5, 4): 10,
        (5, 5): -20,
        (5, 7): 0,
        (5, 8): 10,
        (6, 0): 10,
        (6, 2): 0,
        (6, 3): 10,
        (6, 5): 0,
        (6, 6): -20,
        (6, 7): 10,
        (6, 8): 10,
        (7, 0): 0,
        (7, 1): 10,
        (7, 3): 0,
        (7, 4): 10,
        (7, 6): 10,
        (7, 7): -20,
        (7, 8): 10,
        (8, 1): 0,
        (8, 2): 10,
        (8, 4): 0,
        (8, 5): 10,
        (8, 6): 10,
        (8, 7): 10,
        (8, 8): -20,
    }
    assert (
        qp.Qdict == expected_q_dict
    ), f"Expected {expected_q_dict}, but got {qp.Qdict}"


def test_mis_class():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0)])
    mis = Mis(g)
    assert mis.n == 3, "Mis class did not set the number of nodes correctly"
    assert mis.g == g, "Mis class did not set the graph correctly"

    penalty = 10
    qp = mis.penalty_method(penalty)
    assert isinstance(qp, QUBO), "Mis.penalty_method did not return a QUBO"

    mis_str = str(mis)
    assert mis_str == "Mis", "Mis.__str__ did not return the expected string"
