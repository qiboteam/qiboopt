import networkx as nx
import numpy as np
import pytest
import qiboopt.combinatorial.combinatorial as combinatorial
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models import QAOA
from test_models_variational import assert_regression_fixture

from qiboopt.combinatorial.combinatorial import (
    CombinatorialQAOA,
    CombinatorialMAQAOA,
    CombinatorialLRQAOA,
    MIS,
    TSP,
    MaxCut,
    _calculate_two_to_one,
    _edge_list_from_W,
    _ensure_weight_matrix,
    _maxcut_mixer,
    _maxcut_phaser,
    _normalize,
    _tsp_mixer,
    _tsp_phaser,
)
from qiboopt.opt_class.opt_class import QUBO, LinearProblem


def test__calculate_two_to_one():
    num_cities = 3
    result, _ = _calculate_two_to_one(num_cities)
    expected_array = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    expected_dict = {
        (i, j): int(expected_array[i, j])
        for i in range(num_cities)
        for j in range(num_cities)
    }
    assert (
        expected_dict == result
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
        lp = LinearProblem(row_constraint, -1)
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
        lp = LinearProblem(col_constraint, -1)
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


@pytest.mark.parametrize(
    "distance, penalty",
    [
        (np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]]), 10),
        (
            np.array(
                [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
            ),
            0,
        ),
        (
            np.array(
                [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
            ),
            1000.0,
        ),
        (np.array([[0, 10], [10, 0]]), 1.0),
        (np.array([0]), 1),
    ],
)
def test_tsp_penalty(distance, penalty):
    tsp = TSP(distance)
    qp = tsp.penalty_method(penalty)
    two_to_one, one_to_two = _calculate_two_to_one(tsp.num_cities)
    assert isinstance(qp, QUBO), "TSP.penalty_method() did not return a QUBO."
    assert qp.Qdict, "QUBO dictionary from TSP.penalty_method() is empty."
    for key, value in qp.Qdict.items():
        # key is a pair of indices of the QUBO, we have to get the corresponding city.
        origin, origin_slot = one_to_two[
            key[0]
        ]  # recall the two indices are of the form of (city, slot)
        destination, destination_slot = one_to_two[key[1]]
        if (
            origin != destination
            and (destination_slot - origin_slot) % tsp.num_cities == 1
        ):
            expected_value = distance[origin][destination]
        elif origin == destination and origin_slot == destination_slot:
            expected_value = -2 * penalty
        elif (origin == destination and origin_slot != destination_slot) or (
            origin != destination and origin_slot == destination_slot
        ):
            expected_value = penalty
        else:
            expected_value = 0.0
        assert (
            value == expected_value
        ), f"penalty test: Expected {expected_value} but got {value} for key {key}."


def qaoa_function_of_layer(backend, layer):
    """
    This is a function to study the impact of the number of layers on QAOA, it takes
    in the number of layers and compute the distance of the mode of the histogram obtained
    from QAOA
    """
    num_cities = 3
    distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1], [0, 0.7, 0]])
    # there are two possible cycles, one with distance 1, one with distance 1.9
    distance_matrix = distance_matrix.round(1)

    small_tsp = TSP(distance_matrix, backend=backend)
    initial_state = small_tsp.prepare_initial_state([i for i in range(num_cities)])
    obj_hamil, mixer = small_tsp.hamiltonians()
    qaoa = QAOA(obj_hamil, mixer=mixer)
    initial_state = backend.cast(initial_state, copy=True)
    best_energy, final_parameters, extra = qaoa.minimize(
        initial_p=[0.1 for i in range(layer)],
        initial_state=initial_state,
        method="BFGS",
        options={"maxiter": 1},
    )
    qaoa.set_parameters(final_parameters)
    return qaoa.execute(initial_state)


@pytest.mark.parametrize("nlayers", [2, 4])
def test_tsp(backend, nlayers):
    if nlayers == 4 and backend.platform in ("cupy", "cuquantum"):
        pytest.skip("Failing for cupy and cuquantum.")
    final_state = backend.to_numpy(qaoa_function_of_layer(backend, nlayers))
    atol = 4e-5 if backend.platform in ("cupy", "cuquantum") else 1e-5
    assert_regression_fixture(
        backend, final_state.real, f"tsp_layer{nlayers}_real.out", rtol=1e-3, atol=atol
    )
    assert_regression_fixture(
        backend, final_state.imag, f"tsp_layer{nlayers}_imag.out", rtol=1e-3, atol=atol
    )


def test_mis_class():
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 0)])
    mis = MIS(g)
    assert mis.n == 3, "MIS class did not set the number of nodes correctly"
    assert mis.g == g, "MIS class did not set the graph correctly"

    penalty = 10
    qp = mis.penalty_method(penalty)
    assert isinstance(qp, QUBO), "MIS.penalty_method did not return a QUBO"

    mis_str = str(mis)
    assert mis_str == "MIS", "MIS.__str__ did not return the expected string"


def test_maxcut_helper_functions():
    weights = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float)
    matrix_form, nodes = _ensure_weight_matrix(weights)
    assert np.array_equal(matrix_form, weights)
    assert nodes == [0, 1, 2]

    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 1.5), ("b", "c", 2.0)])
    graph_matrix, graph_nodes = _ensure_weight_matrix(graph)
    idx = {node: i for i, node in enumerate(graph_nodes)}
    assert pytest.approx(graph_matrix[idx["a"], idx["b"]]) == 1.5
    assert pytest.approx(graph_matrix[idx["b"], idx["c"]]) == 2.0

    with pytest.raises(ValueError):
        _ensure_weight_matrix(np.ones((2, 3)))
    with pytest.raises(ValueError):
        _ensure_weight_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    loop_graph = nx.Graph()
    loop_graph.add_weighted_edges_from([("a", "a", 5.0), ("a", "b", 2.5)])
    loop_matrix, loop_nodes = _ensure_weight_matrix(loop_graph)
    loop_idx = {node: i for i, node in enumerate(loop_nodes)}
    assert loop_matrix[loop_idx["a"], loop_idx["a"]] == 0.0
    assert pytest.approx(loop_matrix[loop_idx["a"], loop_idx["b"]]) == 2.5
    with pytest.raises(TypeError):
        _ensure_weight_matrix(123)

    same, same_scale = _normalize(weights, "none")
    assert np.array_equal(same, weights) and same_scale == 1.0

    maxdeg_scaled, maxdeg_scale = _normalize(weights, "maxdeg")
    expected_maxdeg = np.max(np.sum(np.abs(weights), axis=1))
    assert pytest.approx(maxdeg_scale) == expected_maxdeg
    assert np.allclose(maxdeg_scaled, weights / maxdeg_scale)

    sum_scaled, sum_scale = _normalize(weights, "sum")
    expected_sum = np.sum(np.abs(np.triu(weights, 1)))
    assert pytest.approx(sum_scale) == expected_sum
    assert np.allclose(sum_scaled, weights / sum_scale)

    zeros = np.zeros((2, 2), dtype=float)
    zero_maxdeg, zero_maxdeg_scale = _normalize(zeros, "maxdeg")
    assert zero_maxdeg_scale == pytest.approx(1.0)
    assert np.array_equal(zero_maxdeg, zeros)
    zero_sum, zero_sum_scale = _normalize(zeros, "sum")
    assert zero_sum_scale == pytest.approx(1.0)
    assert np.array_equal(zero_sum, zeros)

    with pytest.raises(ValueError):
        _normalize(weights, "invalid")

    edges = _edge_list_from_W(weights)
    assert set(edges) == {(0, 1), (1, 2)}


def test_maxcut_mixer_variants():
    ham_x = _maxcut_mixer(3, mode="x")
    assert isinstance(ham_x, SymbolicHamiltonian)

    edges = [(0, 1), (1, 2)]
    ham_xy = _maxcut_mixer(3, mode="xy", edges=edges)
    assert isinstance(ham_xy, SymbolicHamiltonian)
    fallback_xy = _maxcut_mixer(3, mode="xy")
    assert isinstance(fallback_xy, SymbolicHamiltonian)
    assert len(fallback_xy.terms) == 6

    with pytest.raises(ValueError):
        _maxcut_mixer(2, mode="z")


def test_maxcut_prepare_initial_state_variants(monkeypatch):
    weights = np.zeros((2, 2), dtype=float)
    mc = MaxCut(weights)

    plus_state = mc.prepare_initial_state()
    assert np.allclose(plus_state, np.full(4, 0.5))

    basis_state = mc.prepare_initial_state(init="bitstring", bitstring="10")
    assert np.array_equal(basis_state, np.array([0, 0, 1, 0], dtype=complex))

    with pytest.raises(ValueError):
        mc.prepare_initial_state(init="bitstring")
    with pytest.raises(ValueError):
        mc.prepare_initial_state(init="bitstring", bitstring="1")

    class _DeterministicRNG:
        def integers(self, low, high=None, size=None):
            return np.array([1, 0], dtype=int)

    monkeypatch.setattr(
        "qiboopt.combinatorial.combinatorial.np.random.default_rng",
        lambda: _DeterministicRNG(),
    )
    random_state = mc.prepare_initial_state(init="random")
    assert np.array_equal(random_state, basis_state)

    with pytest.raises(ValueError):
        mc.prepare_initial_state(init="invalid")


def test_maxcut_qubo_and_cut_metrics():
    weights = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    mc = MaxCut(weights, normalize="maxdeg", mixer="x")

    obj_h, mix_h = mc.hamiltonians()
    assert isinstance(obj_h, SymbolicHamiltonian)
    assert isinstance(mix_h, SymbolicHamiltonian)

    qubo_max = mc.to_qubo(maximize=True, use_scaled=False)
    qubo_min = mc.to_qubo(maximize=False, use_scaled=False)

    assert qubo_max.Qdict[(0, 0)] == pytest.approx(-3.0)
    assert qubo_max.Qdict[(0, 1)] == pytest.approx(2.0)
    assert qubo_min.Qdict[(0, 1)] == pytest.approx(-2.0)

    qubo_scaled = mc.to_qubo(maximize=True, use_scaled=True)
    scale = np.max(np.sum(np.abs(weights), axis=1))
    assert qubo_scaled.Qdict[(0, 1)] == pytest.approx(qubo_max.Qdict[(0, 1)] / scale)

    bitstring = "010"
    cut_true = mc.cut_value(bitstring)
    cut_scaled = mc.cut_value(bitstring, use_scaled=True)
    assert cut_true == pytest.approx(4.0)
    assert cut_scaled == pytest.approx(cut_true / scale)

    S, Sp = mc.partition_from_bits(bitstring)
    assert S == {0, 2} and Sp == {1}

    with pytest.raises(ValueError):
        mc.cut_value("01")
    with pytest.raises(ValueError):
        mc.partition_from_bits("01")

    assert mc.rescale_energy(0.5) == pytest.approx(mc.energy_scale * 0.5)
    assert str(mc) == "MaxCut(n=3, normalize='maxdeg', mixer='x')"

    zero_weights = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]], dtype=float)
    zero_mc = MaxCut(zero_weights)
    zero_qubo = zero_mc.to_qubo()
    assert (0, 2) not in zero_qubo.Qdict
    assert zero_mc.cut_value("010") == pytest.approx(3.0)


def test_maxcut_phaser_skips_zero_weights():
    weights = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]], dtype=float)
    ham = _maxcut_phaser(weights)
    assert isinstance(ham, SymbolicHamiltonian)
    assert len(ham.terms) == 2


class _DummyBackend:
    def plus_state(self, nqubits):
        return np.ones(2**nqubits, dtype=float)

    def cast(self, arr, copy=False):
        return np.array(arr, copy=copy)

    def to_numpy(self, arr):
        return np.array(arr)


class _DummyHamiltonian:
    def __init__(self, backend, nqubits=1):
        self.backend = backend
        self.nqubits = nqubits

    def expectation(self, state):
        return float(np.sum(state))


class _DummyProblem:
    def __init__(self, backend):
        self.backend = backend
        self._cost = _DummyHamiltonian(backend)
        self._mixer = _DummyHamiltonian(backend)

    def hamiltonians(self):
        return self._cost, self._mixer

    def prepare_initial_state(self, offset=0):
        base_state = np.array([0.0, 1.0], dtype=float)
        return base_state + offset


class _FakeSolver:
    def __init__(self):
        self.dt = 0.0

    def __call__(self, state):
        return state + self.dt


class _FakeQAOA:
    created = []

    def __init__(self, cost, mixer, **kwargs):
        self.cost = cost
        self.mixer = mixer
        self.kwargs = kwargs
        self.backend = cost.backend
        self.callbacks = []
        self.parameters = None
        self.minimize_calls = []
        _FakeQAOA.created.append(self)

    def minimize(self, initial_p, method, **kwargs):
        self.minimize_calls.append((initial_p, method, kwargs))
        return "best", np.array(initial_p), {"method": method}

    def set_parameters(self, params):
        self.parameters = list(params)

    def execute(self, state):
        return {"state": state, "params": self.parameters}

    def normalize_state(self, state):
        return state

    @property
    def optimizers(self):
        return self

    def optimize(self, loss, initial_vector, args, method, **kwargs):
        result = loss(initial_vector, *args)
        params = np.array(initial_vector, copy=True) + 0.01
        return result, params, {"method": method}


@pytest.fixture
def dummy_setup():
    backend = _DummyBackend()
    return backend, _DummyProblem(backend)


@pytest.fixture
def fake_qaoa(monkeypatch):
    _FakeQAOA.created = []
    monkeypatch.setattr(combinatorial, "QAOA", _FakeQAOA)
    return _FakeQAOA


@pytest.fixture
def fake_solver(monkeypatch):
    monkeypatch.setattr(
        combinatorial, "get_solver", lambda name, step, ham: _FakeSolver()
    )


def test_combinatorialqaoa_requires_positive_layers(dummy_setup, fake_qaoa):
    _, problem = dummy_setup
    with pytest.raises(ValueError):
        CombinatorialQAOA(problem, layers=0)


def test_combinatorialqaoa_prepare_state_guard(dummy_setup, fake_qaoa):
    backend, _ = dummy_setup

    class _NoPrepare:
        def hamiltonians(self):
            ham = _DummyHamiltonian(backend)
            return ham, ham

    helper = CombinatorialQAOA(_NoPrepare(), layers=1)
    with pytest.raises(ValueError):
        helper._prepare_state()


def test_combinatorialqaoa_minimize_and_execute(dummy_setup, fake_qaoa):
    _, problem = dummy_setup
    helper = CombinatorialQAOA(problem, layers=2)

    best, params, info = helper.minimize(state_kwargs={"offset": 1})
    qaoa_instance = fake_qaoa.created[-1]
    assert best == "best"
    assert np.allclose(qaoa_instance.minimize_calls[0][0], [0.1, 0.1])
    assert np.array_equal(
        qaoa_instance.minimize_calls[0][2]["initial_state"], np.array([1.0, 2.0])
    )
    assert params.tolist() == [0.1, 0.1]
    assert info["method"] == "BFGS"

    executed = helper.execute(parameters=[0.5, 0.6], state_kwargs={"offset": 2})
    assert np.array_equal(executed["state"], np.array([2.0, 3.0]))
    assert executed["params"] == [0.5, 0.6]


def test_combinatoriallrqaoa_lr_schedule_and_params(dummy_setup, fake_qaoa):
    _, problem = dummy_setup
    helper = CombinatorialLRQAOA(
        problem, layers=2, delta_beta=0.3, delta_gamma=0.6
    )
    gammas, betas = helper.lr_schedule()
    assert np.allclose(gammas, [0.3, 0.6])
    assert np.allclose(betas, [0.3, 0.15])
    assert np.allclose(helper.lr_parameters(), [0.3, 0.3, 0.6, 0.15])


def test_combinatoriallrqaoa_execute_uses_lr_params(dummy_setup, fake_qaoa):
    _, problem = dummy_setup
    helper = CombinatorialLRQAOA(
        problem, layers=2, delta_beta=0.3, delta_gamma=0.6
    )
    initial_state = np.array([1.0, 2.0])
    out = helper.execute(initial_state=initial_state)
    assert np.array_equal(out["state"], initial_state)
    assert np.allclose(out["params"], [0.3, 0.3, 0.6, 0.15])


def test_combinatoriallrqaoa_minimize_disabled(dummy_setup, fake_qaoa):
    _, problem = dummy_setup
    helper = CombinatorialLRQAOA(problem, layers=1)
    with pytest.raises(NotImplementedError):
        helper.minimize()


def test_combinatoriallrqaoa_sweep_deltas(dummy_setup, fake_qaoa):
    _, problem = dummy_setup
    helper = CombinatorialLRQAOA(problem, layers=1)

    def _score(delta_beta, delta_gamma):
        return delta_beta + delta_gamma

    best, results = helper.sweep_deltas(
        [0.2, 0.1], [0.4, 0.3], score_fn=_score, update_best=True
    )

    assert len(results) == 4
    assert best["delta_beta"] == pytest.approx(0.1)
    assert best["delta_gamma"] == pytest.approx(0.3)
    assert helper.delta_beta == pytest.approx(0.1)
    assert helper.delta_gamma == pytest.approx(0.3)
    assert np.allclose(best["parameters"], helper.lr_parameters(0.1, 0.3))


def test_combinatorialmaqaoa_multi_angle_paths(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=2, ma_default_angle=0.2)

    assert helper._ma_total_angles == 4
    flattened = helper._ma_flatten_angles(
        {"cost": [[0.5], [0.7]], "mixer": [[0.6], [0.8]]}
    )
    assert flattened == [0.5, 0.6, 0.7, 0.8]
    with pytest.raises(ValueError):
        helper._ma_flatten_angles({"cost": [[0.1, 0.2]], "mixer": [[0.3], [0.4]]})

    executed_state = helper.execute(
        parameters={"cost": [[0.1], [0.2]], "mixer": [[0.3], [0.4]]}
    )
    assert np.allclose(executed_state, np.array([1.0, 2.0]))

    result, parameters, extra = helper.minimize(method="sgd")
    assert isinstance(result, float)
    assert len(helper._ma_cached_parameters) == helper._ma_total_angles
    assert np.allclose(parameters, np.array(helper._ma_cached_parameters))


def test_combinatorialqaoa_model_property(dummy_setup, fake_qaoa):
    _, problem = dummy_setup
    helper = CombinatorialQAOA(problem, layers=1)
    assert helper.model is fake_qaoa.created[-1]


def test_combinatorialmaqaoa_default_angles(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=2)
    defaults = helper._default_angles()
    assert len(defaults) == helper._ma_total_angles == 4


def test_combinatorialmaqaoa_ma_zero_layout_raises(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    with pytest.raises(ValueError):
        CombinatorialMAQAOA(
            problem,
            layers=1,
            ma_parameter_layout={"cost": 0, "mixer": 0},
        )


def test_combinatorialmaqaoa_ma_layout_length_mismatch(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    with pytest.raises(ValueError):
        CombinatorialMAQAOA(
            problem,
            layers=2,
            ma_parameter_layout={"cost": [1], "mixer": [1, 1]},
        )


def test_combinatorialmaqaoa_ma_layout_out_of_range(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    with pytest.raises(ValueError):
        CombinatorialMAQAOA(
            problem,
            layers=1,
            ma_parameter_layout={"cost": [5], "mixer": [0]},
        )


def test_combinatorialmaqaoa_flatten_angles_fallback(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=1)
    raw = helper._ma_flatten_angles(np.array([0.3, 0.4, 0.5]))
    assert raw == [0.3, 0.4, 0.5]


def test_combinatorialmaqaoa_flatten_angles_structured(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=1)
    flat = helper._ma_flatten_angles({"cost": [[0.1]], "mixer": [[0.2]]})
    assert flat == [0.1, 0.2]
    assert helper._ma_flatten_angles(None) == helper._ma_default_angles()


def test_combinatorialmaqaoa_flatten_angles_pair(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=1)
    pair_flat = helper._ma_flatten_angles(([[0.25]], [[0.5]]))
    assert pair_flat == [0.25, 0.5]


def test_combinatorialmaqaoa_normalize_layer_angles_mismatch(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=1)
    with pytest.raises(ValueError):
        helper._ma_normalize_layer_angles([[0.1, 0.2]], "cost")


def test_combinatorialmaqaoa_normalize_layer_angles_defaults(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=2)
    assert helper._ma_normalize_layer_angles(None, "cost") == [
        [helper.ma_default_angle],
        [helper.ma_default_angle],
    ]
    padded = helper._ma_normalize_layer_angles([[0.5]], "cost")
    assert padded[1] == [helper.ma_default_angle]


def test_combinatorialmaqaoa_ma_execute_guards(dummy_setup, fake_qaoa, fake_solver):
    backend, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=1)
    with pytest.raises(ValueError):
        helper._ma_execute([0.1])  # wrong length

    # Valid path uses default plus_state and callbacks branch
    params = helper._ma_default_angles()
    helper._qaoa.callbacks = ["cb"]
    called = {"times": 0}
    helper._qaoa.calculate_callbacks = lambda state: called.__setitem__(
        "times", called["times"] + 1
    )
    out = helper._ma_execute(params)
    assert called["times"] > 0
    expected = backend.plus_state(helper.cost_hamiltonian.nqubits) + sum(params)
    assert np.allclose(out, expected)


def test_combinatorialmaqaoa_ma_loss_variants(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=1)
    params = helper._ma_default_angles()

    # Default loss uses hamiltonian.expectation
    loss_val = helper._ma_loss(params, helper.cost_hamiltonian, None, None, {})
    assert isinstance(loss_val, float)

    # Custom user loss with filtered kwargs
    def _user_loss(hamiltonian, state, extra=None):
        return float(np.sum(state) + (extra or 0))

    custom = helper._ma_loss(
        params,
        helper.cost_hamiltonian,
        None,
        _user_loss,
        {"extra": 1.5, "ignored": 99},
    )
    assert isinstance(custom, float)


def test_combinatorialmaqaoa_execute_uses_default_ma_angles(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=1)
    helper._ma_cached_parameters = None
    out = helper.execute(parameters=None)
    assert isinstance(out, np.ndarray)


def test_combinatorialmaqaoa_sanitize_counts_with_none(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=2)
    assert helper._ma_sanitize_counts(None, 3) == [3, 3]


def test_combinatorialmaqaoa_execute_cursor_mismatch(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=1)
    helper._ma_layout = []
    helper._ma_total_angles = 1
    with pytest.raises(ValueError):
        helper._ma_execute([0.1])


def test_combinatorialmaqaoa_ma_minimize_non_sgd(dummy_setup, fake_qaoa, fake_solver):
    _, problem = dummy_setup
    helper = CombinatorialMAQAOA(problem, layers=1)
    best, params, extra = helper.minimize(method="Powell")
    assert np.allclose(best, float(np.array(best)))
    assert isinstance(extra, dict)
