"""
Various combinatorial optimisation applications that are commonly formulated as QUBO problems.
"""

# pylint: disable=too-many-lines

import networkx as nx
import numpy as np
from qibo import gates
from qibo.backends import _check_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models import QAOA
from qibo.models.circuit import Circuit
from qibo.solvers import get_solver
from qibo.symbols import X, Y, Z

from qiboopt.opt_class.opt_class import (
    QUBO,
    LinearProblem,
    variable_to_ind,
)


def _calculate_two_to_one(num_cities):
    """
    Calculates a mapping from two coordinates to one coordinate for the TSP problem.

    Args:
        num_cities (int): The number of cities for the TSP.

    Returns:
        (dictionary): An array that map coordinates of two numbers to one.
    """
    pairs = [(i, j) for i in range(num_cities) for j in range(num_cities)]
    v2i, i2v = variable_to_ind(pairs)
    return v2i, i2v


def _tsp_phaser(distance_matrix, backend=None):
    """
    Constructs the phaser Hamiltonian for the Traveling Salesman Problem (TSP).

    Args:
        distance_matrix (np.ndarray): A matrix representing the distances between cities.
        backend: Backend to be used for the calculations (optional).

    Returns:
        :class:`qibo.hamiltonians.SymbolicHamiltonian`: Phaser Hamiltonian for TSP.
    """
    num_cities = distance_matrix.shape[0]
    two_to_one, _ = _calculate_two_to_one(num_cities)
    form = 0
    form = sum(
        distance_matrix[u, v]
        * Z(int(two_to_one[(u, i)]))
        * Z(int(two_to_one[(v, (i + 1) % num_cities)]))
        for i in range(num_cities)
        for u in range(num_cities)
        for v in range(num_cities)
        if u != v
    )
    ham = SymbolicHamiltonian(form, backend=backend)
    return ham


def _tsp_mixer(num_cities, backend=None):
    """
    Constructs the mixer Hamiltonian for the Traveling Salesman Problem (TSP).

    Args:
        num_cities (int): The number of cities in the TSP.
        backend: Backend to be used for the calculations (optional).

    Returns:
        SymbolicHamiltonian: The mixer Hamiltonian for TSP.
    """

    two_to_one, _ = _calculate_two_to_one(num_cities)

    def splus(u, i):
        """
        Defines the S+ operator for a specific city and position.

        Args:
            u (int): City index.
            i (int): Position index.

        Returns:
            SymbolicHamiltonian: The S+ operator.
        """
        return X(int(two_to_one[(u, i)])) + 1j * Y(int(two_to_one[(u, i)]))

    def sminus(u, i):
        """
        Defines the S- operator for a specific city and position.

        Args:
            u (int): City index.
            i (int): Position index.

        Returns:
            SymbolicHamiltonian: The S- operator.
        """
        return X(int(two_to_one[(u, i)])) - 1j * Y(int(two_to_one[(u, i)]))

    form = 0
    form = sum(
        splus(u, i)
        * splus(v, (i + 1) % num_cities)
        * sminus(u, (i + 1) % num_cities)
        * sminus(v, i)
        + sminus(u, i)
        * sminus(v, (i + 1) % num_cities)
        * splus(u, (i + 1) % num_cities)
        * splus(v, i)
        for i in range(num_cities)
        for u in range(num_cities)
        for v in range(num_cities)
        if u != v
    )
    ham = SymbolicHamiltonian(form, backend=backend)
    return ham


class TSP:
    """
    Class representing the Travelling Salesman Problem (TSP). The implementation
    is based on the work by Hadfield.

    Args:
        distance_matrix: a numpy matrix encoding the distance matrix.
        two_to_one: Mapping from two coordinates to one coordinate
        backend: Backend to use for calculations. If not given the global backend will be used.

    Example:
        .. testcode::

            from qiboopt.combinatorial.combinatorial import TSP
            import numpy as np
            from collections import defaultdict
            from qibo import gates
            from qibo.models import QAOA
            from qibo.result import CircuitResult
            from qibo.models.circuit import Circuit


            def convert_to_standard_Cauchy(config):
                m = int(np.sqrt(len(config)))
                cauchy = [-1] * m  # Cauchy's notation for permutation, e.g. (1,2,0) or (2,0,1)
                for i in range(m):
                    for j in range(m):
                        if config[m * i + j] == '1':
                            cauchy[j] = i  # citi i is in slot j
                for i in range(m):
                    if cauchy[i] == 0:
                        cauchy = cauchy[i:] + cauchy[:i]
                        return tuple(cauchy)
                        # now, the cauchy notation for permutation begins with 0


            def evaluate_dist(cauchy):
                '''
                Given a permutation of 0 to n-1, we compute the distance of the tour

                '''
                m = len(cauchy)
                return sum(distance_matrix[cauchy[i]][cauchy[(i+1)%m]] for i in range(m))


            def qaoa_function_of_layer(layer, distance_matrix):
                '''
                This is a function to study the impact of the number of layers on QAOA,
                it takes in the number of layers and compute the distance of the mode
                of the histogram obtained from QAOA

                '''
                small_tsp = TSP(distance_matrix)
                obj_hamil, mixer = small_tsp.hamiltonians()
                qaoa = QAOA(obj_hamil, mixer=mixer)
                best_energy, final_parameters, extra = qaoa.minimize(initial_p=[0.1] * layer,
                                                     initial_state=initial_state, method='BFGS')
                qaoa.set_parameters(final_parameters)
                quantum_state = qaoa.execute(initial_state)
                circuit = Circuit(9)
                circuit.add(gates.M(*range(9)))
                result = CircuitResult(quantum_state, circuit.measurements,
                        small_tsp.backend, nshots=1000)
                freq_counter = result.frequencies()
                # let's combine freq_counter here, first convert each key and sum up the frequency
                cauchy_dict = defaultdict(int)
                for freq_key in freq_counter:
                    standard_cauchy_key = convert_to_standard_Cauchy(freq_key)
                    cauchy_dict[standard_cauchy_key] += freq_counter[freq_key]
                max_key = max(cauchy_dict, key=cauchy_dict.get)
                return evaluate_dist(max_key)

            np.random.seed(42)
            num_cities = 3
            distance_matrix = np.array([[0, 0.9, 0.8], [0.4, 0, 0.1],[0, 0.7, 0]])
            distance_matrix = distance_matrix.round(1)
            small_tsp = TSP(distance_matrix)
            initial_parameters = np.random.uniform(0, 1, 2)
            initial_state = small_tsp.prepare_initial_state([i for i in range(num_cities)])
            qaoa_function_of_layer(2, distance_matrix)

    Reference:
        1. S. Hadfield, Z. Wang, B. O'Gorman, E. G. Rieffel, D. Venturelli, R. Biswas,
        *From the Quantum Approximate Optimization Algorithm to a Quantum Alternating
        Operator Ansatz*.
        (`arxiv:1709.03489 <https://arxiv.org/abs/1709.03489>`__)
    """

    def __init__(self, distance_matrix, two_to_one=None, backend=None):
        self.backend = _check_backend(backend)

        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.two_to_one, _ = (
            _calculate_two_to_one(self.num_cities) if two_to_one is None else two_to_one
        )

    def hamiltonians(self):
        """
        Constructs the phaser and mixer Hamiltonians for the TSP.

        Returns:
            tuple: A tuple containing the phaser and mixer Hamiltonians.
        """
        return (
            _tsp_phaser(self.distance_matrix, backend=self.backend),
            _tsp_mixer(self.num_cities, backend=self.backend),
        )

    def prepare_initial_state(self, ordering):
        """
        Prepares a valid initial state for TSP QAOA based on the given city ordering.

        Args:
            ordering (list): A list representing the permutation of cities.

        Returns:
            array: The quantum state representing the initial solution.
        """
        n = len(ordering)
        c = Circuit(n**2)
        for i in range(n):
            c.add(gates.X(int(self.two_to_one[(ordering[i], i)])))
        result = self.backend.execute_circuit(c)
        return result.state()

    def penalty_method(self, penalty):
        """
        Constructs the TSP QUBO object using a penalty method for feasibility.

        Args:
            penalty (float): The penalty parameter for constraint violations.

        Returns:
            QUBO: A QUBO object for the TSP with penalties applied.
        """
        q_dict = {
            (
                self.two_to_one[(u, j)],
                self.two_to_one[(v, (j + 1) % self.num_cities)],
            ): self.distance_matrix[u, v]
            for u in range(self.num_cities)
            for v in range(self.num_cities)
            for j in range(self.num_cities)
            if v != u
        }
        qp = QUBO(0, q_dict)

        # row constraints
        for v in range(self.num_cities):
            row_constraint = [0 for _ in range(self.num_cities**2)]
            for j in range(self.num_cities):
                row_constraint[self.two_to_one[(v, j)]] = 1
            lp = LinearProblem(row_constraint, -1)
            tmp_qp = lp.square()
            tmp_qp = tmp_qp * penalty
            qp = qp + tmp_qp

        # column constraints
        for j in range(self.num_cities):
            col_constraint = [0 for _ in range(self.num_cities**2)]
            for v in range(self.num_cities):
                col_constraint[self.two_to_one[(v, j)]] = 1
            lp = LinearProblem(col_constraint, -1)
            tmp_qp = lp.square()
            tmp_qp = tmp_qp * penalty
            qp = qp + tmp_qp
        return qp


class MIS:
    """
    Class for representing the Maximal Independent Set (MIS) problem.

    The MIS problem involves selecting the largest subset of non-adjacent vertices in a graph.

    Args:
        g (networkx.Graph): A graph object representing the problem.

    Example:
        .. testcode::

            import networkx as nx
            from qiboopt.combinatorial.combinatorial import MIS

            g = nx.Graph()
            g.add_edges_from([(0, 1), (1, 2), (2, 0)])
            mis = MIS(g)
            penalty = 10
            qp = mis.penalty_method(penalty)
    """

    def __init__(self, g):
        """

        Args:
            g: A networkx object
        Returns:
            :class:`qiboopt.opt_class.opt_class.QUBO` representation
        """
        self.g = g
        self.n = g.number_of_nodes()

    def penalty_method(self, penalty):
        """
        Constructs the QUBO Hamiltonian for the MIS problem using a penalty method.

        Args:
            penalty (float): The penalty parameter for constraint violations.

        Returns:
            QUBO (:class:`qiboopt.opt_class.opt_class.QUBO`): A QUBO object for the
            Maximal Independent Set (MIS) problem.
        """
        q_dict = {(i, i): -1 for i in range(self.n)}
        q_dict = {**q_dict, **{(u, v): penalty for u, v in self.g.edges}}
        return QUBO(0, q_dict)

    def __str__(self):
        return self.__class__.__name__


def _ensure_weight_matrix(g_or_w):
    """
    Accepts either:
      - a symmetric numpy array (n x n) with zero diagonal, or
      - a networkx.Graph with optional 'weight' on edges (default 1.0).
    Returns:
      weight_matrix (np.ndarray, shape [n, n]), node_list (list): order of nodes if
      graph given.
    """
    if isinstance(g_or_w, np.ndarray):
        weight_matrix = np.array(g_or_w, dtype=float)
        if weight_matrix.shape[0] != weight_matrix.shape[1]:
            raise ValueError("Weight matrix must be square.")
        if not np.allclose(weight_matrix, weight_matrix.T, atol=1e-12):
            raise ValueError("Weight matrix must be symmetric for Max-Cut.")
        np.fill_diagonal(weight_matrix, 0.0)
        node_list = list(range(weight_matrix.shape[0]))
        return weight_matrix, node_list

    if isinstance(g_or_w, nx.Graph):
        nodes = list(g_or_w.nodes())
        n = len(nodes)
        idx = {u: i for i, u in enumerate(nodes)}
        weight_matrix = np.zeros((n, n), dtype=float)
        for u, v, data in g_or_w.edges(data=True):
            w = float(data.get("weight", 1.0))
            i, j = idx[u], idx[v]
            if i == j:
                continue
            weight_matrix[i, j] = weight_matrix[j, i] = w
        return weight_matrix, nodes

    raise TypeError("Expected a numpy array or a networkx.Graph.")


def _edge_list_from_weight_matrix(weight_matrix, tol=1e-12):
    """Return list of (i,j) with i<j where |W_ij|>tol."""
    n = weight_matrix.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(weight_matrix[i, j]) > tol:
                edges.append((i, j))
    return edges


# Hamiltonian Builders


def _maxcut_phaser(weight_matrix, backend=None, drop_constant=True):
    """
    Phase-separator for Max-Cut.

    We use the convention where the classical optimizer MINIMIZES the expectation
    value of the Hamiltonian. To align this with MAXIMIZING the cut, we drop the
    constant and use +0.5 * w_ij * Z_i Z_j so that minimizing energy prefers
    anti-aligned spins on edges (Z_i Z_j -> -1), i.e. larger cuts.

      Original:  H = sum_{i<j} w_ij * (1 - Z_i Z_j)/2
      Dropping constant => proportional to (+0.5) * sum w_ij * Z_i Z_j
    """
    n = weight_matrix.shape[0]
    form = 0
    constant = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w = weight_matrix[i, j]
            if w == 0:
                continue
            # keep only the ZZ term by default (constant dropped)
            form += (0.5 * w) * Z(i) * Z(j)
            if not drop_constant:
                constant += 0.5 * w
            # if you ever want the constant term: add (+0.5*w) to a running scalar
            # but SymbolicHamiltonian doesn't need it for optimization/QAOA.
    if not drop_constant and constant:
        form += constant
    return SymbolicHamiltonian(form, backend=backend)


def _maxcut_mixer(n, mode="x", edges=None, backend=None):
    """
    QAOA mixer for Max-Cut.

    Options:
      - mode='x':  transverse-field mixer H_M = sum_i X_i (default)
      - mode='xy': edge-based XY mixer H_M = sum_{(i,j) in E} (X_i X_j + Y_i Y_j)
                    Useful on sparse graphs; requires 'edges' list.
    """
    mode = (mode or "x").lower()
    form = 0
    if mode == "x":
        for i in range(n):
            form += X(i)
        return SymbolicHamiltonian(form, backend=backend)

    if mode == "xy":
        if not edges:
            # Fallback: complete graph if edges not provided
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for i, j in edges:
            form += X(i) * X(j) + Y(i) * Y(j)
        return SymbolicHamiltonian(form, backend=backend)

    raise ValueError("mixer must be one of {'x','xy'}.")


def _normalize(weight_matrix, mode):
    """
    Normalizes W and returns (W_scaled, scale_factor c).

    mode:
      - 'none':   no scaling (c = 1).
                Use when you care about true weights and you're tuning QAOA angles per-instance.

      - 'maxdeg': divide by c = max_i sum_j |W_ij|.
                Good default for batches of graphs with varying degrees/weight
                keeps each qubit's local scale ~O(1), so one γ-range works across instances.

      - 'sum':    divide by c = sum_{i<j} |W_ij|.
                Use for *cross-size/density comparisons* (energy per unit weight)
                or to improve numeric conditioning when weights are very large.

    Note: Scaling by positive c does NOT change the argmax cut. It only rescales energies
    (and thus the effective QAOA angle γ).
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return weight_matrix.copy(), 1.0

    if mode == "maxdeg":
        c = np.max(np.sum(np.abs(weight_matrix), axis=1))
        if c == 0:
            c = 1.0
        return weight_matrix / c, c

    if mode == "sum":
        c = np.sum(np.abs(np.triu(weight_matrix, 1)))
        if c == 0:
            c = 1.0
        return weight_matrix / c, c

    raise ValueError("normalize must be one of {'none','maxdeg','sum'}.")


class MaxCut:  # pylint: disable=too-many-instance-attributes
    """
    Max-Cut problem (unconstrained) with QUBO and QAOA-ready Hamiltonians.

    Args:
        graph_or_weights: either a symmetric (n x n) numpy array of weights
                          or a networkx.Graph with 'weight' on edges (default 1.0).
        backend: qibo backend; if None uses global backend.
        normalize: {'none','maxdeg','sum'}
            - 'none'   : keep true weights; best when reporting real cut values or tuning
                         γ per instance (single-instance studies).
            - 'maxdeg' : good default for *batches* with varying degrees/weights; keeps
                         local scales O(1) so γ-ranges transfer better across instances.
            - 'sum'    : good for cross-size/density *benchmarking* (energy per unit
                         total weight) or very large weights (conditioning).
        mixer: {'x','xy'}
            - 'x'  : standard transverse-field mixer (cheap, robust default).
            - 'xy' : pairwise XY on edges; can help exploration on sparse graphs
                     (costlier; deeper two-qubit circuits).

    Example:
        .. testcode::

            import numpy as np
            from qibo.models import QAOA

            W = np.array([
                [0, 0.8, 0.8, 0.8, 0.8],
                [0.8, 0, 0.8, 0.8, 0.8],
                [0.8, 0.8, 0, 0.2, 0.2],
                [0.8, 0.8, 0.2, 0, 0.2],
                [0.8, 0.8, 0.2, 0.2, 0]
            ])

            mc = MaxCut(W)
            obj_h, mix_h = mc.hamiltonians()

            # QAOA (start from |+>^n)
            init_state = mc.prepare_initial_state(init="plus")

            qaoa = QAOA(obj_h, mixer=mix_h)
            best_energy, params, _ = qaoa.minimize(initial_p=[0.1]*2,
                                                   initial_state=init_state,
                                                   method='BFGS')

            # Optional: QUBO coefficients for classical solvers
            qubo = mc.to_qubo()
            # qubo.q_dict has (i,j) -> coefficient, including diagonals for linear terms.
    """

    def __init__(self, graph_or_weights, backend=None, normalize="none", mixer="x"):
        self.backend = _check_backend(backend)
        # pylint: disable=invalid-name
        self.W_raw, self.nodes = _ensure_weight_matrix(graph_or_weights)
        self.n = self.W_raw.shape[0]

        # normalization
        self.W, self.energy_scale = _normalize(self.W_raw, normalize)
        # pylint: enable=invalid-name
        self.normalize = normalize

        # mixer choice
        self.mixer_mode = mixer
        self._edges = _edge_list_from_weight_matrix(self.W)  # for 'xy' mixer

        # index <-> node mapping
        self.ind_to_node = dict(enumerate(self.nodes))
        self.node_to_ind = {u: i for i, u in enumerate(self.nodes)}

    # Hamiltonians
    def hamiltonians(self):
        """
        Constructs the phaser and mixer Hamiltonians for the Max-Cut problem.

        Returns:
            tuple: A tuple containing the phaser and mixer Hamiltonians.
        """
        return (
            _maxcut_phaser(self.W, backend=self.backend),
            _maxcut_mixer(
                self.n, mode=self.mixer_mode, edges=self._edges, backend=self.backend
            ),
        )

    # Initial states
    def prepare_initial_state(self, init="plus", bitstring=None):
        """
        Prepare an initial state for QAOA.
        Options:
          - init="plus": |+>^n (Hadamard on all qubits).
          - init="bitstring": computational basis per 'bitstring' (list/str of 0/1).
          - init="random": random computational basis (uniform over {0,1}^n).

        Returns:
            np.ndarray: state vector from backend.execute_circuit(c)
        """
        c = Circuit(self.n)
        if init == "plus":
            for i in range(self.n):
                c.add(gates.H(i))
        elif init == "bitstring":
            if bitstring is None:
                raise ValueError("Provide 'bitstring' when init='bitstring'.")
            bits = [
                int(b)
                for b in (
                    bitstring.strip() if isinstance(bitstring, str) else bitstring
                )
            ]
            if len(bits) != self.n:
                raise ValueError("Bitstring length must equal number of vertices.")
            for i, b in enumerate(bits):
                if b == 1:
                    c.add(gates.X(i))
        elif init == "random":
            rng = np.random.default_rng()
            for i, b in enumerate(rng.integers(0, 2, size=self.n)):
                if b == 1:
                    c.add(gates.X(i))
        else:
            raise ValueError("init must be one of {'plus','bitstring','random'}.")

        result = self.backend.execute_circuit(c)
        return result.state()

    # QUBO
    def to_qubo(self, maximize=True, use_scaled=False):
        """
        QUBO of the cut objective using binary variables x_i ∈ {0,1}:
          Cut(x) = Σ_{i<j} W_ij [x_i XOR x_j] = Σ W_ij (x_i + x_j - 2 x_i x_j).

        Coefficients:
          for each i<j:
            q_{ii} += W_ij
            q_{jj} += W_ij
            q_{ij} += -2 W_ij

        Args:
          maximize (bool): if True (default), flip sign so standard *minimizing*
            QUBO solvers minimize -Cut. Set False if your solver maximizes.
          use_scaled (bool): if True, build QUBO with the (normalized) W; if False
            (default) use the original W so energies are in true units.

        When to use:
          * use_scaled=True if you want numerical conditioning consistent with the
            Hamiltonian used in QAOA (e.g., for hybrid workflows).
          * use_scaled=False if you want the solver’s objective in *original units*.
        """
        weight_matrix = self.W if use_scaled else self.W_raw
        q = {}
        n = self.n
        s = -1.0 if maximize else 1.0  # flip sign for minimization solvers
        for i in range(n):
            for j in range(i + 1, n):
                w = weight_matrix[i, j]
                if w == 0:
                    continue
                q[(i, i)] = q.get((i, i), 0.0) + s * w
                q[(j, j)] = q.get((j, j), 0.0) + s * w
                q[(i, j)] = q.get((i, j), 0.0) + s * (-2.0 * w)
        return QUBO(0.0, q)

    # Utilities
    def cut_value(self, bits, use_scaled=False):
        """
        Evaluate the cut weight of a bit assignment.

        Args:
          bits: list/array/str of 0/1 of length n (1 means S', 0 means S).
          use_scaled: if True, evaluate under normalized W (for comparisons to
                      scaled Hamiltonian energies). Otherwise (default), return
                      the *true* cut weight.

        Returns:
          float: sum_{i<j} W_ij * [bits_i XOR bits_j]
        """
        x = np.array(
            [int(b) for b in (bits.strip() if isinstance(bits, str) else bits)],
            dtype=int,
        )
        if x.size != self.n:
            raise ValueError("Assignment length must equal number of vertices.")
        weight_matrix = self.W if use_scaled else self.W_raw
        val = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if weight_matrix[i, j] == 0:
                    continue
                val += weight_matrix[i, j] * (x[i] ^ x[j])
        return float(val)

    def partition_from_bits(self, bits):
        """
        Convert a bitstring to sets S (bit 0) and S' (bit 1) with original node labels.
        """
        x = [int(b) for b in (bits.strip() if isinstance(bits, str) else bits)]
        if len(x) != self.n:
            raise ValueError("Assignment length must equal number of vertices.")
        left_set = {self.ind_to_node[i] for i, b in enumerate(x) if b == 0}
        right_set = {self.ind_to_node[i] for i, b in enumerate(x) if b == 1}
        return left_set, right_set

    # Energy rescaling
    def rescale_energy(self, E_scaled):  # pylint: disable=invalid-name
        """
        Convert an energy/expectation computed with the *scaled* Hamiltonian
        back to original units: E_true = energy_scale * E_scaled.
        """
        return self.energy_scale * E_scaled

    def __str__(self):
        return (
            f"{self.__class__.__name__}(n={self.n}, normalize='{self.normalize}', "
            f"mixer='{self.mixer_mode}')"
        )


class CombinatorialQAOA:
    """
    Lightweight helper around :class:`qibo.models.QAOA` for the problems defined in
    this module (or any object exposing ``hamiltonians()`` and, optionally,
    ``prepare_initial_state``).

    For multi-angle or linear-ramp parametrizations use
    :class:`CombinatorialMAQAOA` or :class:`CombinatorialLRQAOA`.

    Args:
        problem: Problem instance (e.g., :class:`TSP`, :class:`MaxCut`) that provides
            the Hamiltonians and, if needed, an initial-state factory.
        layers (int): Number of QAOA layers (controls the default initial parameter
            vector length). Must be positive. [Default 1]
        cost_hamiltonian: Optional override of the cost Hamiltonian returned by the
            problem. [Default None]
        mixer_hamiltonian: Optional override of the mixer Hamiltonian. [Default None]
        qaoa_kwargs (dict): Extra keyword arguments passed directly to
            :class:`qibo.models.QAOA`. [Default None]

    Example:
        >>> mc = MaxCut(W)
        >>> helper = CombinatorialQAOA(mc, layers=2)
        >>> energy, params, _ = helper.minimize(state_kwargs={"init": "plus"})
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        problem,
        layers=1,
        *,
        cost_hamiltonian=None,
        mixer_hamiltonian=None,
        qaoa_kwargs=None,
    ):
        """
        Initialize a QAOA helper for combinatorial problems.

        Args:
            problem (object): Problem exposing ``hamiltonians()`` and optionally
                ``prepare_initial_state``.
            layers (int): Number of alternating cost/mixer layers.
            cost_hamiltonian (qibo.hamiltonians.SymbolicHamiltonian | None): Cost
                Hamiltonian override; defaults to the problem-provided one.
            mixer_hamiltonian (qibo.hamiltonians.SymbolicHamiltonian | None): Mixer
                Hamiltonian override; defaults to the problem-provided one.
            qaoa_kwargs (dict | None): Extra keyword arguments forwarded to
                :class:`qibo.models.QAOA`.
        """
        if layers <= 0:
            raise ValueError("layers must be a positive integer.")
        self.problem = problem
        self.layers = int(layers)
        qaoa_kwargs = qaoa_kwargs or {}

        if cost_hamiltonian is None or mixer_hamiltonian is None:
            cost_from_problem, mixer_from_problem = self.problem.hamiltonians()
            cost_hamiltonian = cost_hamiltonian or cost_from_problem
            mixer_hamiltonian = mixer_hamiltonian or mixer_from_problem

        self.cost_hamiltonian = cost_hamiltonian
        self.mixer_hamiltonian = mixer_hamiltonian
        self._qaoa = QAOA(
            self.cost_hamiltonian, mixer=self.mixer_hamiltonian, **qaoa_kwargs
        )

    @property
    def model(self):
        """
        Expose the underlying :class:`qibo.models.QAOA` instance.

        Returns:
            qibo.models.QAOA: Configured model instance.
        """
        return self._qaoa

    def _prepare_state(self, state_kwargs=None):
        """
        Prepare an initial state using the problem's factory.

        Args:
            state_kwargs (dict | None): Keyword arguments forwarded to the problem's
                ``prepare_initial_state`` method.

        Returns:
            np.ndarray: Prepared initial state.
        """
        if not hasattr(self.problem, "prepare_initial_state"):
            raise ValueError(
                "Problem does not define 'prepare_initial_state'; "
                "please supply 'initial_state' explicitly."
            )
        kwargs = state_kwargs or {}
        return self.problem.prepare_initial_state(**kwargs)

    def _default_angles(self):
        """
        Build a default parameter vector for optimization.

        Returns:
            list[float]: Default angles sized to ``layers``.
        """
        return [0.1] * self.layers

    def minimize(
        self,
        *,
        initial_angles=None,
        initial_state=None,
        state_kwargs=None,
        method="BFGS",
        **minimize_kwargs,
    ):
        """
        Optimize the QAOA parameters for the provided problem.

        Args:
            initial_angles (list[float] | np.ndarray | None): Initial parameter vector.
            initial_state (np.ndarray | None): Optional state vector to start the
                algorithm.
            state_kwargs (dict | None): Keyword arguments forwarded to the problem's
                ``prepare_initial_state`` when ``initial_state`` is not supplied.
            method (str): Classical optimizer passed to :func:`QAOA.minimize`.
            **minimize_kwargs: Additional keyword arguments forwarded to the
                optimizer.

        Returns:
            tuple: ``(best_energy, parameters, info)`` matching the optimizer output.
        """
        angles = initial_angles if initial_angles is not None else self._default_angles()
        call_kwargs = dict(minimize_kwargs)

        if "initial_state" not in call_kwargs:
            if initial_state is not None:
                call_kwargs["initial_state"] = initial_state
            else:
                call_kwargs["initial_state"] = self._prepare_state(state_kwargs)

        return self._qaoa.minimize(initial_p=angles, method=method, **call_kwargs)

    def execute(self, parameters=None, initial_state=None, state_kwargs=None):
        """
        Execute the QAOA circuit using stored or provided parameters.

        Args:
            parameters (iterable | None): Parameter vector to load before execution.
            initial_state (np.ndarray | None): State vector to evolve. If omitted, it
                is constructed via ``state_kwargs`` and the problem instance.
            state_kwargs (dict | None): Keyword arguments passed to the problem's
                ``prepare_initial_state`` when ``initial_state`` is omitted.

        Returns:
            np.ndarray: Final state prepared by QAOA.
        """
        if parameters is not None:
            self._qaoa.set_parameters(parameters)

        state = initial_state if initial_state is not None else self._prepare_state(
            state_kwargs
        )
        print(f"Parameters: {parameters}")
        print(f"State: {state}")
        return self._qaoa.execute(state)


class CombinatorialMAQAOA(CombinatorialQAOA):  # pylint: disable=too-many-instance-attributes
    """
    Multi-angle QAOA helper that assigns separate parameters to each operator block
    in a layer.

    Args:
        problem: Problem instance (e.g., :class:`TSP`, :class:`MaxCut`) that provides
            the Hamiltonians and, if needed, an initial-state factory.
        layers (int): Number of QAOA layers (controls the default initial parameter
            vector length). Must be positive. [Default 1]
        cost_hamiltonian: Optional override of the cost Hamiltonian returned by the
            problem. [Default None]
        mixer_hamiltonian: Optional override of the mixer Hamiltonian. [Default None]
        qaoa_kwargs (dict): Extra keyword arguments passed directly to
            :class:`qibo.models.QAOA`. [Default None]
        ma_cost_operators (list): Ordered cost Hamiltonians used inside MA layers.
            [Default None]
        ma_mixer_operators (list): Ordered mixer Hamiltonians used inside MA layers.
            [Default None]
        ma_parameter_layout (dict): Optional per-layer counts for MA angles.
            [Default None]
        ma_default_angle (float): Default initial value for every MA angle.
            [Default 0.1]
        ma_solver (str): Solver backend used for the MA operator exponentials.
            [Default "exp"]

    Example:
        >>> ma_helper = CombinatorialMAQAOA(
        ...     problem,
        ...     layers=2,
        ...     ma_cost_operators=[ham1, ham2],
        ...     ma_mixer_operators=[mixer],
        ... )
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        problem,
        layers=1,
        *,
        cost_hamiltonian=None,
        mixer_hamiltonian=None,
        qaoa_kwargs=None,
        ma_cost_operators=None,
        ma_mixer_operators=None,
        ma_parameter_layout=None,
        ma_default_angle=0.1,
        ma_solver="exp",
    ):
        """
        Initialize a multi-angle QAOA helper for combinatorial problems.

        Args:
            problem (object): Problem exposing ``hamiltonians()`` and optionally
                ``prepare_initial_state``.
            layers (int): Number of alternating cost/mixer layers.
            cost_hamiltonian (qibo.hamiltonians.SymbolicHamiltonian | None): Cost
                Hamiltonian override; defaults to the problem-provided one.
            mixer_hamiltonian (qibo.hamiltonians.SymbolicHamiltonian | None): Mixer
                Hamiltonian override; defaults to the problem-provided one.
            qaoa_kwargs (dict | None): Extra keyword arguments forwarded to
                :class:`qibo.models.QAOA`.
            ma_cost_operators (list[qibo.hamiltonians.SymbolicHamiltonian] | None):
                Ordered cost operators used in each multi-angle layer.
            ma_mixer_operators (list[qibo.hamiltonians.SymbolicHamiltonian] | None):
                Ordered mixer operators used in each multi-angle layer.
            ma_parameter_layout (dict | None): Optional per-layer operator counts with
                keys ``cost`` and ``mixer``.
            ma_default_angle (float): Default initial value used when angles are not
                provided.
            ma_solver (str): Solver backend name for exponentiating multi-angle
                operators.
        """
        super().__init__(
            problem,
            layers,
            cost_hamiltonian=cost_hamiltonian,
            mixer_hamiltonian=mixer_hamiltonian,
            qaoa_kwargs=qaoa_kwargs,
        )
        self.ma_default_angle = ma_default_angle
        self._ma_cached_parameters = None
        self._ma_cost_ops = None
        self._ma_mixer_ops = None
        self._ma_cost_solvers = None
        self._ma_mixer_solvers = None
        self._ma_layout = None
        self._ma_total_angles = 0
        self._init_ma_support(
            solver_name=ma_solver,
            cost_ops=ma_cost_operators,
            mixer_ops=ma_mixer_operators,
            layout_spec=ma_parameter_layout,
        )

    def _default_angles(self):
        """
        Build a default parameter vector for multi-angle optimization.

        Returns:
            list[float]: Default angles ordered by the multi-angle layout.
        """
        return self._ma_default_angles()

    def minimize(
        self,
        *,
        initial_angles=None,
        initial_state=None,
        state_kwargs=None,
        method="BFGS",
        **minimize_kwargs,
    ):
        """
        Optimize the QAOA parameters using the multi-angle optimizer.

        Args:
            initial_angles (list[float] | np.ndarray | dict | None): Initial parameter
                vector; accepts flattened angles or per-block dictionaries.
            initial_state (np.ndarray | None): Optional state vector to start the
                algorithm.
            state_kwargs (dict | None): Keyword arguments forwarded to the problem's
                ``prepare_initial_state`` when ``initial_state`` is not supplied.
            method (str): Classical optimizer passed to the multi-angle optimizer.
            **minimize_kwargs: Additional keyword arguments forwarded to the
                optimizer.

        Returns:
            tuple: ``(best_energy, parameters, info)`` matching the optimizer output.
        """
        return self._ma_minimize(
            initial_angles=initial_angles,
            initial_state=initial_state,
            state_kwargs=state_kwargs,
            method=method,
            **minimize_kwargs,
        )

    def execute(self, parameters=None, initial_state=None, state_kwargs=None):
        """
        Execute the QAOA circuit using the multi-angle schedule.

        Args:
            parameters (iterable | dict | None): Parameter vector to load before
                execution; accepts structured multi-angle inputs.
            initial_state (np.ndarray | None): State vector to evolve. If omitted, it
                is constructed via ``state_kwargs`` and the problem instance.
            state_kwargs (dict | None): Keyword arguments passed to the problem's
                ``prepare_initial_state`` when ``initial_state`` is omitted.

        Returns:
            np.ndarray: Final state prepared by QAOA.
        """
        flat_params = (
            self._ma_flatten_angles(parameters)
            if parameters is not None
            else self._ma_cached_parameters
        )
        if flat_params is None:
            flat_params = self._ma_default_angles()
        state = (
            initial_state if initial_state is not None else self._prepare_state(state_kwargs)
        )
        return self._ma_execute(flat_params, state)

    def _init_ma_support(self, *, solver_name, cost_ops, mixer_ops, layout_spec):
        """
        Initialize multi-angle operator lists, solvers, and layout.

        Args:
            solver_name (str): Solver backend name.
            cost_ops (list[qibo.hamiltonians.SymbolicHamiltonian] | None): Cost
                operators to include in each block.
            mixer_ops (list[qibo.hamiltonians.SymbolicHamiltonian] | None): Mixer
                operators to include in each block.
            layout_spec (dict | None): Optional per-layer counts for ``cost`` and
                ``mixer`` operators.
        """
        cost_sequence = cost_ops or [self.cost_hamiltonian]
        mixer_sequence = mixer_ops or [self.mixer_hamiltonian]
        self._ma_cost_ops = list(cost_sequence)
        self._ma_mixer_ops = list(mixer_sequence)
        self._ma_cost_solvers = [
            get_solver(solver_name, 1e-2, ham) for ham in self._ma_cost_ops
        ]
        self._ma_mixer_solvers = [
            get_solver(solver_name, 1e-2, ham) for ham in self._ma_mixer_ops
        ]
        self._ma_layout = self._ma_build_layout(layout_spec)
        self._ma_total_angles = sum(
            len(block["cost"]) + len(block["mixer"]) for block in self._ma_layout
        )
        if self._ma_total_angles == 0:
            raise ValueError("MA configuration must include at least one operator.")

    def _ma_build_layout(self, layout_spec):
        """
        Build the per-layer solver layout for multi-angle execution.

        Args:
            layout_spec (dict | None): Optional per-layer counts for ``cost`` and
                ``mixer`` operators.

        Returns:
            list[dict]: Layout with ``cost`` and ``mixer`` solver lists per layer.
        """
        cost_counts, mixer_counts = self._ma_expand_counts(layout_spec)
        layout = []
        for layer in range(self.layers):
            layout.append(
                {
                    "cost": self._ma_cost_solvers[: cost_counts[layer]],
                    "mixer": self._ma_mixer_solvers[: mixer_counts[layer]],
                }
            )
        return layout

    def _ma_expand_counts(self, layout_spec):
        """
        Resolve how many cost and mixer operators each layer should use.

        Args:
            layout_spec (dict | None): User-provided overrides for ``cost`` and
                ``mixer`` counts per layer.

        Returns:
            tuple[list[int], list[int]]: Per-layer counts for cost and mixer blocks.
        """
        default_cost = len(self._ma_cost_solvers)
        default_mixer = len(self._ma_mixer_solvers)
        if layout_spec is None:
            cost_counts = [default_cost] * self.layers
            mixer_counts = [default_mixer] * self.layers
        else:
            raw_cost = layout_spec.get("cost", default_cost)
            raw_mixer = layout_spec.get("mixer", default_mixer)
            cost_counts = self._ma_sanitize_counts(raw_cost, default_cost)
            mixer_counts = self._ma_sanitize_counts(raw_mixer, default_mixer)
        return cost_counts, mixer_counts

    def _ma_sanitize_counts(self, counts, max_available):
        """
        Validate and normalize per-layer operator counts.

        Args:
            counts (int | Iterable[int] | None): Requested counts per layer.
            max_available (int): Maximum allowed operators in the sequence.

        Returns:
            list[int]: Normalized counts per layer.
        """
        if counts is None:
            counts = max_available
        if isinstance(counts, int):
            expanded = [counts] * self.layers
        else:
            expanded = list(counts)
            if len(expanded) != self.layers:
                raise ValueError("MA layout length must equal number of layers.")
        for value in expanded:
            if not 0 <= value <= max_available:
                raise ValueError("MA layout entries must be within available operator range.")
        return expanded

    def _ma_default_angles(self):
        """
        Build the default flattened parameter vector for multi-angle mode.

        Returns:
            list[float]: Default angles ordered by the multi-angle layout.
        """
        defaults = []
        for block in self._ma_layout:
            defaults.extend([self.ma_default_angle] * len(block["cost"]))
            defaults.extend([self.ma_default_angle] * len(block["mixer"]))
        return defaults

    def _ma_flatten_angles(self, angles):
        """
        Convert structured multi-angle parameters into a flat list.

        Args:
            angles (dict | list | tuple | np.ndarray | None): Angle specification,
                accepting ``{"cost": [...], "mixer": [...]}``, a tuple/list pair, or an
                already-flat iterable.

        Returns:
            list[float]: Flattened parameter vector matching the layout order.
        """
        if angles is None:
            return self._ma_default_angles()
        if isinstance(angles, dict):
            cost_layers = angles.get("cost")
            mixer_layers = angles.get("mixer")
        elif isinstance(angles, (list, tuple)) and len(angles) == 2:
            cost_layers, mixer_layers = angles
        else:
            return list(angles)
        normalized_cost = self._ma_normalize_layer_angles(cost_layers, "cost")
        normalized_mixer = self._ma_normalize_layer_angles(mixer_layers, "mixer")
        flat = []
        for layer_index, _ in enumerate(self._ma_layout):
            flat.extend(normalized_cost[layer_index])
            flat.extend(normalized_mixer[layer_index])
        return flat

    def _ma_normalize_layer_angles(self, layers, kind):
        """
        Ensure per-layer angle blocks match the expected multi-angle layout.

        Args:
            layers (Iterable[Iterable[float]] | None): Layered angles for the given
                ``kind``.
            kind (str): Either ``"cost"`` or ``"mixer"``.

        Returns:
            list[list[float]]: Normalized angle blocks per layer.
        """
        target_lengths = [len(block[kind]) for block in self._ma_layout]
        if layers is None:
            return [[self.ma_default_angle] * count for count in target_lengths]
        layer_sequence = list(layers)
        normalized = []
        for idx, count in enumerate(target_lengths):
            layer_values = layer_sequence[idx] if len(layer_sequence) > idx else None
            if layer_values is None:
                normalized.append([self.ma_default_angle] * count)
            else:
                if len(layer_values) != count:
                    raise ValueError("MA angle blocks must match layout specification.")
                normalized.append([float(v) for v in layer_values])
        return normalized

    def _ma_execute(self, parameters, initial_state=None):
        """
        Apply the configured multi-angle schedule to evolve a state.

        Args:
            parameters (Iterable[float]): Flattened multi-angle parameter vector.
            initial_state (np.ndarray | None): Starting state; defaults to ``|+>`` if
                omitted.

        Returns:
            np.ndarray: Normalized state after applying all multi-angle layers.
        """
        backend = self.cost_hamiltonian.backend
        param_list = []
        for idx, p in enumerate(parameters):
            raw = self.cost_hamiltonian.backend.to_numpy(p)
            arr = np.asarray(raw)
            if arr.size != 1:
                raise ValueError(f"MA parameter at index {idx} is not scalar.")
            val = arr.item()
            if isinstance(val, complex):
                if not np.allclose(val.imag, 0.0, atol=1e-12):
                    raise ValueError(
                        f"MA parameter at index {idx} has non-zero imaginary component."
                    )
                val = val.real
            param_list.append(float(val))
        if len(param_list) != self._ma_total_angles:
            raise ValueError("Unexpected number of MA parameters.")
        if initial_state is None:
            state = backend.plus_state(self.cost_hamiltonian.nqubits)
        else:
            state = backend.cast(initial_state, copy=True)
        cursor = 0
        for block in self._ma_layout:
            for solver in block["cost"]:
                state = self._ma_apply_solver(solver, param_list[cursor], state)
                cursor += 1
            for solver in block["mixer"]:
                state = self._ma_apply_solver(solver, param_list[cursor], state)
                cursor += 1
        if cursor != len(param_list):
            raise ValueError("MA execution consumed unexpected number of parameters.")
        return self._qaoa.normalize_state(state)

    def _ma_apply_solver(self, solver, angle, state):
        """
        Apply a single solver to evolve the state by a given angle.

        Args:
            solver: Callable evolution operator produced by ``get_solver``.
            angle (float): Evolution duration to set on the solver.
            state (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Updated state after applying the solver.
        """
        solver.dt = angle
        new_state = solver(state)
        if self._qaoa.callbacks:
            new_state = self._qaoa.normalize_state(new_state)
            self._qaoa.calculate_callbacks(new_state)
        return new_state

    def _ma_minimize(  # pylint: disable=too-many-locals
        self,
        *,
        initial_angles,
        initial_state,
        state_kwargs,
        method,
        **minimize_kwargs,
    ):
        """
        Optimize parameters in multi-angle mode.

        Args:
            initial_angles (Iterable[float] | dict | None): Initial parameter
                specification.
            initial_state (np.ndarray | None): Starting state vector.
            state_kwargs (dict | None): Arguments forwarded to the state-preparation
                helper when ``initial_state`` is not provided.
            method (str): Optimizer name passed to the internal optimizer.
            **minimize_kwargs: Additional optimizer keyword arguments.

        Returns:
            tuple: ``(best_energy, parameters, extra)`` from the optimizer.
        """
        flat_initial = (
            self._ma_flatten_angles(initial_angles)
            if initial_angles is not None
            else self._ma_default_angles()
        )
        prepared_state = (
            initial_state
            if initial_state is not None
            else self._prepare_state(state_kwargs)
        )
        backend = self.cost_hamiltonian.backend
        minimize_options = dict(minimize_kwargs)
        loss_func = minimize_options.pop("loss_func", None)
        loss_func_param = minimize_options.pop("loss_func_param", {})
        initial_vector = backend.cast(np.array(flat_initial))

        def _loss(params, hamiltonian, state, user_loss, user_kwargs):
            return self._ma_loss(params, hamiltonian, state, user_loss, user_kwargs)

        if method == "sgd":

            def loss(p, h, s, lf, lfp):
                return _loss(backend.cast(p), h, s, lf, lfp)

        else:

            def loss(p, h, s, lf, lfp):
                return backend.to_numpy(_loss(p, h, s, lf, lfp))

        result, parameters, extra = self._qaoa.optimizers.optimize(
            loss,
            initial_vector,
            args=(self.cost_hamiltonian, prepared_state, loss_func, loss_func_param),
            method=method,
            **minimize_options,
            backend=self._qaoa.backend,
        )
        self._ma_cached_parameters = (
            [float(v) for v in parameters] if parameters is not None else None
        )
        return result, parameters, extra

    def _ma_loss(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, params, hamiltonian, base_state, user_loss, user_kwargs
    ):
        """
        Evaluate the loss for multi-angle optimization.

        Args:
            params (Iterable[float]): Flattened parameter vector.
            hamiltonian (qibo.hamiltonians.SymbolicHamiltonian): Cost Hamiltonian used
                for expectation evaluation.
            base_state (np.ndarray | None): Optional precomputed initial state.
            user_loss (callable | None): Optional custom loss function.
            user_kwargs (dict): Keyword arguments forwarded to the custom loss.

        Returns:
            float: Loss value in backend form.
        """
        backend = hamiltonian.backend
        if base_state is not None:
            state = backend.cast(base_state, copy=True)
        else:
            state = None
        evolved = self._ma_execute(params, state)
        if user_loss is None:
            return hamiltonian.expectation(evolved)
        filtered_kwargs = {
            key: user_kwargs[key]
            for key in user_kwargs
            if key in user_loss.__code__.co_varnames
        }
        loss_args = {**filtered_kwargs, "hamiltonian": hamiltonian, "state": evolved}
        return user_loss(**loss_args)


class CombinatorialLRQAOA(CombinatorialQAOA):
    """
    Linear-ramp QAOA helper with fixed, interleaved parameters per layer.

    The linear ramp schedule uses:
      beta_i = (1 - i / p) * delta_beta
      gamma_i = ((i + 1) / p) * delta_gamma
    with i = 0..p-1 and parameters ordered as
      [gamma_0, beta_0, gamma_1, beta_1, ...].

    Args:
        problem: Problem instance (e.g., :class:`TSP`, :class:`MaxCut`) that provides
            the Hamiltonians and, if needed, an initial-state factory.
        layers (int): Number of QAOA layers (p). Must be positive. [Default 1]
        delta_beta (float): Ramp scale for beta parameters. [Default 0.3]
        delta_gamma (float): Ramp scale for gamma parameters. [Default 0.6]
        cost_hamiltonian: Optional override of the cost Hamiltonian returned by the
            problem. [Default None]
        mixer_hamiltonian: Optional override of the mixer Hamiltonian. [Default None]
        qaoa_kwargs (dict): Extra keyword arguments passed directly to
            :class:`qibo.models.QAOA`. [Default None]
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        problem,
        layers=1,
        *,
        delta_beta=0.3,
        delta_gamma=0.6,
        cost_hamiltonian=None,
        mixer_hamiltonian=None,
        qaoa_kwargs=None,
    ):
        super().__init__(
            problem,
            layers,
            cost_hamiltonian=cost_hamiltonian,
            mixer_hamiltonian=mixer_hamiltonian,
            qaoa_kwargs=qaoa_kwargs,
        )
        self.delta_beta = float(delta_beta)
        self.delta_gamma = float(delta_gamma)

    def lr_schedule(self, delta_beta=None, delta_gamma=None):
        """
        Build the linear-ramp beta and gamma schedules.

        Args:
            delta_beta (float | None): Override for beta ramp scale.
            delta_gamma (float | None): Override for gamma ramp scale.

        Returns:
            tuple[list[float], list[float]]: (gammas, betas) for all layers.
        """
        db = self.delta_beta if delta_beta is None else float(delta_beta)
        dg = self.delta_gamma if delta_gamma is None else float(delta_gamma)
        p = self.layers
        betas = [(1.0 - (i / p)) * db for i in range(p)]
        gammas = [((i + 1) / p) * dg for i in range(p)]
        return gammas, betas

    def lr_parameters(self, delta_beta=None, delta_gamma=None):
        """
        Build interleaved parameters for the linear-ramp schedule.

        Args:
            delta_beta (float | None): Override for beta ramp scale.
            delta_gamma (float | None): Override for gamma ramp scale.

        Returns:
            list[float]: Interleaved parameter list
                [gamma_0, beta_0, gamma_1, beta_1, ...].
        """
        gammas, betas = self.lr_schedule(
            delta_beta=delta_beta, delta_gamma=delta_gamma
        )
        params = []
        for gamma, beta in zip(gammas, betas):
            params.extend([gamma, beta])
        return params

    def _default_angles(self):
        """
        Build the default parameter vector for linear-ramp execution.

        Returns:
            list[float]: Interleaved linear-ramp parameters.
        """
        return self.lr_parameters()

    def minimize(self, *args, **kwargs):
        """
        Disabled for LR-QAOA.

        Raises:
            NotImplementedError: LR-QAOA uses fixed parameters; use ``execute`` or
                ``sweep_deltas`` to select ramps.
        """
        raise NotImplementedError(
            "LR-QAOA uses a fixed linear-ramp schedule; use execute() or sweep_deltas()."
        )

    def execute(self, parameters=None, initial_state=None, state_kwargs=None):
        """
        Execute the QAOA circuit using the linear-ramp schedule by default.

        Args:
            parameters (iterable | None): Optional parameter vector to load before
                execution. When omitted, uses the LR interleaved schedule.
            initial_state (np.ndarray | None): State vector to evolve. If omitted, it
                is constructed via ``state_kwargs`` and the problem instance.
            state_kwargs (dict | None): Keyword arguments passed to the problem's
                ``prepare_initial_state`` when ``initial_state`` is omitted.

        Returns:
            np.ndarray: Final state prepared by QAOA.
        """
        params = parameters if parameters is not None else self.lr_parameters()
        return super().execute(
            parameters=params, initial_state=initial_state, state_kwargs=state_kwargs
        )

    def _coerce_score(self, value):
        raw = self.cost_hamiltonian.backend.to_numpy(value)
        arr = np.asarray(raw)
        if arr.size != 1:
            raise ValueError("Sweep score must be scalar.")
        val = arr.item()
        if isinstance(val, complex):
            if not np.allclose(val.imag, 0.0, atol=1e-12):
                raise ValueError("Sweep score has non-zero imaginary component.")
            val = val.real
        return float(val)

    def _sweep_values(self, values):
        if np.isscalar(values):
            return [float(values)]
        return [float(v) for v in values]

    def sweep_deltas(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        delta_betas,
        delta_gammas,
        *,
        initial_state=None,
        state_kwargs=None,
        score_fn=None,
        update_best=False,
    ):
        """
        Sweep delta_beta and delta_gamma over a grid and score each run.

        Args:
            delta_betas (Iterable[float] | float): Candidate beta ramp scales.
            delta_gammas (Iterable[float] | float): Candidate gamma ramp scales.
            initial_state (np.ndarray | None): Optional state vector to start from.
            state_kwargs (dict | None): Keyword arguments forwarded to
                ``prepare_initial_state`` if ``initial_state`` is omitted.
            score_fn (callable | None): Optional scoring function. If omitted,
                uses the cost Hamiltonian expectation value.
            update_best (bool): When True, update ``delta_beta`` and ``delta_gamma``
                with the best-performing pair.

        Returns:
            tuple: ``(best, results)`` where ``best`` is a dict with keys
            ``delta_beta``, ``delta_gamma``, ``score``, and ``parameters``, and
            ``results`` is a list of all evaluated entries.
        """
        betas = self._sweep_values(delta_betas)
        gammas = self._sweep_values(delta_gammas)
        if not betas or not gammas:
            raise ValueError("Sweep ranges must be non-empty.")

        if initial_state is None:
            base_state = self._prepare_state(state_kwargs)
        else:
            base_state = initial_state

        backend = self.cost_hamiltonian.backend
        results = []
        best = None
        for db in betas:
            for dg in gammas:
                params = self.lr_parameters(delta_beta=db, delta_gamma=dg)
                state = backend.cast(base_state, copy=True)
                self._qaoa.set_parameters(params)
                evolved = self._qaoa.execute(state)
                if score_fn is None:
                    score = self.cost_hamiltonian.expectation(evolved)
                else:
                    score_kwargs = {
                        "hamiltonian": self.cost_hamiltonian,
                        "state": evolved,
                        "delta_beta": db,
                        "delta_gamma": dg,
                    }
                    filtered = {
                        key: value
                        for key, value in score_kwargs.items()
                        if key in score_fn.__code__.co_varnames
                    }
                    score = score_fn(**filtered)
                score_val = self._coerce_score(score)
                entry = {
                    "delta_beta": db,
                    "delta_gamma": dg,
                    "score": score_val,
                    "parameters": params,
                }
                results.append(entry)
                if best is None or entry["score"] < best["score"]:
                    best = entry

        if update_best and best is not None:
            self.delta_beta = best["delta_beta"]
            self.delta_gamma = best["delta_gamma"]

        return best, results
