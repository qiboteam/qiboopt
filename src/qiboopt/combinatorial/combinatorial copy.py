"""
Various combinatorial optimisation applications that are commonly formulated as QUBO problems.
"""

import networkx as nx
import numpy as np
from qibo import gates
from qibo.backends import _check_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models import QAOA
from qibo.models.circuit import Circuit
from qibo.solvers import get_solver  # MA: solvers for multi-angle layer applications
from qibo.symbols import X, Y, Z

from qiboopt.opt_class.opt_class import (
    QUBO,
    LinearProblem,
    variable_dict_to_ind_dict,
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
    Class representing the Travelling Salesman Problem (TSP). The implementation is based on the work by Hadfield.

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
        1. S. Hadfield, Z. Wang, B. O'Gorman, E. G. Rieffel, D. Venturelli, R. Biswas, *From the Quantum Approximate
        Optimization Algorithm to a Quantum Alternating Operator Ansatz*.
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
      W (np.ndarray, shape [n, n]), node_list (list): order of nodes if graph given.
    """
    if isinstance(g_or_w, np.ndarray):
        W = np.array(g_or_w, dtype=float)
        if W.shape[0] != W.shape[1]:
            raise ValueError("Weight matrix must be square.")
        if not np.allclose(W, W.T, atol=1e-12):
            raise ValueError("Weight matrix must be symmetric for Max-Cut.")
        np.fill_diagonal(W, 0.0)
        node_list = list(range(W.shape[0]))
        return W, node_list

    if isinstance(g_or_w, nx.Graph):
        nodes = list(g_or_w.nodes())
        n = len(nodes)
        idx = {u: i for i, u in enumerate(nodes)}
        W = np.zeros((n, n), dtype=float)
        for u, v, data in g_or_w.edges(data=True):
            w = float(data.get("weight", 1.0))
            i, j = idx[u], idx[v]
            if i == j:
                continue
            W[i, j] = W[j, i] = w
        return W, nodes

    raise TypeError("Expected a numpy array or a networkx.Graph.")


def _edge_list_from_W(W, tol=1e-12):
    """Return list of (i,j) with i<j where |W_ij|>tol."""
    n = W.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(W[i, j]) > tol:
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
    for i in range(n):
        for j in range(i + 1, n):
            w = weight_matrix[i, j]
            if w == 0:
                continue
            # keep only the ZZ term by default (constant dropped)
            form += (0.5 * w) * Z(i) * Z(j)
            # if you ever want the constant term: add (+0.5*w) to a running scalar
            # but SymbolicHamiltonian doesn't need it for optimization/QAOA.
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


def _normalize(W, mode):
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
        return W.copy(), 1.0

    if mode == "maxdeg":
        c = np.max(np.sum(np.abs(W), axis=1))
        if c == 0:
            c = 1.0
        return W / c, c

    if mode == "sum":
        c = np.sum(np.abs(np.triu(W, 1)))
        if c == 0:
            c = 1.0
        return W / c, c

    raise ValueError("normalize must be one of {'none','maxdeg','sum'}.")


class MaxCut:
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
        self.W_raw, self.nodes = _ensure_weight_matrix(graph_or_weights)
        self.n = self.W_raw.shape[0]

        # normalization
        self.W, self.energy_scale = _normalize(self.W_raw, normalize)
        self.normalize = normalize

        # mixer choice
        self.mixer_mode = mixer
        self._edges = _edge_list_from_W(self.W)  # for 'xy' mixer

        # index <-> node mapping
        self.ind_to_node = {i: u for i, u in enumerate(self.nodes)}
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
        Wq = self.W if use_scaled else self.W_raw
        q = {}
        n = self.n
        s = -1.0 if maximize else 1.0  # flip sign for minimization solvers
        for i in range(n):
            for j in range(i + 1, n):
                w = Wq[i, j]
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
        Wv = self.W if use_scaled else self.W_raw
        val = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if Wv[i, j] == 0:
                    continue
                val += Wv[i, j] * (x[i] ^ x[j])
        return float(val)

    def partition_from_bits(self, bits):
        """
        Convert a bitstring to sets S (bit 0) and S' (bit 1) with original node labels.
        """
        x = [int(b) for b in (bits.strip() if isinstance(bits, str) else bits)]
        if len(x) != self.n:
            raise ValueError("Assignment length must equal number of vertices.")
        S = {self.ind_to_node[i] for i, b in enumerate(x) if b == 0}
        Sp = {self.ind_to_node[i] for i, b in enumerate(x) if b == 1}
        return S, Sp

    # Energy rescaling
    def rescale_energy(self, E_scaled):
        """
        Convert an energy/expectation computed with the *scaled* Hamiltonian
        back to original units: E_true = energy_scale * E_scaled.
        """
        return self.energy_scale * E_scaled

    def __str__(self):
        return f"{self.__class__.__name__}(n={self.n}, normalize='{self.normalize}', mixer='{self.mixer_mode}')"


class CombinatorialQAOA:
    """
    Helper around :class:`qibo.models.QAOA` that stays problem-aware (it can call
    ``problem.prepare_initial_state``) and optionally supports multi-angle (MA)
    schedules. This version mirrors the key guards present in qibo's own QAOA:

    - validates cost/mixer type and qubit count
    - supplies a default transverse-field mixer when none is provided
    - enforces an even, 2p-parameter vector for standard QAOA paths
    - falls back to |+>^n when no initial-state factory is available
    - keeps callback/solver behaviour provided by the underlying qibo QAOA

    Args:
        problem: Problem instance (e.g., :class:`TSP`, :class:`MaxCut`) that provides
            Hamiltonians and, optionally, ``prepare_initial_state``.
        layers (int): Number of QAOA layers (p). Must be positive. [Default 1]
        cost_hamiltonian: Optional override of the cost Hamiltonian. [Default None]
        mixer_hamiltonian: Optional override of the mixer Hamiltonian. If ``None``,
            a transverse-field X mixer is built to match ``cost_hamiltonian``. [Default None]
        qaoa_kwargs (dict): Extra kwargs forwarded to :class:`qibo.models.QAOA`
            (e.g., ``solver``, ``callbacks``, ``accelerators``). [Default None]
        multi_angle (bool): Enable MA mode using dedicated cost/mixer operator lists. [Default False]
        ma_cost_operators (list): Ordered cost Hamiltonians used inside MA layers.
        ma_mixer_operators (list): Ordered mixer Hamiltonians used inside MA layers.
        ma_parameter_layout (dict): Optional per-layer counts for MA angles.
        ma_default_angle (float): Default initial value for every MA angle.
        ma_solver (str): Solver backend used for the MA operator exponentials.

    Example:
        >>> mc = MaxCut(W)
        >>> helper = CombinatorialQAOA(mc, layers=2)
        >>> energy, params, _ = helper.minimize(state_kwargs={"init": "plus"})
    """

    def __init__(
        self,
        problem,
        layers=1,
        *,
        cost_hamiltonian=None,
        mixer_hamiltonian=None,
        qaoa_kwargs=None,
        multi_angle=False,  # MA: flag to toggle multi-angle behavior
        ma_cost_operators=None,  # MA: optional per-angle cost Hamiltonians
        ma_mixer_operators=None,  # MA: optional per-angle mixer Hamiltonians
        ma_parameter_layout=None,  # MA: per-layer angle counts configuration
        ma_default_angle=0.1,  # MA: default initialization value for MA angles
        ma_solver="exp",  # MA: solver used for MA operator exponentials
    ):
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
        self.mixer_hamiltonian = (
            mixer_hamiltonian or self._build_default_mixer(self.cost_hamiltonian)
        )
        self._validate_hamiltonians()

        self._qaoa = QAOA(
            self.cost_hamiltonian, mixer=self.mixer_hamiltonian, **qaoa_kwargs
        )
        # Keep direct access to the backend for defaults / plus-state fallbacks.
        self.backend = self.cost_hamiltonian.backend
        self.multi_angle = bool(multi_angle)  # MA: remember if MA mode is enabled
        self.ma_default_angle = ma_default_angle  # MA: store default MA parameter value
        self._ma_cached_parameters = None  # MA: cache last MA parameter vector
        self._ma_cost_ops = None  # MA: placeholder for MA cost Hamiltonians
        self._ma_mixer_ops = None  # MA: placeholder for MA mixer Hamiltonians
        self._ma_cost_solvers = None  # MA: placeholder for MA cost solvers
        self._ma_mixer_solvers = None  # MA: placeholder for MA mixer solvers
        self._ma_layout = None  # MA: per-layer MA operator schedule
        self._ma_total_angles = 0  # MA: track MA parameter vector length
        if self.multi_angle:  # MA: configure MA helpers only when requested
            self._init_ma_support(  # MA: bootstrap MA operator sequences and solvers
                solver_name=ma_solver,  # MA: pass solver choice to helper
                cost_ops=ma_cost_operators,  # MA: provide custom cost operators
                mixer_ops=ma_mixer_operators,  # MA: provide custom mixer operators
                layout_spec=ma_parameter_layout,  # MA: optional layout overrides
            )

    @property
    def model(self):
        """Expose the underlying :class:`qibo.models.QAOA` instance."""
        return self._qaoa

    def _build_default_mixer(self, cost_hamiltonian):
        """Create a transverse-field X mixer matching the cost Hamiltonian backend."""
        from qibo import hamiltonians as qham

        trotter = isinstance(cost_hamiltonian, SymbolicHamiltonian)
        return qham.X(
            cost_hamiltonian.nqubits,
            dense=not trotter,
            backend=cost_hamiltonian.backend,
        )

    def _validate_hamiltonians(self):
        """Ensure cost and mixer share type and qubit count, mirroring qibo.QAOA guards."""
        if type(self.mixer_hamiltonian) != type(self.cost_hamiltonian):
            raise TypeError(
                f"Given Hamiltonian is of type {type(self.cost_hamiltonian)} "
                f"while mixer is of type {type(self.mixer_hamiltonian)}."
            )
        if self.mixer_hamiltonian.nqubits != self.cost_hamiltonian.nqubits:
            raise ValueError(
                f"Given Hamiltonian acts on {self.cost_hamiltonian.nqubits} qubits "
                f"while mixer acts on {self.mixer_hamiltonian.nqubits}."
            )

    def _prepare_state(self, state_kwargs=None):
        """Return initial state from problem factory when available, else |+>^n."""
        if hasattr(self.problem, "prepare_initial_state"):
            kwargs = state_kwargs or {}
            return self.problem.prepare_initial_state(**kwargs)
        # Fallback to uniform superposition
        return self.backend.plus_state(self.cost_hamiltonian.nqubits)

    def _default_angles(self):
        if self.multi_angle:  # MA: switch to MA-specific initialization when enabled
            return self._ma_default_angles()  # MA: use MA layout-driven defaults
        return [0.1] * (2 * self.layers)  # standard QAOA needs 2p angles

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
            initial_angles (list/np.ndarray): Initial parameter vector for QAOA. If
                omitted, a constant vector of length ``layers`` filled with 0.1 is used.
            initial_state (np.ndarray): Optional state vector to start the algorithm.
            state_kwargs (dict): Keyword arguments forwarded to the problem's
                ``prepare_initial_state`` method when ``initial_state`` is not supplied.
            method (str): Classical optimizer passed to :func:`QAOA.minimize`.
            **minimize_kwargs: Additional keyword arguments forwarded to
                :func:`QAOA.minimize`.

        Returns:
            tuple: ``(best_energy, parameters, info)`` from :func:`QAOA.minimize`.
        """
        if self.multi_angle:  # MA: delegate optimization to MA-specific routine
            return self._ma_minimize(  # MA: run custom MA optimizer
                initial_angles=initial_angles,  # MA: forward MA parameters
                initial_state=initial_state,  # MA: forward MA initial state
                state_kwargs=state_kwargs,  # MA: forward MA state kwargs
                method=method,  # MA: optimizer choice for MA
                **minimize_kwargs,  # MA: pass remaining kwargs
            )

        angles = initial_angles if initial_angles is not None else self._default_angles()
        if len(angles) % 2 != 0:
            raise ValueError("QAOA parameter vector must have even length (γ, β pairs).")
        if len(angles) != 2 * self.layers:
            raise ValueError(
                f"Expected {2 * self.layers} parameters for {self.layers} layers, got {len(angles)}."
            )
        call_kwargs = dict(minimize_kwargs)

        if "initial_state" not in call_kwargs:
            if initial_state is not None:
                call_kwargs["initial_state"] = initial_state
            else:
                call_kwargs["initial_state"] = self._prepare_state(state_kwargs)

        return self._qaoa.minimize(initial_p=angles, method=method, **call_kwargs)

    def execute(self, parameters=None, initial_state=None, state_kwargs=None):
        """
        Execute the QAOA circuit using the stored (or provided) parameters.
        Used for validating functionality
        
        Args:
            parameters (iterable): Optional parameter vector to load before execution.
            initial_state (np.ndarray): State vector to evolve. If omitted, it is
                constructed via ``state_kwargs`` and the problem instance.
            state_kwargs (dict): Keyword arguments passed to the problem's
                ``prepare_initial_state`` (only used when ``initial_state`` is omitted).

        Returns:
            np.ndarray: Final state prepared by QAOA.
        """
        if self.multi_angle:  # MA: run dedicated MA execution path
            flat_params = (  # MA: ensure dict-like inputs become flat vectors
                self._ma_flatten_angles(parameters)
                if parameters is not None
                else self._ma_cached_parameters
            )  # MA: reuse cached MA params when none provided
            if flat_params is None:  # MA: fall back to default MA vector if needed
                flat_params = self._ma_default_angles()  # MA: default MA parameters
            state = (  # MA: prepare initial state consistently
                initial_state
                if initial_state is not None
                else self._prepare_state(state_kwargs)
            )  # MA: reuse single-shot state prep
            return self._ma_execute(flat_params, state)  # MA: execute MA layers

        if parameters is not None:
            if len(parameters) % 2 != 0:
                raise ValueError("QAOA parameter vector must have even length (γ, β pairs).")
            self._qaoa.set_parameters(parameters)

        state = initial_state if initial_state is not None else self._prepare_state(
            state_kwargs
        )
        if self._qaoa.params is None:
            # Provide a sensible default if execute is called before minimize/set_parameters.
            self._qaoa.set_parameters(self._default_angles())
        return self._qaoa.execute(state)

    def _init_ma_support(self, *, solver_name, cost_ops, mixer_ops, layout_spec):  # MA: initialize MA data
        cost_sequence = cost_ops or [self.cost_hamiltonian]  # MA: derive cost operator list
        mixer_sequence = mixer_ops or [self.mixer_hamiltonian]  # MA: derive mixer operator list
        self._ma_cost_ops = list(cost_sequence)  # MA: store cost operators deterministically
        self._ma_mixer_ops = list(mixer_sequence)  # MA: store mixer operators deterministically
        self._ma_cost_solvers = [  # MA: allocate solvers per cost operator
            get_solver(solver_name, 1e-2, ham) for ham in self._ma_cost_ops
        ]  # MA: instantiate cost solvers
        self._ma_mixer_solvers = [  # MA: allocate solvers per mixer operator
            get_solver(solver_name, 1e-2, ham) for ham in self._ma_mixer_ops
        ]  # MA: instantiate mixer solvers
        self._ma_layout = self._ma_build_layout(layout_spec)  # MA: build per-layer schedule
        self._ma_total_angles = sum(  # MA: record total number of MA parameters
            len(block["cost"]) + len(block["mixer"]) for block in self._ma_layout
        )  # MA: accumulate angles across layers
        if self._ma_total_angles == 0:  # MA: ensure MA configuration is meaningful
            raise ValueError("MA configuration must include at least one operator.")  # MA: guard empty MA setup

    def _ma_build_layout(self, layout_spec):  # MA: translate layout spec into solver schedule
        cost_counts, mixer_counts = self._ma_expand_counts(layout_spec)  # MA: compute per-layer counts
        layout = []  # MA: storage for computed layout
        for layer in range(self.layers):  # MA: iterate over layers to collect solvers
            layout.append(  # MA: append combined block for this layer
                {
                    "cost": self._ma_cost_solvers[: cost_counts[layer]],  # MA: select cost solvers for this layer
                    "mixer": self._ma_mixer_solvers[: mixer_counts[layer]],  # MA: select mixer solvers for this layer
                }
            )  # MA: store combined entry
        return layout  # MA: return constructed layout

    def _ma_expand_counts(self, layout_spec):  # MA: resolve layout counts for cost/mixer
        default_cost = len(self._ma_cost_solvers)  # MA: default uses all cost solvers
        default_mixer = len(self._ma_mixer_solvers)  # MA: default uses all mixer solvers
        if layout_spec is None:  # MA: no overrides provided
            cost_counts = [default_cost] * self.layers  # MA: use full cost set each layer
            mixer_counts = [default_mixer] * self.layers  # MA: use full mixer set each layer
        else:  # MA: interpret user-specified layout
            raw_cost = layout_spec.get("cost", default_cost)  # MA: extract cost override
            raw_mixer = layout_spec.get("mixer", default_mixer)  # MA: extract mixer override
            cost_counts = self._ma_sanitize_counts(raw_cost, default_cost)  # MA: normalize cost counts
            mixer_counts = self._ma_sanitize_counts(raw_mixer, default_mixer)  # MA: normalize mixer counts
        return cost_counts, mixer_counts  # MA: deliver resolved counts

    def _ma_sanitize_counts(self, counts, max_available):  # MA: clean up per-layer count spec
        if counts is None:  # MA: treat None as default
            counts = max_available  # MA: fallback to max operators
        if isinstance(counts, int):  # MA: scalar specification
            expanded = [counts] * self.layers  # MA: replicate value per layer
        else:  # MA: assume iterable specification
            expanded = list(counts)  # MA: coerce to list for inspection
            if len(expanded) != self.layers:  # MA: ensure length matches depth
                raise ValueError("MA layout length must equal number of layers.")  # MA: guard mismatched layout
        for value in expanded:  # MA: validate individual entries
            if not (0 <= value <= max_available):  # MA: enforce feasible counts
                raise ValueError("MA layout entries must be within available operator range.")  # MA: reject invalid counts
        return expanded  # MA: return normalized list

    def _ma_default_angles(self):  # MA: compute default flattened MA parameter vector
        defaults = []  # MA: container for default parameters
        for block in self._ma_layout:  # MA: iterate through layer schedule
            defaults.extend([self.ma_default_angle] * len(block["cost"]))  # MA: append default gammas
            defaults.extend([self.ma_default_angle] * len(block["mixer"]))  # MA: append default betas
        return defaults  # MA: deliver default vector

    def _ma_flatten_angles(self, angles):  # MA: coerce structured MA angles into flat list
        if angles is None:  # MA: use defaults when nothing provided
            return self._ma_default_angles()  # MA: fallback to default vector
        if isinstance(angles, dict):  # MA: dict with explicit cost/mixer entries
            cost_layers = angles.get("cost")  # MA: extract layerwise cost angles
            mixer_layers = angles.get("mixer")  # MA: extract layerwise mixer angles
        elif (
            isinstance(angles, (list, tuple))  # MA: support tuple/list forms
            and len(angles) == 2  # MA: expect pair (cost, mixer)
        ):
            cost_layers, mixer_layers = angles  # MA: unpack tuple/list pair
        else:  # MA: already flat
            return list(angles)  # MA: return flat copy unchanged
        normalized_cost = self._ma_normalize_layer_angles(cost_layers, "cost")  # MA: enforce layer shapes for cost angles
        normalized_mixer = self._ma_normalize_layer_angles(mixer_layers, "mixer")  # MA: enforce layer shapes for mixer angles
        flat = []  # MA: accumulator for flattened parameters
        for layer_index, block in enumerate(self._ma_layout):  # MA: iterate layers
            flat.extend(normalized_cost[layer_index])  # MA: append this layer's cost angles
            flat.extend(normalized_mixer[layer_index])  # MA: append this layer's mixer angles
        return flat  # MA: return flattened vector

    def _ma_normalize_layer_angles(self, layers, kind):  # MA: reshape user MA data per layer
        target_lengths = [len(block[kind]) for block in self._ma_layout]  # MA: compute required counts
        if layers is None:  # MA: allocate defaults when missing
            return [  # MA: build list of defaults per layer
                [self.ma_default_angle] * count for count in target_lengths  # MA: replicate default value
            ]  # MA: deliver generated defaults
        layer_sequence = list(layers)  # MA: force concrete indexing
        normalized = []  # MA: store normalized layers
        for idx, count in enumerate(target_lengths):  # MA: iterate expected layer sizes
            layer_values = layer_sequence[idx] if len(layer_sequence) > idx else None  # MA: pull user data or None
            if layer_values is None:  # MA: fill missing entries
                normalized.append([self.ma_default_angle] * count)  # MA: use defaults for missing layer
            else:  # MA: validate provided data
                if len(layer_values) != count:  # MA: ensure size matches expectation
                    raise ValueError("MA angle blocks must match layout specification.")  # MA: guard inconsistent data
                normalized.append([float(v) for v in layer_values])  # MA: coerce entries to floats
        return normalized  # MA: deliver normalized angles

    def _ma_execute(self, parameters, initial_state=None):  # MA: execute MA layered evolution
        backend = self.cost_hamiltonian.backend  # MA: cache backend for casting
        param_list = [float(p) for p in parameters]  # MA: convert parameters to floats
        if len(param_list) != self._ma_total_angles:  # MA: ensure parameter length matches layout
            raise ValueError("Unexpected number of MA parameters.")  # MA: report mismatched vector
        if initial_state is None:  # MA: default starting state
            state = backend.plus_state(self.cost_hamiltonian.nqubits)  # MA: |+>^n default
        else:  # MA: user-provided state
            state = backend.cast(initial_state, copy=True)  # MA: copy to avoid mutation
        cursor = 0  # MA: track current parameter index
        for block in self._ma_layout:  # MA: iterate cost/mixer sequences per layer
            for solver in block["cost"]:  # MA: apply cost solvers sequentially
                state = self._ma_apply_solver(solver, param_list[cursor], state)  # MA: evolve by cost operator
                cursor += 1  # MA: advance cursor after cost
            for solver in block["mixer"]:  # MA: apply mixer solvers sequentially
                state = self._ma_apply_solver(solver, param_list[cursor], state)  # MA: evolve by mixer operator
                cursor += 1  # MA: advance cursor after mixer
        if cursor != len(param_list):  # MA: confirm full parameter consumption
            raise ValueError("MA execution consumed unexpected number of parameters.")  # MA: guard against layout drift
        return self._qaoa.normalize_state(state)  # MA: normalize before returning

    def _ma_apply_solver(self, solver, angle, state):  # MA: helper to apply a solver with callbacks
        solver.dt = angle  # MA: set evolution duration for this gate
        new_state = solver(state)  # MA: apply Hamiltonian exponential
        if self._qaoa.callbacks:  # MA: reuse callback machinery
            new_state = self._qaoa.normalize_state(new_state)  # MA: normalize before callbacks
            self._qaoa.calculate_callbacks(new_state)  # MA: trigger callbacks on MA path
        return new_state  # MA: return evolved state

    def _ma_minimize(
        self,
        *,
        initial_angles,
        initial_state,
        state_kwargs,
        method,
        **minimize_kwargs,
    ):  # MA: optimize MA parameters
        flat_initial = (  # MA: ensure we have a starting vector
            self._ma_flatten_angles(initial_angles)
            if initial_angles is not None
            else self._ma_default_angles()
        )  # MA: fallback to defaults when absent
        prepared_state = (  # MA: reuse provided or freshly prepared state
            initial_state
            if initial_state is not None
            else self._prepare_state(state_kwargs)
        )  # MA: unify initial state handling
        backend = self.cost_hamiltonian.backend  # MA: backend shortcut
        minimize_options = dict(minimize_kwargs)  # MA: copy optimizer kwargs
        loss_func = minimize_options.pop("loss_func", None)  # MA: extract custom loss
        loss_func_param = minimize_options.pop("loss_func_param", {})  # MA: extract loss kwargs
        initial_vector = backend.cast(np.array(flat_initial))  # MA: cast initial vector to backend

        def _loss(params, helper, hamiltonian, state, user_loss, user_kwargs):  # MA: closure for optimizer
            return helper._ma_loss(params, hamiltonian, state, user_loss, user_kwargs)  # MA: delegate to helper

        if method == "sgd":  # MA: adapt loss for SGD optimizer expectations
            loss = (
                lambda p, helper, h, s, lf, lfp: _loss(
                    backend.cast(p), helper, h, s, lf, lfp
                )
            )  # MA: cast parameters before loss evaluation
        else:  # MA: non-SGD paths expect numpy floats
            loss = (
                lambda p, helper, h, s, lf, lfp: backend.to_numpy(
                    _loss(p, helper, h, s, lf, lfp)
                )
            )  # MA: convert backend tensors to numpy scalars

        result, parameters, extra = self._qaoa.optimizers.optimize(  # MA: run optimizer using qibo utilities
            loss,
            initial_vector,
            args=(self, self.cost_hamiltonian, prepared_state, loss_func, loss_func_param),  # MA: pass helper args
            method=method,
            **minimize_options,
            backend=self._qaoa.backend,
        )  # MA: reuse qibo optimizer interface
        self._ma_cached_parameters = (  # MA: cache optimized MA parameters as floats
            [float(v) for v in parameters] if parameters is not None else None
        )  # MA: detach from backend tensors
        return result, parameters, extra  # MA: mirror QAOA minimize signature

    def _ma_loss(self, params, hamiltonian, base_state, user_loss, user_kwargs):  # MA: evaluate MA loss
        backend = hamiltonian.backend  # MA: reuse backend for casting
        if base_state is not None:  # MA: copy provided state
            state = backend.cast(base_state, copy=True)  # MA: copy to avoid side effects
        else:  # MA: default to None
            state = None  # MA: indicate default |+>^n usage
        evolved = self._ma_execute(params, state)  # MA: run MA circuit
        if user_loss is None:  # MA: default to expectation value
            return hamiltonian.expectation(evolved)  # MA: compute standard energy
        filtered_kwargs = {  # MA: map allowed kwargs for custom loss
            key: user_kwargs[key]
            for key in user_kwargs
            if key in user_loss.__code__.co_varnames
        }  # MA: filter unsupported parameters
        loss_args = {**filtered_kwargs, "hamiltonian": hamiltonian, "state": evolved}  # MA: compose loss inputs
        return user_loss(**loss_args)  # MA: evaluate custom loss
