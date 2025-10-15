"""
Various combinatorial optimisation applications that are commonly formulated as QUBO problems.
"""

import numpy as np
from qibo import gates
from qibo.backends import _check_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.circuit import Circuit
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
            from qibo import Circuit, gates
            from qibo.models import QAOA
            from qibo.result import CircuitResult


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

        The TSP is formulated as:

        .. math::

            \\min \\sum_{u,v,j} d_{u,v} \\, x_{u,j} \\, x_{v,j+1}

        Subject to constraints:

        .. math::

            \\sum_j x_{v,j} = 1 \\quad \\forall v \\
            \\sum_v x_{v,j} = 1 \\quad \\forall j

        The penalty method converts this to an unconstrained QUBO:

        .. math::

            f(x) = \\text{objective}(x) + \\lambda
            \\left[
                \\sum_v \\left(\\sum_j x_{v,j} - 1\\right)^2 +
                \\sum_j \\left(\\sum_v x_{v,j} - 1\\right)^2
            \\right]

        Args:
            penalty (float): The penalty parameter for constraint violations. It should be large enough to enforce
                             constraints but not so large as to cause numerical issues.

        Returns:
            QUBO: A QUBO object for the TSP with penalties applied.
        Raises:
            ValueError: If penalty is negative.
        """
        if penalty < 0:
            raise ValueError(f"Penalty must be positive, got {penalty}")
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
    

import numpy as np
import networkx as nx
from qibo import gates
from qibo.backends import _check_backend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models.circuit import Circuit
from qibo.symbols import X, Z

from qiboopt.opt_class.opt_class import QUBO


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


def _maxcut_phaser(weight_matrix, backend=None, drop_constant=True):
    """
    Builds the Max-Cut objective (phase-separator) Hamiltonian:
      H_C = sum_{i<j} w_ij * (1 - Z_i Z_j)/2
    Constants can be dropped since they only shift energy.
    """
    n = weight_matrix.shape[0]
    form = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weight_matrix[i, j]
            if w == 0:
                continue
            # keep only the ZZ term by default (constant dropped)
            form += (-0.5 * w) * Z(i) * Z(j)
            # if you ever want the constant term: add (+0.5*w) to a running scalar
            # but SymbolicHamiltonian doesn't need it for optimization/QAOA.
    return SymbolicHamiltonian(form, backend=backend)


def _maxcut_mixer(n, backend=None):
    """
    Standard (unconstrained) QAOA mixer for Max-Cut:
      H_M = sum_i X_i
    """
    form = 0
    for i in range(n):
        form += X(i)
    return SymbolicHamiltonian(form, backend=backend)


class MaxCut:
    """
    Max-Cut problem (unconstrained) with QUBO and QAOA-ready Hamiltonians.

    Args:
        graph_or_weights: either a symmetric (n x n) numpy array of weights
                          or a networkx.Graph with 'weight' on edges (default 1.0).
        backend: qibo backend; if None uses global backend.

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

    def __init__(self, graph_or_weights, backend=None):
        self.backend = _check_backend(backend)
        self.W, self.nodes = _ensure_weight_matrix(graph_or_weights)
        self.n = self.W.shape[0]
        # index <-> node mapping (useful if a networkx graph was provided)
        self.ind_to_node = {i: u for i, u in enumerate(self.nodes)}
        self.node_to_ind = {u: i for i, u in enumerate(self.nodes)}

    # ---------- Hamiltonians (for QAOA) ----------
    def hamiltonians(self):
        """
        Returns:
            (SymbolicHamiltonian, SymbolicHamiltonian):
                (phase-separator, mixer)
        """
        return (
            _maxcut_phaser(self.W, backend=self.backend),
            _maxcut_mixer(self.n, backend=self.backend),
        )

    # ---------- Initial states ----------
    def prepare_initial_state(self, init="plus", bitstring=None):
        """
        Prepare an initial state for QAOA.
        Options:
          - init="plus": |+>^n (Hadamard on all qubits), the standard QAOA start.
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
            if isinstance(bitstring, str):
                bits = [int(b) for b in bitstring.strip()]
            else:
                bits = list(map(int, bitstring))
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

    # ---------- QUBO ----------
    def to_qubo(self, maximize=True):
        """
        Build the QUBO representation:
            Cut(x) = sum_{i<j} w_ij [x_i XOR x_j]
                   = sum_{i<j} w_ij (x_i + x_j - 2 x_i x_j)

        In QUBO form (using diagonals for linear terms, since x_i^2 = x_i):
            For each i<j:
              q_{ii} += w_ij
              q_{jj} += w_ij
              q_{ij} += -2 w_ij

        Max-Cut is a maximization. Most QUBO solvers *minimize*, so by default we
        negate the objective (maximize=True -> minimize -Cut). If you need the
        maximizing form, pass maximize=False to leave signs as-is.

        Returns:
            QUBO: qiboopt QUBO object with q_dict filled.
        """
        q = {}
        n = self.n
        for i in range(n):
            for j in range(i + 1, n):
                w = self.W[i, j]
                if w == 0:
                    continue
                lin = w
                quad = -2.0 * w
                # sign flip if we want to minimize (standard) instead of maximize
                s = -1.0 if maximize else 1.0
                # diagonals accumulate linear terms
                q[(i, i)] = q.get((i, i), 0.0) + s * lin
                q[(j, j)] = q.get((j, j), 0.0) + s * lin
                # off-diagonal quadratic term
                q[(i, j)] = q.get((i, j), 0.0) + s * quad

        return QUBO(0.0, q)

    # ---------- Utilities ----------
    def cut_value(self, bits):
        """
        Evaluate cut weight for a bit assignment.
        Args:
            bits: list/array/str of 0/1 of length n (1 means S', 0 means S).
        Returns:
            float: sum_{i<j} w_ij * [bits_i XOR bits_j]
        """
        if isinstance(bits, str):
            x = np.array([int(b) for b in bits.strip()], dtype=int)
        else:
            x = np.array(bits, dtype=int)
        if x.size != self.n:
            raise ValueError("Assignment length must equal number of vertices.")

        val = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.W[i, j] == 0:
                    continue
                val += self.W[i, j] * (x[i] ^ x[j])
        return float(val)

    def partition_from_bits(self, bits):
        """
        Convert a bitstring to sets S, S' with original node labels (if graph input).
        Returns:
            (set, set): S (bit 0), S' (bit 1)
        """
        if isinstance(bits, str):
            x = [int(b) for b in bits.strip()]
        else:
            x = list(map(int, bits))
        if len(x) != self.n:
            raise ValueError("Assignment length must equal number of vertices.")

        S = {self.ind_to_node[i] for i, b in enumerate(x) if b == 0}
        Sp = {self.ind_to_node[i] for i, b in enumerate(x) if b == 1}
        return S, Sp

    def __str__(self):
        return f"{self.__class__.__name__}(n={self.n})"
