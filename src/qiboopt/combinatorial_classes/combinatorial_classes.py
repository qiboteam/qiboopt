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
    linear_problem,
)


def _calculate_two_to_one(num_cities):
    """
    Calculates a mapping from two coordinates to one coordinate for the TSP problem.

    Args:
        num_cities (int): The number of cities for the TSP.

    Returns:
        (np.ndarray): A 2D array mapping two coordinates to one.
    """
    return np.arange(num_cities**2).reshape(num_cities, num_cities)


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
    two_to_one = _calculate_two_to_one(num_cities)
    form = 0
    form = sum(
        distance_matrix[u, v]
        * Z(int(two_to_one[u, i]))
        * Z(int(two_to_one[v, (i + 1) % num_cities]))
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

    two_to_one = _calculate_two_to_one(num_cities)

    def splus(u, i):
        """
        Defines the S+ operator for a specific city and position.

        Args:
            u (int): City index.
            i (int): Position index.

        Returns:
            SymbolicHamiltonian: The S+ operator.
        """
        return X(int(two_to_one[u, i])) + 1j * Y(int(two_to_one[u, i]))

    def sminus(u, i):
        """
        Defines the S- operator for a specific city and position.

        Args:
            u (int): City index.
            i (int): Position index.

        Returns:
            SymbolicHamiltonian: The S- operator.
        """
        return X(int(two_to_one[u, i])) - 1j * Y(int(two_to_one[u, i]))

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
    # for i in range(num_cities):
    #     for u in range(num_cities):
    #         for v in range(num_cities):
    #             if u != v:
    #                 form += splus(u, i) * splus(v, (i + 1) % num_cities) * sminus(
    #                     u, (i + 1) % num_cities
    #                 ) * sminus(v, i) + sminus(u, i) * sminus(
    #                     v, (i + 1) % num_cities
    #                 ) * splus(
    #                     u, (i + 1) % num_cities
    #                 ) * splus(
    #                     v, i
    #                 )
    ham = SymbolicHamiltonian(form, backend=backend)
    return ham


class TSP:
    """
    Class representing the Travelling Salesman Problem (TSP). The implementation is based on the work by Hadfield.

    Args:
        distance_matrix: a numpy matrix encoding the distance matrix.
        backend: Backend to use for calculations. If not given the global backend will be used.

    Example:
        .. testcode::

            from qibo.models.tsp import TSP
            import numpy as np
            from collections import defaultdict
            from qibo import gates
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
        Optimization Algorithm to a Quantum Alternating Operator Ansatz*. (`arxiv:1709.03489 <https://arxiv.org/abs/1709.03489>`__)
    """

    def __init__(self, distance_matrix, backend=None):
        self.backend = _check_backend(backend)

        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.two_to_one = _calculate_two_to_one(self.num_cities)

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
            c.add(gates.X(int(self.two_to_one[ordering[i], i])))
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
                self.two_to_one[u, j],
                self.two_to_one[v, (j + 1) % self.num_cities],
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
                row_constraint[self.two_to_one[v, j]] = 1
            lp = linear_problem(row_constraint, -1)
            tmp_qp = lp.square()
            tmp_qp = tmp_qp * penalty
            qp = qp + tmp_qp

        # column constraints
        for j in range(self.num_cities):
            col_constraint = [0 for _ in range(self.num_cities**2)]
            for v in range(self.num_cities):
                col_constraint[self.two_to_one[v, j]] = 1
            lp = linear_problem(col_constraint, -1)
            tmp_qp = lp.square()
            tmp_qp = tmp_qp * penalty
            qp = qp + tmp_qp
        return qp


class Mis:
    """
    Class for representing the Maximal Independent Set (MIS) problem.

    The MIS problem involves selecting the largest subset of non-adjacent vertices in a graph.

    Args:
        g (networkx.Graph): A graph object representing the problem.

    Example:
        .. testcode::

            import networkx as nx
            from qiboopt.combinatorial_classes.combinatorial_classes import Mis

            g = nx.Graph()
            g.add_edges_from([(0, 1), (1, 2), (2, 0)])
            mis = Mis(g)
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
