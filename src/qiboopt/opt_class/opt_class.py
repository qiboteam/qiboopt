"""
Optimisation classes
"""

import itertools
from collections import defaultdict

import numpy as np
from qibo import Circuit, gates, hamiltonians
from qibo.backends import _check_backend
from qibo.config import raise_error
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.models import QAOA
from qibo.optimizers import optimize
from qibo.symbols import Z


class QUBO:

    def __init__(self, offset, *args):
        """Initializes the QUBO class

        Args:
            offset (float): The constant offset of the QUBO problem.
            args (dict or np.ndarray): Input for parameters for QUBO or Ising formulation.
                 If len(args)==1, args needs to be a dictionary representing the quadratic
                 coefficient assigned to the QDict object. It represents the matrix Q.
                 If len(args)==2, arg needs to be a list of two dictionaries representing the
                 inputs h and J for Ising formulation.

                 We have the following relation

                    .. math::

                     s'  J  s + h'  s = offset + x'  Q x

                 where
                    h (dict[variable, bias]): Linear biases as a dict of the form {v: bias, ...},
                        where keys are variables of the model and values are biases.
                    J (dict[(variable, variable), bias]): Quadratic biases as a dict of the form
                        {(u, v): bias, ...}, where keys are 2-tuples of variables of the model
                        and values are opt_class biases associated with the pair of
                        variables (the interaction).

        Example:
            .. testcode::
                Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, Qdict)
                print(qp.Qdict)
                # >>> {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}

                h = {3: 1.0, 4: 0.82, 5: 0.23}
                J = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, h, J)
                print(qp.Qdict)
                # >>> ({3: 1.0, 4: 0.82, 5: 0.23}, {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0})
        """

        self.offset = offset
        if len(args) == 1 and isinstance(args[0], dict):
            self.Qdict = args[0]
            if self.Qdict:
                self.n = max(max(key) for key in self.Qdict) + 1
            else:
                self.n = 0
            self.h, self.J, self.ising_constant = self.qubo_to_ising()
        elif len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            h = args[0]
            J = args[1]
            self.h = h
            self.J = J
            self.Qdict = {(v, v): 2.0 * bias for v, bias in h.items()}
            self.n = 0

            # next the opt_class biases
            for (u, v), bias in self.Qdict.items():
                if bias:
                    self.Qdict[(u, v)] = 4.0 * bias
                    self.Qdict[(u, u)] = self.Qdict.get((u, u), 0) - 2.0 * bias
                    self.Qdict[(v, v)] = self.Qdict.get((v, v), 0) - 2.0 * bias
                    self.n = max([self.n, u, v])
            self.n += 1
            # finally adjust the offset based on QUBO definitions rather than Ising formulation
            self.offset += sum(J.values()) - sum(h.values())
        else:
            raise_error(TypeError, "Invalid input for QUBO.")

        # Define other class attributes
        self.n_layers = None
        self.num_betas = None

    def _phase_separation(self, circuit, gamma):
        """
        Applies the phase separation layer (corresponding to the Ising model Hamiltonian).
        This step encodes the interaction terms into the quantum circuit.
        """
        # Apply R_z gates for diagonal terms (h_i)
        circuit.add(
            gates.RZ(i, -2 * gamma * self.h[i]) for i in range(self.n)
        )  # -2 * gamma * h_i

        # Apply CNOT and R_z for off-diagonal terms (J_ij)
        for i in range(self.n):
            for j in range(self.n):
                if (i, j) in self.J:
                    weight = self.J[(i, j)]
                    if weight:
                        circuit.add(gates.CNOT(i, j))
                        circuit.add(
                            gates.RZ(j, -2 * gamma * weight)
                        )  # -2 * gamma * J_ij
                        circuit.add(gates.CNOT(i, j))

    def _default_mixer(self, circuit, beta, alpha=None):
        """
        Applies the mixer layer (uniform superposition evolution).
        This step applies RX rotations on each qubit to spread the superposition.
        """
        for i in range(self.n):
            circuit.add(gates.RX(i, 2 * beta))  # Apply RX gates for mixer
            if alpha:
                circuit.add(gates.RY(i, 2 * alpha))

    def _build(self, gammas, betas, alphas=None, custom_mixer=None):
        """
        Constructs the full QAOA circuit for the Ising model with p layers.
        custom_mixer (List[:class:`qibo.models.Circuit`]): An optional function that takes as input custom mixers.
            If len(custom_mixer) == 1, then use this one circuit as mixer for all layers.
            If len(custom_mixer) == len(gammas), then use each circuit as mixer for each layer.
            If len(custom_mixer) != 1 and != len(gammas), raise an error.
        """
        p = len(gammas)

        # Apply initial Hadamard gates (uniform superposition)
        circuit = Circuit(self.n, density_matrix=True)
        circuit.add(gates.H(i) for i in range(self.n))

        for layer in range(p):
            self._phase_separation(
                circuit, gammas[layer]
            )  # Phase separation (Ising model encoding)
            if alphas is not None:
                self._default_mixer(circuit, betas[layer], alphas[layer])
            else:
                if custom_mixer:
                    if len(gammas) != len(betas):
                        raise_error(
                            ValueError, f"Input {len(gammas) = } != {len(betas) = }."
                        )

                    # Extract number of betas per layer
                    betas_per_layer = len(betas) // p
                    if len(custom_mixer) == 1:
                        circuit += custom_mixer[0](
                            betas[
                                layer * betas_per_layer : (layer + 1) * betas_per_layer
                            ]
                        )
                    elif len(custom_mixer) == len(gammas):
                        circuit += custom_mixer[layer](
                            betas[
                                layer * betas_per_layer : (layer + 1) * betas_per_layer
                            ]
                        )

                    # print("<<< OLD <<<")
                    # for data in custom_mixer[layer].raw['queue']:
                    #     print(data)

                    # num_param_gates = len(
                    #     custom_mixer[layer].trainable_gates
                    # )  # sum(1 for data in custom_mixer[layer].raw['queue'] if data['name'] == 'crx')
                    # new_beta = np.repeat(betas[layer], num_param_gates)
                    # custom_mixer[layer].set_parameters(new_beta)

                    # print(">>> NEW >>>")
                    # for data in custom_mixer[layer].raw['queue']:
                    #     print(data)

                else:
                    self._default_mixer(circuit, betas[layer])

        circuit.add(gates.M(i) for i in range(self.n))

        return circuit

    def __add__(self, other_quadratic):
        """
        Args:
            other_Quadratic: another QUBO class object
        Returns:
            QUBO: A new QUBO object representing the sum of self and other_Quadratic

        Example:
            .. testcode::
                Qdict1 = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp1 = QUBO(0, Qdict1)
                Qdict2 = {(0, 0): 2.0, (1, 1): 1.0}
                qp2 = QUBO(1, Qdict2)
                qp3 = qp1 + qp2
                print(qp3.Qdict)
                # >>> {(0, 0): 3.0, (0, 1): 0.5, (1, 1): 0.0}
                print(qp3.offset)
                # >>> 1.0
                print(qp1.Qdict)  # Original qp1 unchanged
                # >>> {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
        """
        # Create a deep copy of the current QUBO's Qdict
        new_Qdict = self.Qdict.copy()

        # Add the other QUBO's coefficients
        for key, value in other_quadratic.Qdict.items():
            new_Qdict[key] = new_Qdict.get(key, 0.0) + value

        # Calculate the new offset
        new_offset = self.offset + other_quadratic.offset

        # Create and return a new QUBO object
        return self.__class__(new_offset, new_Qdict)

    def __mul__(self, scalar):
        """
        Implements scalar multiplication: qp * 2

        Args:
            scalar (float): The scalar value to multiply by
        Returns:
            QUBO: A new QUBO object with all coefficients multiplied by the scalar

        Example:
            .. testcode::
                Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, Qdict)
                qp2 = qp * 2
                print(qp2.Qdict)
                # >>> {(0, 0): 2.0, (0, 1): 1.0, (1, 1): -2.0}
                print(qp.Qdict)  # Original unchanged
                # >>> {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply QUBO by scalar (int or float)")

        new_Qdict = {key: value * scalar for key, value in self.Qdict.items()}
        new_offset = self.offset * scalar
        return self.__class__(new_offset, new_Qdict)

    def __rmul__(self, scalar):
        """
        Implements right scalar multiplication: 2 * qp

        Args:
            scalar (float): The scalar value to multiply by
        Returns:
            QUBO: A new QUBO object with all coefficients multiplied by the scalar
        """
        return self.__mul__(scalar)

    def qubo_to_ising(self, constant=0.0):
        """Convert a QUBO problem to an Ising problem.

        Maps a quadratic unconstrained binary optimisation (QUBO) problem defined over
        binary variables (0 or 1 values), where the linear term is contained along x' Qx
        the diagonal of Q, to an Ising model defined on spins (variables with {-1, +1} values).
        Returns `h` and `J` that define the Ising model as well as `constant` representing the
        offset in energy between the two problem formulations.

        .. math::

             x'  Q  x  = constant + s'  J  s + h'  s

        Args:
            Q (dict[(variable, variable), coefficient]): QUBO coefficients in a dict of form
                {(u, v): coefficient, ...}, where keys are 2-tuples of variables of the model
                and values are biases associated with the pair of variables. Tuples (u, v)
                represent interactions and (v, v) linear biases.
            constant (float): Constant offset to be applied to the energy. Defaults to 0.

        Returns:
            (dict, dict, float): A 3-tuple containing:
            h (dict): Linear coefficients of the Ising problem.
            J (dict): Quadratic coefficients of the Ising problem.
            constant (float): The new energy offset.

        Example:
            This example converts a QUBO problem of two variables that have positive
            biases of value 1 and are positively coupled with an interaction of value 1
            to an Ising problem, and shows the new energy offset.
        """
        h = {}
        J = {}
        linear_offset = 0.0
        quadratic_offset = 0.0

        for (u, v), bias in self.Qdict.items():
            if u == v:
                h[u] = h.get(u, 0) + bias / 2
                linear_offset += bias

            else:
                if bias:
                    J[(u, v)] = bias / 4
                h[u] = h.get(u, 0) + bias / 4
                h[v] = h.get(v, 0) + bias / 4
                quadratic_offset += bias

        constant += 0.5 * linear_offset + 0.25 * quadratic_offset

        return h, J, constant

    def construct_symbolic_Hamiltonian_from_QUBO(self):
        """Constructs a symbolic Hamiltonian from the QUBO problem by converting it
        to an Ising model.

        The method calls the qubo_to_ising function to convert the QUBO formulation
        into an Ising Hamiltonian with linear and quadratic terms. Then, it creates
        a symbolic Hamiltonian using the qibo library.

        Returns:
            ham (`qibo.hamiltonians.hamiltonians.SymbolicHamiltonian`): A symbolic
                Hamiltonian that corresponds to the QUBO problem.
        """
        # Correct the call to qubo_to_ising (no need to pass self.Qdict)
        h, J, constant = self.qubo_to_ising()

        # Create a symbolic Hamiltonian using qibo symbols
        symbolic_ham = sum(h[i] * Z(i) for i in h)
        symbolic_ham += sum(J[u, v] * Z(u) * Z(v) for (u, v) in J)
        symbolic_ham += constant

        # Return the symbolic Hamiltonian using qibo's Hamiltonian object
        ham = hamiltonians.SymbolicHamiltonian(symbolic_ham)
        return ham

    def evaluate_f(self, x):
        """Evaluates the quadratic function for a given binary vector.

        Args:
            x (list): A list representing the binary vector for which to evaluate the function.

        Returns:
            f_value (float): The evaluated function value.

        Example:
            .. testcode::

                Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, Qdict)
                x = [1, 1]
                print(qp.evaluate_f(x))
                # >>> 0.5
        """
        f_value = self.offset
        for i in range(self.n):
            if x[i]:
                f_value += (
                    self.Qdict[(i, i)] if (i, i) in self.Qdict else 0.0
                )  # manage diagonal term first
                f_value += sum(
                    self.Qdict.get((i, j), 0) + self.Qdict.get((j, i), 0)
                    for j in range(i + 1, self.n)
                    if x[j]
                )
        return f_value

    def evaluate_grad_f(self, x):
        """Evaluates the gradient of the quadratic function at a given binary vector.

        Args:
            x (List[int]): A list representing the binary vector for which to evaluate the gradient.

        Returns:
            grad (List): List of float representing the gradient vector.

        Example:
            .. testcode::

                Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, Qdict)
                x = [1, 1]
                print(qp.evaluate_grad_f(x))
                # >>> [1.5, -0.5]
        """
        grad = np.asarray([self.Qdict.get((i, i), 0) for i in range(self.n)])
        for i in range(self.n):
            for j in range(self.n):
                if j != i and x[j] == 1:
                    grad[i] += self.Qdict.get((i, j), 0) + self.Qdict.get((j, i), 0)
        return grad

    def tabu_search(self, max_iterations=100, tabu_size=10):
        """Solves the QUBO problem using the Tabu search algorithm.

        Args:
            max_iterations (int): Maximum number of iterations to run the Tabu search.
                Defaults to 100.
            tabu_size (int): Size of the Tabu list.

        Returns:
            best_solution (list): List of ints representing the best binary vector found.
            best_obj_value (float): The corresponding objective value.

        Example:
            .. testcode::

                Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, Qdict)
                best_solution, best_obj_value = qp.tabu_search(50, 5)
                print(best_solution)
                # >>> [0, 1]
                print(best_obj_value)
                # >>> 0.5
        """
        x = np.random.randint(2, size=self.n)  # Initial solution
        best_solution = x.copy()
        best_obj_value = self.evaluate_f(x)
        tabu_list = []

        for _ in range(max_iterations):
            neighbors = []
            for i in range(self.n):
                neighbor = x.copy()
                neighbor[i] = 1 - neighbor[i]  # Flip a bit
                neighbors.append((neighbor, self.evaluate_f(neighbor)))

            # Choose the best neighbor that is not tabu
            best_neighbor = min(neighbors, key=lambda x: x[1])
            best_neighbor_solution, best_neighbor_obj = best_neighbor

            # Update the current solution if it's better than the previous best and not tabu
            if (
                best_neighbor_obj < best_obj_value
                and best_neighbor_solution.tolist() not in tabu_list
            ):
                x = best_neighbor_solution
                best_solution = x.copy()
                best_obj_value = best_neighbor_obj

            # Add the best neighbor to the tabu list
            tabu_list.append(best_neighbor_solution.tolist())
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

        return best_solution, best_obj_value

    def brute_force(self):
        """Solves the QUBO problem by evaluating all possible binary vectors.
            Note that this approach is very slow.

        Returns:
            opt_vector (list): List of ints representing the optimal binary vector.
            min_value (float): The minimum value of the objective function.

        Example:
            .. testcode::

                Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, Qdict)
                opt_vector, min_value = qp.brute_force()
                print(opt_vector)
                # >>> [0, 1]
                print(min_value)
                # >>> -1.0
        """
        possible_values = {}
        # A list of all the possible permutations for x vector
        vec_permutations = itertools.product([0, 1], repeat=self.n)

        for permutation in vec_permutations:
            value = self.evaluate_f(permutation)
            possible_values[value] = permutation
        min_value = min(
            possible_values.keys()
        )  # Lowest value of the objective function
        opt_vector = tuple(
            possible_values[min_value]
        )  # Optimum vector x that produces the lowest value
        return opt_vector, min_value

    def canonical_q(self):
        """Converts the QUBO matrix to canonical form where only terms with i < j are retained.

        Returns:
            self.Qdict (dict): A dictionary and also update Qdict
        """
        Qdict = {
            (i, j): self.Qdict.get((i, j), 0) + self.Qdict.get((j, i), 0)
            for i in range(self.n)
            for j in range(i, self.n)
            if (i, j) in self.Qdict or (j, i) in self.Qdict
        }
        self.Qdict = Qdict
        return self.Qdict

    def qubo_to_qaoa_circuit(self, gammas, betas, alphas=None, custom_mixer=None):
        """
        Constructs a QAOA or XQAOA circuit for the given QUBO problem.

        Args:
            gammas (List[float]): parameters for phasers
            betas (List[float]): parameters for X mixers
            alphas (List[float], optional): parameters for Y mixers for XQAOA
            custom_mixer (List[:class:`qibo.models.Circuit`]): optional argument that takes as input custom mixers.
                If len(custom_mixer) == 1, then use this one circuit as mixer for all layers.
                If len(custom_mixer) == len(gammas), then use each circuit as mixer for each layer.
                If len(custom_mixer) != 1 and != len(gammas), raise an error.

        Returns:
            circuit (:class:`qibo.models.Circuit`): The QAOA or XQAOA circuit corresponding to the QUBO problem.
        """
        if alphas is not None:  # Use XQAOA, ignore mixer_function
            circuit = self._build(gammas, betas, alphas)
        else:
            if custom_mixer:
                circuit = self._build(
                    gammas, betas, alphas=None, custom_mixer=custom_mixer
                )
            else:
                circuit = self._build(gammas, betas)
        return circuit

    def train_QAOA(
        self,
        gammas=None,
        betas=None,
        alphas=None,
        p=None,
        nshots=int(1e3),
        regular_loss=True,
        maxiter=10,
        method="cobyla",
        cvar_delta=0.25,
        custom_mixer=None,
        backend=None,
        noise_model=None,
    ):
        """
        Constructs the QAOA or XQAOA circuit with optional parameters for the mixers or phases before using a classical
        optimiser to search for the optimal parameters which minimise the cost function (either expected value or
        Conditional Variance at Risk (CVaR).

        Args:
            gammas (List[float], optional): parameters for phasers.
            betas  (List[float], optional): parameters for X mixers.
            alphas (List[float], optional): parameters for Y mixers for XQAOA. Defaults to None.
            p (int, optional): number of layers.
            nshots (int, optional): number of shots
            regular_loss (Bool, optional): If False, Conditional Variance at Risk (CVaR) is used as cost function.
                Defaults to True, where expected value is used as cost function.
            maxiter (int, optional): Maximum number of iterations used in the minimiser. Defaults to 10.
            cvar_delta (float, optional): Represents the quantile threshold used for calculating the CVaR. Defaults to
                `0.25`.
            custom_mixer (List[:class:`qibo.models.Circuit`]): optional argument that takes as input custom mixers.
                If len(custom_mixer) == 1, then use this one circuit as mixer for all layers.
                If len(custom_mixer) == len(gammas), then use each circuit as mixer for each layer.
                If len(custom_mixer) != 1 and != len(gammas), raise an error.
            backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used in the execution.
                If ``None``, it uses the current backend. Defaults to ``None``.
            noise_model (:class:`qibo.noise.NoiseModel`, optional): noise model applied to simulate noisy computations.
                Defaults to None.

        Returns:
            Tuple[float, List[float], dict, :class:`qibo.models.Circuit`, dict]: A tuple containing:
                - best (float): The lowest cost value achieved.
                - params (List[float]): Optimised QAOA parameters.
                - extra (dict): Additional metadata (e.g., convergence info).
                - circuit (:class:`qibo.models.Circuit`): Final circuit used for evaluation.
                - frequencies (dict): Bitstring outcome frequencies from measurement.

        Example:
            .. testcode::

                Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
                qp = QUBO(0, Qdict)
                opt_vector, min_value = qp.brute_force()

                # Train regular QAOA
                output = QUBO(0, Qdict).train_QAOA(p=10)
        """

        backend = _check_backend(backend)

        if p is None and gammas is None:
            raise_error(
                ValueError,
                "Either p or gammas must be provided to define the number of layers.",
            )
        elif p is None:
            p = len(gammas)

        elif gammas is None:
            # if no gammas are provided, we randomly generate them to be between 0 and 2pi
            gammas = np.random.rand(p) * 2 * np.pi
            betas = np.random.rand(p) * 2 * np.pi
        else:
            if len(gammas) != p:
                raise_error(
                    ValueError,
                    f"gammas must be of length {p}, but got {len(gammas)}.",
                )

        self.n_layers = p
        self.num_betas = len(betas)

        circuit = self.qubo_to_qaoa_circuit(
            gammas, betas, alphas=alphas, custom_mixer=custom_mixer
        )
        if noise_model is not None:
            circuit = noise_model.apply(circuit)

        n_params = 3 * p if alphas else 2 * p

        # Block packing: [all gammas][all betas][all alphas]
        parameters = list(gammas) + list(betas)
        if alphas is not None:
            parameters += list(alphas)

        if regular_loss:

            def myloss(parameters):
                """
                Computes the expectation value as loss.

                Args:
                    parameters (List[float]): Parameters used in the circuit.

                Returns:
                    loss (float): The computed expectation value.
                """

                p = len(gammas)
                if alphas is not None:
                    gammas_ = parameters[:p]
                    betas_ = parameters[p : 2 * p]
                    unpacked_alphas = parameters[2 * p : 3 * p]
                else:
                    gammas_ = parameters[:p]
                    betas_ = parameters[p : 2 * p]
                    unpacked_alphas = None
                circuit = self.qubo_to_qaoa_circuit(
                    gammas_, betas_, alphas=unpacked_alphas, custom_mixer=custom_mixer
                )
                if noise_model is not None:
                    circuit = noise_model.apply(circuit)
                # print("Regular loss" "s circuit:\n")
                # print(circuit)

                result = circuit(None, nshots)
                result_counter = result.frequencies(binary=True)
                energy_dict = defaultdict(int)
                for key in result_counter:
                    x = [int(sub_key) for sub_key in key]
                    energy_dict[self.evaluate_f(x)] += result_counter[key]
                loss = sum(key * energy_dict[key] / nshots for key in energy_dict)
                return loss

        else:

            def myloss(parameters, delta=cvar_delta):
                """
                Computes the CVaR of the energy distribution for a given quantile threshold `delta`.

                Args:
                    parameters (List[float]): Parameters used in the circuit.
                    delta (float): Quantile threshold for CVaR (defaults to 0.25)

                Returns:
                    cvar (float): The computed CVaR value.
                """
                m = len(parameters)
                if alphas is not None:
                    gammas_ = parameters[:p]
                    betas_ = parameters[p : 2 * p]
                    unpacked_alphas = parameters[2 * p : 3 * p]
                else:
                    gammas_ = parameters[:p]
                    betas_ = parameters[p : 2 * p]
                    unpacked_alphas = None
                circuit = self.qubo_to_qaoa_circuit(
                    gammas_, betas_, alphas=unpacked_alphas, custom_mixer=custom_mixer
                )
                if noise_model is not None:
                    circuit = noise_model.apply(circuit)
                # print("CVaR loss" "s circuit:\n")
                # print(circuit)
                # print(">> Optimisation step:\n")
                # for data in circuit.raw["queue"]:
                #     print(data)

                result = backend.execute_circuit(circuit, nshots=nshots)
                result_counter = result.frequencies(binary=True)

                energy_dict = defaultdict(int)
                for key in result_counter:
                    # key is the binary string, value is the frequency
                    x = [int(sub_key) for sub_key in key]
                    energy_dict[self.evaluate_f(x)] += result_counter[key]

                # Normalize frequencies to probabilities
                total_counts = sum(energy_dict.values())
                energy_probs = {
                    key: value / total_counts for key, value in energy_dict.items()
                }

                # Sort energies and compute cumulative probability
                sorted_energies = sorted(
                    energy_probs.items()
                )  # List of (energy, probability)
                cumulative_prob = 0
                selected_energies = []

                for energy, prob in sorted_energies:
                    if cumulative_prob + prob > cvar_delta:
                        # Include only the fraction of the probability needed to reach `cvar_delta`
                        excess_prob = cvar_delta - cumulative_prob
                        selected_energies.append((energy, excess_prob))
                        cumulative_prob = cvar_delta
                        break
                    else:  # pragma: no cover
                        selected_energies.append((energy, prob))
                        cumulative_prob += prob

                # Compute CVaR as weighted average of selected energies
                cvar = (
                    sum(energy * prob for energy, prob in selected_energies)
                    / cvar_delta
                )
                return cvar

        best, params, extra = optimize(
            myloss, parameters, method=method, options={"maxiter": maxiter}
        )
        # Unpack optimised parameters in the same way as in myloss (block format)
        if alphas is not None:
            optimised_gammas = params[:p]
            optimised_betas = params[p : 2 * p]
            optimised_alphas = params[2 * p : 3 * p]
        else:
            optimised_gammas = params[:p]
            optimised_betas = params[p : 2 * p]
            optimised_alphas = None
        circuit = self.qubo_to_qaoa_circuit(
            gammas=optimised_gammas,
            betas=optimised_betas,
            alphas=optimised_alphas,
            custom_mixer=custom_mixer,
        )
        original_circuit = Circuit.copy(circuit)
        if noise_model is not None:
            circuit = noise_model.apply(circuit)

        result = backend.execute_circuit(circuit, nshots=nshots)

        if noise_model is not None:
            return (
                best,
                params,
                extra,
                circuit,
                result.frequencies(binary=True),
                original_circuit,
            )
        else:
            return best, params, extra, circuit, result.frequencies(binary=True)

    def qubo_to_qaoa_object(self, params: list = None):
        """
        Generates a QAOA object for the QUBO problem.

        Args:
            params (List[float]): Parameters of the QAOA given in block format:
                e.g. [all_gammas, all_betas, all_alphas] (if alphas is not None)
        Returns:
            qaoa (`qibo.models.QAOA`): A QAOA circuit for the QUBO problem.
        """

        # Convert QUBO to Ising Hamiltonian
        h, J, _constant = self.qubo_to_ising()

        # Create the Ising Hamiltonian using Qibo
        symbolic_ham = sum(h[i] * Z(i) for i in h)
        symbolic_ham += sum(J[(u, v)] * Z(u) * Z(v) for (u, v) in J)

        # Define the QAOA model
        hamiltonian = SymbolicHamiltonian(symbolic_ham)
        qaoa = QAOA(hamiltonian)

        # Optionally set parameters
        if params is not None:
            qaoa.set_parameters(np.array(params))
        return qaoa


class linear_problem:
    """A class used to represent a linear problem of the form Ax + b.

    Args:
        A (np.ndarray): Coefficient matrix.
        b (np.ndarray): Constant vector.
        n (int): Dimension of the problem, inferred from the size of b.

    Example:
        .. testcode::

            A1 = np.array([[1, 2], [3, 4]])
            b1 = np.array([5, 6])
            lp1 = linear_problem(A1, b1)
            A2 = np.array([[1, 1], [1, 1]])
            b2 = np.array([1, 1])
            lp2 = linear_problem(A2, b2)
            lp3 = lp1 + lp2
            print(lp3.A)
            # >>> [[2 3]
            #      [4 5]]
            print(lp3.b)
            # >>> [6 7]
            print(lp1.A)  # Original lp1 unchanged
            # >>> [[1 2]
            #      [3 4]]
    """

    def __init__(self, A, b):
        # TODO: raise ValueError if A and b have incompatible dimensions.
        self.A = np.atleast_2d(A)
        self.b = np.array([b]) if np.isscalar(b) else np.asarray(b)
        self.n = self.A.shape[1]

    def __add__(self, other_linear):
        """
        Args:
            other_linear: another linear_problem class object
        Returns:
            linear_problem: A new linear_problem object representing the sum of self and other_linear
        """
        # Create copies of the matrices and vectors
        new_A = self.A.copy() + other_linear.A
        new_b = self.b.copy() + other_linear.b

        # Create and return a new linear_problem object
        return self.__class__(new_A, new_b)

    def __mul__(self, scalar):
        """
        Implements scalar multiplication: lp * 2

        Args:
            scalar (float): The scalar value to multiply by
        Returns:
            linear_problem: A new linear_problem object with A and b multiplied by the scalar

        Example:
            .. testcode::
                A = np.array([[1, 2], [3, 4]])
                b = np.array([5, 6])
                lp = linear_problem(A, b)
                lp2 = lp * 2
                print(lp2.A)
                # >>> [[2 4]
                #      [6 8]]
                print(lp2.b)
                # >>> [10 12]
                print(lp.A)  # Original unchanged
                # >>> [[1 2]
                #      [3 4]]
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply linear_problem by scalar (int or float)")

        new_A = self.A.copy() * scalar
        new_b = self.b.copy() * scalar
        return self.__class__(new_A, new_b)

    def __rmul__(self, scalar):
        """
        Implements right scalar multiplication: 2 * lp

        Args:
            scalar (float): The scalar value to multiply by
        Returns:
            linear_problem: A new linear_problem object with A and b multiplied by the scalar
        """
        return self.__mul__(scalar)

    def evaluate_f(self, x):
        """Evaluates the linear function Ax + b at a given point x.

        Args:
            x (np.ndarray): Input vector at which to evaluate the linear function.

        Returns:
            numpy.ndarray: The value of the linear function Ax + b at the given x.

        Example:
            .. testcode::

                A = np.array([[1, 2], [3, 4]])
                b = np.array([5, 6])
                lp = linear_problem(A, b)
                x = np.array([1, 1])
                result = lp.evaluate_f(x)
                print(result)
                # [ 8 13]
        """
        return self.A @ x + self.b

    def square(self):
        """Squares the linear problem to obtain a quadratic problem.

        Returns:
            :class:`qiboopt.opt_class.opt_class.QUBO`: A quadratic problem
            corresponding to squaring the linear function.

        Example:
            .. testcode::

                A = np.array([[1, 2], [3, 4]])
                b = np.array([5, 6])
                lp = linear_problem(A, b)
                Quadratic = lp.square()
                print(Quadratic.Qdict)
                # >>> {(0, 0): 56, (0, 1): 14, (1, 0): 14, (1, 1): 88}
                print(Quadratic.offset)
                # >>> 61
        """
        quadraticpart = self.A.T @ self.A + np.diag(2 * (self.b @ self.A))
        offset = np.dot(self.b, self.b)
        num_rows, num_cols = quadraticpart.shape
        Qdict = {
            (i, j): quadraticpart[i, j]
            for i in range(num_rows)
            for j in range(num_cols)
        }
        return QUBO(offset, Qdict)
