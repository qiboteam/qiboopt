Quickstart
----------

Once installed, ``qiboopt`` allows the general user to solve QUBO problems with the built-in ``QUBO`` class.
Along with the ``QUBO`` class, there are some combinatorial classes found in :class:`qiboopt.combinatorial`.

Formulating a QUBO problem:

- Maximal Independent Set:

.. code-block:: python

   import networkx as nx
   from qiboopt.combinatorial.combinatorial import MIS

   g = nx.Graph()
   g.add_edges_from([(0, 1), (1, 2), (2, 0)])
   mis = MIS(g)
   penalty = 10
   qp = mis.penalty_method(penalty)

- Shortest Vector Problem:

.. code-block:: python

   Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
   qp = QUBO(0, Qdict)

   # Brute force search by evaluating all possible binary vectors.
   opt_vector, min_value = qp.brute_force()

QUBO problems can be solved using the `QAOA <https://arxiv.org/abs/1709.03489>`_ method:

.. code-block:: python

   from qiboopt.opt_class.opt_class import QUBO
   # Train 2 layers of regular QAOA
   gammas = [0.1, 0.2]
   betas = [0.3, 0.4]
   output = qp.train_QAOA(gammas=gammas, betas=betas)

or the more modern `XQAOA <https://arxiv.org/abs/2302.04479>`_ approach:

.. code-block:: python

   from qiboopt.opt_class.opt_class import QUBO
   # Train 2 layers of XQAOA
   gammas = [0.1, 0.2]
   betas = [0.3, 0.4]
   alphas = [0.5, 0.6]
   output = qp.train_QAOA(gammas=gammas, betas=betas, alphas=alphas)

The Conditional Variance at Risk (CVaR) can also be used as an alternative loss function in solving the QUBO problem:

.. code-block:: python

   from qiboopt.opt_class.opt_class import QUBO
   # Train 2 layers of regular QAOA with CVaR
   gammas = [0.1, 0.2]
   betas = [0.3, 0.4]
   output = qp.train_QAOA(gammas=gammas, betas=betas, regular_loss=False, cvar_delta=0.1)
