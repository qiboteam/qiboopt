Quickstart
----------

Once installed, ``qiboopt`` allows the general user to solve QUBO problems with the built-in ``QUBO`` class.
Along with the ``QUBO`` class, there are some combinatorial classes found in :class:`qiboopt.combinatorial_classes`.

Formuating a QUBO problem:

- Maximal Independent Set:

.. code-block:: python

   import networkx as nx
   from qiboopt.combinatorial_classes.combinatorial_classes import Mis

   g = nx.Graph()
   g.add_edges_from([(0, 1), (1, 2), (2, 0)])
   mis = Mis(g)
   penalty = 10
   qp = mis.penalty_method(penalty)

- Shortest Vector Problem:

.. code-block:: python

   Qdict = {(0, 0): 1.0, (0, 1): 0.5, (1, 1): -1.0}
   qp = QUBO(0, Qdict)

   # Brute force search by evaluating all possible binary vectors.
   opt_vector, min_value = qp.brute_force()

Use `QAOA <https://arxiv.org/abs/1709.03489>`_ to solve the QUBO problems (``qp``):

.. code-block:: python

   from qiboopt.opt_class.opt_class import QUBO
   # Train 2 layers of regular QAOA
   gammas = [0.1, 0.2]
   betas = [0.3, 0.4]
   output = qp.train_QAOA(gammas=gammas, betas=betas)

Use `XQAOA <https://arxiv.org/abs/2302.04479>`_ to solve the QUBO problems (``qp``):

.. code-block:: python

   from qiboopt.opt_class.opt_class import QUBO
   # Train 2 layers of XQAOA
   gammas = [0.1, 0.2]
   betas = [0.3, 0.4]
   alphas = [0.5, 0.6]
   output = qp.train_QAOA(gammas=gammas, betas=betas, alphas=alphas)

Use `QAOA <https://arxiv.org/abs/1709.03489>`_ to solve the QUBO problems (``qp``) using Conditional Variance at Risk (CVaR) loss function:

.. code-block:: python

   from qiboopt.opt_class.opt_class import QUBO
   # Train 2 layers of regular QAOA with CVaR
   gammas = [0.1, 0.2]
   betas = [0.3, 0.4]
   output = qp.train_QAOA(gammas=gammas, betas=betas, regular_loss=False, cvar_delta=0.1)
