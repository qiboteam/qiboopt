.. _combinatorial:

Combinatorial classes
---------------------

Two Quadratic Unconstrained Binary Optimisation (QUBO) example problems are listed here: the :ref:`Travelling Salesman Problem <TSP>` and the :ref:`Maximum Independent Set <MIS>`.

.. _TSP:

Travelling Salesman Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Travelling Salesman Problem (sometimes referred to as the Travelling Salesperson Problem), commonly abbreviated as TSP, is a NP-hard problem in combinatorial optimisation.

Briefly, the problem revolves around finding the shortest possible route for a salesman to visit some cities before returning to the origin. TSP is usually formulated as a graph problem with nodes specifying the cities and edges denoting the distances between each city.

The idea behind TSP can be mapped to similar-type problems. For instance, what is the optimal route for the salesman to take in order to minimise something.

In this module, the TSP class follows `Hadfield's 2017 paper <https://arxiv.org/abs/1709.03489>`_.

.. autoclass:: qiboopt.combinatorial.combinatorial.TSP
    :members:
    :member-order: bysource

.. _MIS:

Maximum Independent Set
^^^^^^^^^^^^^^^^^^^^^^^

The MIS problem involves selecting the largest subset of non-adjacent vertices in a graph.

.. autoclass:: qiboopt.combinatorial.combinatorial.Mis
    :members:
    :member-order: bysource
