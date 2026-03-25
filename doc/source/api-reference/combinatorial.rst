.. _combinatorial:

Combinatorial classes
---------------------

Four Quadratic Unconstrained Binary Optimisation (QUBO) example problems are listed here: the :ref:`Travelling Salesman Problem <TSP>`, the :ref:`Quadratic Assignment Problem <QAP>`, the :ref:`Minimal Vertex Cover <MVC>` and the :ref:`Maximum Independent Set <MIS>`.

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

.. _QAP:

Quadratic Assignment Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Quadratic Assignment Problem, commonly abbreviated as QAP, is a NP-hard problem in combinatorial optimisation.

Briefly, the problem concerns assigning a set of facilities to a set of locations in a way that minimises the total cost. The cost is typically determined by the flow between facilities and the distance between their assigned locations. QAP is usually formulated using two matrices: one representing flows between facilities and another representing distances between locations.

.. autoclass:: qiboopt.combinatorial.combinatorial.QAP
    :members:
    :member-order: bysource

.. _MVC:

Minimum Vertex Cover
^^^^^^^^^^^^^^^^^^^^

The Minimum Vertex Cover Problem, commonly abbreviated as MVC, is a NP-hard problem in combinatorial optimisation.
Briefly, the problem involves selecting a subset of vertices in a graph such that every edge in the graph is incident to at least one selected vertex. The objective is typically to minimise the number of selected vertices. MVC is usually formulated as a graph problem where nodes represent entities and edges represent relationships between them.

.. autoclass:: qiboopt.combinatorial.combinatorial.MVC
    :members:
    :member-order: bysource


.. _MIS:

Maximum Independent Set
^^^^^^^^^^^^^^^^^^^^^^^

The MIS problem involves selecting the largest subset of non-adjacent vertices in a graph.

.. autoclass:: qiboopt.combinatorial.combinatorial.MIS
    :members:
    :member-order: bysource
