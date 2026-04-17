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

The Quadratic Assignment Problem, commonly abbreviated as QAP, is a NP-hard problem in combinatorial optimisation, first introduced by Koopmans and Beckmann.

Briefly, the problem concerns assigning a set of facilities to a set of locations in a way that minimises the total assignment cost. The assignment cost is typically determined by the flow between facilities and the distance between their assigned locations. QAP is usually formulated using two matrices: one representing flows between facilities and another representing distances between locations.

To map QAO to a QUBO, we define binary variables :math:`x \in \{0, 1\}`, where 1 denotes if the :math:`i`-th facility is assigned to the :math:`j`-th location and 0 otherwise.

The objective function is

.. math::
    \min \sum_{i,k=1}^{n} \sum_{j,l=1}^{n} f_{ik} \, d_{jl} \, x_{ij} \, x_{kl}

subject to the assignment constraints

.. math::

    \sum_{k=1}^{n} x_{ik} = 1 \quad \forall i \in \{1,\dots,n\}

and

.. math::

    \sum_{i=1}^{n} x_{ik} = 1 \quad \forall k \in \{1,\dots,n\}

where :math:`f_{ik}` are the elements of the flow matrix :math:`F \in \mathbb{R}^{n \times n}`,
representing the interaction between facilities :math:`i` and :math:`k`, and :math:`d_{jl}` are
the elements of the distance matrix :math:`D \in \mathbb{R}^{n \times n}`, representing the
distance between locations :math:`j` and :math:`l`.

.. autoclass:: qiboopt.combinatorial.combinatorial.QAP
    :members:
    :member-order: bysource

.. _MVC:

Minimum Weighted Vertex Cover
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Minimum Weighted Vertex Cover Problem, commonly abbreviated as MWVC, is a NP-hard problem in combinatorial optimisation.

Briefly, the problem involves selecting a subset of vertices in a graph such that every edge in the graph is incident to at least one selected vertex. Each vertex is assigned a weight and the objective is to minimise the total weight of the selected vertices (rather than their count). MWVC is usually formulated as a graph problem where nodes represent entities and edges represent relationships between them.

Given a graph :math:`G = (V, E)`, the goal of MWVC is to select a subset of vertices in
such a way that every edge is covered while minimising total vertex weight.

Define binary variables :math:`x_i \in \{0, 1\}` for each vertex :math:`i \in V`, where :math:`x_i = 1` indicates that vertex :math:`i` is selected and 0 otherwise.

To map MWVC to a QUBO, constraints are enforced via penalty terms:

.. math::
    \min \sum_{i \in V} w_i x_i + P \sum_{(i,j) \in E} (1 - x_i - x_j)^2

where :math:`w_i` is the weight of vertex :math:`i` and :math:`P` is a sufficiently large penalty coefficient enforcing coverage.

.. autoclass:: qiboopt.combinatorial.combinatorial.MWVC
    :members:
    :member-order: bysource

.. _MIS:

Maximum Independent Set
^^^^^^^^^^^^^^^^^^^^^^^

The MIS problem involves selecting the largest subset of non-adjacent vertices in a graph.

.. autoclass:: qiboopt.combinatorial.combinatorial.MIS
    :members:
    :member-order: bysource
