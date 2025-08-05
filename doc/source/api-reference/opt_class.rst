.. _opt_class:

Optimisation classes
--------------------

This module contains classes for formulating and solving QUBO and linear problems.

.. _QUBO:

QUBO
^^^^

QUBO, short for Quadratic Unconstrained Binary Optimisation, are a class of problems which are NP-complete.

When formulated carefully, QUBO problems can be mapped to solve a host of optimisation problems such as :ref:`Travelling Salesman Problem <TSP>`, :ref:`Maximum Independent Set <MIS>`, Quadratic Assignment Problem, Maximum Clique problem, Maximum Cut problem, etc.


.. autoclass:: qiboopt.opt_class.opt_class.QUBO
    :members:
    :member-order: bysource

Linear Problems
^^^^^^^^^^^^^^^

Linear problem write up goes here.

.. _LP:

.. autoclass:: qiboopt.opt_class.opt_class.LinearProblem
    :members:
    :member-order: bysource
