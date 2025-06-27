Installation instructions
=========================

We recommend starting with a fresh virtual environment to avoid dependency conflicts with previously installed packages.

.. code-block:: bash

   $ python -m venv ./env
   source activate ./env/bin/activate

Installing with pip
-------------------

The ``qibo-comb-optimisation`` package along with its dependencies can be installed through pip:

.. code-block:: bash

   pip install qibo_comb_optimisation

Installing from source
----------------------

The latest (development) version can be installed directly from the source repository on Github:

.. code-block::

    git clone https://github.com/qiboteam/qibo_comb_optimisation
    cd qibo_comb_optimisation
    pip install .
