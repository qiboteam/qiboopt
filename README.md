# Qiboopt

Qiboopt is a plugin to [Qibo](https://github.com/qiboteam/qibo) for solving combinatorial optimization problems.

## Documentation

For the complete documentation on qiboopt, please refer to [qiboopt](https://qibo.science/qiboopt/latest/).

## Minimum working example

Constructing a Maximal Independent Set problem as a QUBO and solving it using [QAOA](https://arxiv.org/abs/1709.03489):

```python
import networkx as nx
from qiboopt.combinatorial.combinatorial import MIS

# Defining the problem, and converting it to a QUBO
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 0)])
mis = MIS(G)
penalty = 10
qp = mis.penalty_method(penalty)  # qp is a QUBO class in qiboopt

# Train 2 layers of regular QAOA
gammas = [0.1, 0.2]
betas = [0.3, 0.4]
output = qp.train_QAOA(gammas=gammas, betas=betas)
print(output)
```

## Contact

To get in touch with the community and the developers, consider joining the Qibo workspace on Matrix:

[![Matrix](https://img.shields.io/matrix/qibo%3Amatrix.org?logo=matrix)](https://matrix.to/#/#qibo:matrix.org)

If you have a question about the project, contact us at [📫](mailto:qiboteam@qibo.science).

## Contributing

Contributions, issues and feature requests are welcome.
