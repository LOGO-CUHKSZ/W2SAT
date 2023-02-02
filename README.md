# W2SAT: Learning to generate SAT instances from Weighted Literal Incidence Graphs

This is the official implementation of the our paper ["W2SAT: Learning to generate SAT instances from Weighted Literal Incidence Graphs"](https://arxiv.org/abs/2302.00272).

The dependency of the code is in ``requirement.txt``.

The framework can be found in ``main.ipynb``.

Some related method should be found in utils. (The algorithm ``OWC`` in the paper is implemented by function ``lazy_clique_edge_cover``)

The ``tool`` directory contains some scripts for evaluation. The user maybe need to re-complie the ``glucose`` and modify the path in scripts according to platform for solver performance testing.
