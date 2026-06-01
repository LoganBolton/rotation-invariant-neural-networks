# rotation-invariant-neural-networks

This repository contains code for the paper:

**Roxana Bujack, Emily Shinkle, Alice Allen, Tomas Suk, and Nicholas Lubbers**. *Flexible Moment-Invariant Bases from Irreducible Tensors*. arXiv preprint arXiv:2503.21939, 2025. [https://arxiv.org/abs/2503.21939](https://arxiv.org/abs/2503.21939)

If you use this code in your research, please cite the paper:
```bibtex @article{bujack2025flexible, title={Flexible Moment-Invariant Bases from Irreducible Tensors}, author={Bujack, Roxana and Shinkle, Emily and Allen, Alice and Suk, Tomas and Lubbers, Nicholas}, journal={arXiv preprint arXiv:2503.21939}, year={2025}, url={https://arxiv.org/abs/2503.21939}, doi={10.48550/arXiv.2503.21939} } ```


Triad National Security, LLC (Triad) owns the copyright to rotation-invariant-neural-networks, which it identifies as project number O4827.

Copyright 2025. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

## Setup

```
# Download the submodule contents.
git submodule update --init --recursive

# Create/sync the uv environment from pyproject.toml and uv.lock.
uv sync

# Run the test suite.
uv run pytest
```
## Todo

1. The current method to make classification work with the central node readout only is to zero out the contribution of all the atoms besides the central atom. In the future, it would be nice to have a `ClassificationNode` that inputs an atom's features into an MLP to make predictions that way.