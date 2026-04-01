Enumerates lattice points $\\{x\in\mathbb{Z}^{\text{dim}}: Hx\geq\text{rhs}\\}$ for $H\in\mathbb{Z}^{N,\text{dim}}$ and $\text{rhs}\in\mathbb{Z}$.

The main use case is for finding lattice points in convex cones, for which $H$ are the inwards-facing hyperplanes. If $\text{rhs}=0$, this will find lattice points in the cone, including its boundary. If $\text{rhs}=1$, then this only finds lattice points in the strict interior of the cone.

## Algorithm Notes

This repo contains a Cython wrapper of a C implementation of [Kannan's algorithm](https://doi.org/10.1287/moor.12.3.415). See [this webpage](https://cseweb.ucsd.edu/~daniele/Lattice/Enum.html) for some other relevant work. The core algorithm enumerates lattice points in square boxes $|x_i|\leq B$ for $B\geq 1$. I.e.,

$$ \\{x\in\mathbb{Z}^{\text{dim}}: Hx\geq\text{rhs} \text{ and } |x|_\infty \leq B\\}. $$

A helper method is provided in case the user wants $N$ points but doesn't care about box size. In this case, boxes of increasing sizes are studied until $\geq N$ lattice points are found.

## Organization

```
conevecs/
├── conevecs/
│   ├── box_enum.h               # STB-style library for the Kannan enumeration
|   ├── box_enum.pyx             # Cython wrapper
|   └── conevecs.py              # a wrapper for box_enum, increasing box size until N points are found
├── tests/
│   ├── test_box_enum.py         # generic tests
│   ├── test_manwe.py            # tests relating to 'Manwe' https://arxiv.org/abs/2406.13751
│   ├── benchmark_box_enum.py    # simple B-dilation benchmark for 'Manwe'
|   └── c/                       # simple C-kernel tests (no Python interface)
├── pyproject.toml
└── setup.py
```
