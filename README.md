# fannypack

![build](https://github.com/brentyi/fannypack/workflows/build/badge.svg)
![mypy](https://github.com/brentyi/fannypack/workflows/mypy/badge.svg?branch=master)
![lint](https://github.com/brentyi/fannypack/workflows/lint/badge.svg)
[![codecov](https://codecov.io/gh/brentyi/fannypack/branch/master/graph/badge.svg)](https://codecov.io/gh/brentyi/fannypack)

A loose collection of tools for training PyTorch models.

Contents include helpers for:

- Experiment management, Tensorboard logging, and checkpointing (Python + CLI)
- Reading and manipulating arrays and tensors stored in containers: converting
  between types, moving across (Torch) devices, slicing across shared dimensions
- Freezing and unfreezing portions of PyTorch modules
- Reading and writing time series data/trajectory files via hdf5
- and a lot more!

See [documentation](https://brentyi.github.io/fannypack) for full overview of
functionality.

---

### Installation

Standard installation:

```
pip install fannypack
```

Install from source:

```
git clone https://github.com/brentyi/fannypack.git
cd fannypack && pip install -e .
```

---

### Development

Tests can be run with `pytest`, and documentation can be built by running
`make github` in the `docsource/` directory.

Tooling: [black](https://github.com/psf/black) and
[isort](https://github.com/timothycrosley/isort) for formatting,
[flake8](https://flake8.pycqa.org/en/latest/) for linting, and
[mypy](https://github.com/python/mypy) for static type-checking.

Until `numpy 1.20.0` [is released](https://github.com/numpy/numpy/pull/16515),
type-checking also requires that NumPy stubs are installed manually:

```
pip install https://github.com/numpy/numpy-stubs/tarball/master
```
