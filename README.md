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

Github: https://github.com/brentyi/fannypack
