# fannypack

![build](https://github.com/brentyi/fannypack/workflows/build/badge.svg)
![mypy](https://github.com/brentyi/fannypack/workflows/mypy/badge.svg?branch=master)
![lint](https://github.com/brentyi/fannypack/workflows/lint/badge.svg)
[![codecov](https://codecov.io/gh/brentyi/fannypack/branch/master/graph/badge.svg)](https://codecov.io/gh/brentyi/fannypack)

Helpers for PyTorch.

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

To enable autocomplete for the `buddy` commandline tool:

- bash:

  ```sh
  # Append to .bashrc
  eval "$(register-python-argcomplete buddy)"
  ```

- zsh:
  ```sh
  # Append to .zshrc
  autoload -U +X compinit && compinit
  autoload -U +X bashcompinit && bashcompinit
  eval "$(register-python-argcomplete buddy)"
  ```

---

Github: https://github.com/brentyi/fannypack

Documentation (work-in-progress): https://brentyi.github.io/fannypack
