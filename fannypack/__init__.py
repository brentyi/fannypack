from typing import TYPE_CHECKING

# When not type-checking, submodules are loaded lazily to reduce import time
if TYPE_CHECKING:
    from . import data, nn, utils

__all__ = ["data", "nn", "utils"]


# Lazy submodule loading
def __getattr__(name):
    import importlib

    module = importlib.import_module(__name__)
    if name not in __all__:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}")
    imported = importlib.import_module(f".{name}", module.__spec__.parent)
    setattr(module, name, imported)
    return imported
