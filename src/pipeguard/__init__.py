"""PipeGuard DSL-neutral reference checker."""

from .model import load_module, normalize_module

__all__ = ["check_module", "load_module", "normalize_module"]
__version__ = "0.2.0"


def __getattr__(name: str):
    if name != "check_module":
        raise AttributeError(name)
    from .checker import check_module

    globals()[name] = check_module
    return check_module
