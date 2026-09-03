"""PipeGuard DSL-neutral reference checker."""

from .checker import check_module
from .model import load_module, normalize_module

__all__ = ["check_module", "load_module", "normalize_module"]
__version__ = "0.2.0"
