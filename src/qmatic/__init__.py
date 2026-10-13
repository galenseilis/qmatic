from importlib.metadata import PackageNotFoundError, version

from . import dists, lindley

__all__ = ["dists", "lindley"]

try:
    __version__ = version("qmatic")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"
