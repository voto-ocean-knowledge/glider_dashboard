"""Glider Dashboard: interactive exploration of ocean glider data from ERDDAP."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("glider-dashboard")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "unknown"

__all__ = ["__version__"]
