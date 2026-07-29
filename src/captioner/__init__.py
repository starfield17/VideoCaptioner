"""Video Captioner package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("captioner")
except PackageNotFoundError:
    __version__ = "0+unknown"
