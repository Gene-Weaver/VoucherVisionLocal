"""VoucherVisionLocal backend package."""

from pathlib import Path

APP_NAME = "VoucherVisionLocal"
PHASE = 2


def _read_version() -> str:
    """The VERSION file beside app.py wins (a source checkout); else package metadata."""
    version_file = Path(__file__).resolve().parent.parent / "VERSION"
    try:
        text = version_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("vouchervision-local")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    return "0.0.0"


__version__ = _read_version()
