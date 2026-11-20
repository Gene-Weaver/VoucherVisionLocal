"""VoucherVisionLocal backend package."""

import sys
from pathlib import Path

APP_NAME = "VoucherVisionLocal"
PHASE = 2


def block_openvino_telemetry() -> None:
    """Keep OpenVINO from sending usage events to Google Analytics.

    `import openvino` imports openvino.tools.ovc, which sends a "general_import"
    event through the `openvino_telemetry` package unless the user opted out
    globally. When that package cannot be imported, OpenVINO uses its own
    no-op stub instead. Marking it unimportable here affects this process only
    and leaves the user's global consent file alone.
    """
    sys.modules.setdefault("openvino_telemetry", None)


block_openvino_telemetry()


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
