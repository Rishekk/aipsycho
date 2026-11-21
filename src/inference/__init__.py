"""
Inference package.

Names are imported lazily (PEP 562) so that light modules such as
``src.inference.post_processor`` and ``src.inference.modalities`` can be imported
without pulling in torch / OpenCV / MediaPipe. ``from src.inference import
RealtimePredictor`` keeps working.
"""
import importlib

_LAZY = {
    "RealtimePredictor": ".realtime_predictor",
    "BatchPredictor":    ".predictor",
    "PostProcessor":     ".post_processor",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
