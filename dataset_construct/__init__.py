"""Vis-Interact Dataset Construction Package"""
from .core import get_config


def create_pipeline(*args, **kwargs):
    from .pipeline import create_pipeline as _create_pipeline

    return _create_pipeline(*args, **kwargs)

__version__ = "2.0.0"
__all__ = ["get_config", "create_pipeline"]
