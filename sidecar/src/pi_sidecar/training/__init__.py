"""
Training Package.

Provides training pipeline integration for the sidecar.
"""

from .service import RunInfo, RunStatus, TrainingService

__all__ = ["TrainingService", "RunStatus", "RunInfo"]
