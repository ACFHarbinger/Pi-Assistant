from enum import Enum


class RunStatus(Enum):
    """Status of a training run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"