import pytest
from pi_sidecar.enums.run import RunStatus


def test_run_status_values():
    """Test RunStatus enum has expected values."""
    assert RunStatus.PENDING.value == "pending"
    assert RunStatus.RUNNING.value == "running"
    assert RunStatus.COMPLETED.value == "completed"
    assert RunStatus.FAILED.value == "failed"
    assert RunStatus.CANCELLED.value == "cancelled"


def test_run_status_from_string():
    """Test constructing RunStatus from string."""
    assert RunStatus("pending") == RunStatus.PENDING
    assert RunStatus("running") == RunStatus.RUNNING
    assert RunStatus("completed") == RunStatus.COMPLETED


def test_run_status_invalid():
    """Test invalid RunStatus raises ValueError."""
    with pytest.raises(ValueError):
        RunStatus("invalid_status")
