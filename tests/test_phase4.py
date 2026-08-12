"""Unit test suite for Phase 4: Sandboxing, State Management, and CI/CD.

Verifies E2BSandboxEngine quota bounds, local fallback execution, and CheckpointManager
DeltaState C/R snapshot save and restore.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from src.sandbox.e2b_sandbox import E2BSandboxEngine, ResourceQuota
from src.state.checkpoint_manager import CheckpointManager, CheckpointState


@pytest.mark.asyncio
async def test_sandbox_local_fallback_success():
    """Verify E2BSandboxEngine executes code safely and returns output."""
    engine = E2BSandboxEngine(quota=ResourceQuota(timeout_seconds=5.0))
    code = "print('Hello Athena Sandbox')"
    res = await engine.execute_code(code, language="python")
    assert res.exit_code == 0
    assert "Hello Athena Sandbox" in res.stdout
    assert res.execution_time_ms > 0.0


@pytest.mark.asyncio
async def test_sandbox_timeout_quota():
    """Verify E2BSandboxEngine enforces timeout quota bounds."""
    engine = E2BSandboxEngine(quota=ResourceQuota(timeout_seconds=1.0))
    code = "import time; time.sleep(5)"
    res = await engine.execute_code(code, language="python")
    assert res.exit_code == 124
    assert "quota violation" in res.stderr.lower()


def test_checkpoint_manager_save_and_restore():
    """Verify CheckpointManager creates and restores CheckpointState snapshots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(storage_dir=tmpdir)
        turn_history = [
            {"turn": 1, "prompt": "Initial task", "output": "Plan created"},
            {"turn": 2, "prompt": "Execute Athena query", "output": "Query completed"},
        ]
        tool_mem = {"last_query": "SELECT user_id FROM logs WHERE dt='2026-08-12' LIMIT 10;"}

        # Create checkpoint
        saved = manager.create_checkpoint(
            checkpoint_id="ckpt-001",
            step_index=2,
            turn_history=turn_history,
            tool_memory=tool_mem,
        )
        assert saved.checkpoint_id == "ckpt-001"
        assert saved.step_index == 2

        # Restore checkpoint
        restored = manager.restore_checkpoint("ckpt-001")
        assert restored is not None
        assert restored.step_index == 2
        assert len(restored.turn_history) == 2
        assert restored.tool_memory["last_query"] == tool_mem["last_query"]
