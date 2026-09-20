"""Unit test suite for Phase 4: Sandboxing, State Management, and CI/CD.

Verifies E2BSandboxEngine isolate-or-refuse policy, quota bounds, local fallback execution,
and ConversationCheckpointManager / CheckpointManager snapshot save and restore.
"""

from __future__ import annotations

import tempfile

import pytest

from src.sandbox.e2b_sandbox import (
    E2BSandboxEngine,
    ResourceQuota,
    SandboxUnavailableError,
)
from src.state.checkpoint_manager import (
    CheckpointManager,
    CheckpointState,
    ConversationCheckpointManager,
    ConversationStateCheckpoint,
)


@pytest.mark.asyncio
async def test_sandbox_default_refuses_unisolated():
    """Verify E2BSandboxEngine refuses execution by default when no isolated backend is available."""
    # Ensure no E2B key is used and container runtime is not considered available
    engine = E2BSandboxEngine(api_key="", allow_unsafe_local=False)
    # If a live container runtime happens to be running on the host, resolution would use container;
    # otherwise it refuses. If container is not running, it must raise SandboxUnavailableError.
    if not engine.is_container_available() and not engine.is_gvisor_available():
        with pytest.raises(SandboxUnavailableError) as exc_info:
            await engine.execute_code("print('Refused')", language="python")
        assert "No isolated sandbox backend" in str(exc_info.value) or "Refusing" in str(exc_info.value)


@pytest.mark.asyncio
async def test_sandbox_local_fallback_success():
    """Verify E2BSandboxEngine executes code safely and returns output when allow_unsafe_local=True."""
    engine = E2BSandboxEngine(quota=ResourceQuota(timeout_seconds=5.0), allow_unsafe_local=True)
    code = "print('Hello Athena Sandbox')"
    res = await engine.execute_code(code, language="python")
    assert res.exit_code == 0
    assert "Hello Athena Sandbox" in res.stdout
    assert res.execution_time_ms > 0.0


@pytest.mark.asyncio
async def test_sandbox_timeout_quota():
    """Verify E2BSandboxEngine enforces timeout quota bounds in local dev mode."""
    engine = E2BSandboxEngine(quota=ResourceQuota(timeout_seconds=1.0), allow_unsafe_local=True)
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
        assert saved.schema_version == 2
        assert isinstance(saved, ConversationStateCheckpoint)
        assert issubclass(CheckpointState, ConversationStateCheckpoint)

        # Restore checkpoint
        restored = manager.restore_checkpoint("ckpt-001")
        assert restored is not None
        assert restored.step_index == 2
        assert len(restored.turn_history) == 2
        assert restored.tool_memory["last_query"] == tool_mem["last_query"]
        assert restored.schema_version == 2


def test_conversation_checkpoint_manager_alias():
    """Verify ConversationCheckpointManager functions identically to CheckpointManager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConversationCheckpointManager(storage_dir=tmpdir)
        state = manager.create_checkpoint(
            checkpoint_id="conv-001",
            step_index=1,
            turn_history=[{"turn": 1}],
        )
        assert state.checkpoint_id == "conv-001"
        restored = manager.restore_checkpoint("conv-001")
        assert restored is not None
        assert restored.checkpoint_id == "conv-001"
