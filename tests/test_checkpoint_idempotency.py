"""Test Suite for Checkpoint Idempotency, Effect Ledger, and State Migration (ARS-T06, E3).

Verifies:
1. Idempotency key computation is deterministic and canonicalizes arguments.
2. Lossless round-trip serialization and deserialization of ConversationStateCheckpoint.
3. Backward compatibility / migration for legacy v1 checkpoints (with os_effects, missing schema_version).
4. E3: Interruption after a side effect, followed by resumption, ensures the side effect
   is executed exactly once and replayed from the effect ledger.
5. Absence of claims regarding OS-level checkpoint/restore (CRIU) or 'Crab' in state modules.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.engine.agent_loop import AgentLoop, AgentPlanner
from src.engine.dispatcher import ParallaxToolDispatcher
from src.engine.executor import ExecutiveEngineProcess
from src.reasoning.schemas import ActionType, ReasoningStep
from src.state.checkpoint_manager import (
    ConversationCheckpointManager,
    compute_idempotency_key,
)


def test_idempotency_key_generation():
    """Verify that compute_idempotency_key produces stable, deterministic hashes."""
    key1 = compute_idempotency_key("athena_query", "athena_query", {"sql": "SELECT 1", "limit": 10})
    # Same arguments, different dictionary key order
    key2 = compute_idempotency_key("athena_query", "athena_query", {"limit": 10, "sql": "SELECT 1"})
    assert key1 == key2

    # Different arguments produce different keys
    key3 = compute_idempotency_key("athena_query", "athena_query", {"sql": "SELECT 2", "limit": 10})
    assert key1 != key3

    # Different tool name produces different key
    key4 = compute_idempotency_key("sandbox_execute", "sandbox_execute", {"sql": "SELECT 1", "limit": 10})
    assert key1 != key4


def test_checkpoint_roundtrip_lossless():
    """Verify saving and restoring a ConversationStateCheckpoint produces identical data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConversationCheckpointManager(storage_dir=tmpdir)
        turn_history = [
            {"turn": 1, "prompt": "Analyze dataset", "output": "Created plan with 2 steps"},
            {"turn": 2, "prompt": "Execute step 1", "output": "Query executed successfully"},
        ]
        tool_memory = {"last_query_id": "q-98765", "row_count": 42}
        idempotency_key = compute_idempotency_key("athena_query", "athena_query", {"sql": "SELECT count(*) FROM table1"})
        effect_ledger = {
            idempotency_key: {
                "recorded_at": 1700000000.0,
                "data": {"status": "SUCCEEDED", "rows": 42},
            }
        }

        saved_checkpoint = manager.create_checkpoint(
            checkpoint_id="chk-lossless-001",
            step_index=1,
            turn_history=turn_history,
            tool_memory=tool_memory,
            effect_ledger=effect_ledger,
        )

        restored_checkpoint = manager.restore_checkpoint("chk-lossless-001")
        assert restored_checkpoint is not None
        assert restored_checkpoint.to_dict() == saved_checkpoint.to_dict()
        assert restored_checkpoint.checkpoint_id == "chk-lossless-001"
        assert restored_checkpoint.step_index == 1
        assert restored_checkpoint.turn_history == turn_history
        assert restored_checkpoint.tool_memory == tool_memory
        assert restored_checkpoint.effect_ledger == effect_ledger
        assert restored_checkpoint.has_effect(idempotency_key) is True
        assert restored_checkpoint.get_effect(idempotency_key) == {"status": "SUCCEEDED", "rows": 42}


def test_legacy_checkpoint_v1_migration():
    """Verify that a legacy v1 checkpoint with 'os_effects' is cleanly loaded and migrated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        legacy_data = {
            "checkpoint_id": "legacy-001",
            "step_index": 1,
            "turn_history": [{"turn": 1, "note": "legacy turn"}],
            "tool_memory": {"legacy_key": "legacy_val"},
            "os_effects": ["file_written:/tmp/out.csv"],
            "timestamp": 1690000000.0,
        }

        legacy_file = Path(tmpdir) / "legacy-001.json"
        with open(legacy_file, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f)

        manager = ConversationCheckpointManager(storage_dir=tmpdir)
        restored = manager.restore_checkpoint("legacy-001")

        assert restored is not None
        assert restored.checkpoint_id == "legacy-001"
        assert restored.schema_version == 1  # defaulted from legacy
        assert restored.has_effect("file_written:/tmp/out.csv") is True
        assert restored.get_effect("file_written:/tmp/out.csv") == {"legacy": "file_written:/tmp/out.csv"}
        assert restored.tool_memory == {"legacy_key": "legacy_val"}


@pytest.mark.asyncio
async def test_e3_interrupt_and_resume_idempotent_execution():
    """E3 Test: Simulate an interruption after a side effect, then resume.

    Asserts:
    1. First run executes step 1 (side effect recorded) and fails at step 2.
    2. Side effect was executed once.
    3. Resume re-runs the plan with the saved checkpoint:
       - Step 1 is detected in the effect_ledger and NOT executed again.
       - Step 2 is executed.
    4. Total execution count of the side effect is strictly 1.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_manager = ConversationCheckpointManager(storage_dir=tmpdir)

        # Create spy counters to measure actual invocations
        side_effect_execution_count = 0
        step2_execution_count = 0

        async def effectful_handler(args: dict[str, Any]) -> dict[str, Any]:
            nonlocal side_effect_execution_count
            side_effect_execution_count += 1
            return {"action": "write_record", "id": args.get("id"), "count": side_effect_execution_count}

        should_fail_step2 = True

        async def step2_handler(args: dict[str, Any]) -> dict[str, Any]:
            nonlocal step2_execution_count
            step2_execution_count += 1
            if should_fail_step2:
                raise RuntimeError("Simulated network interrupt during step 2")
            return {"status": "step2_complete"}

        executor = ExecutiveEngineProcess(authorized_tools={"effectful_tool", "step2_tool"})
        executor.register_tool("effectful_tool", effectful_handler)
        executor.register_tool("step2_tool", step2_handler)

        dispatcher = ParallaxToolDispatcher(executor=executor)
        loop = AgentLoop(dispatcher=dispatcher)

        # Build a 2-step plan
        step1 = ReasoningStep(
            step_number=1,
            rationale="Perform first side effect",
            action_type=ActionType.SANDBOX_EXECUTE,
            tool_name="effectful_tool",
            tool_args={"id": "record-123"},
        )
        step2 = ReasoningStep(
            step_number=2,
            rationale="Perform second step which initially fails",
            action_type=ActionType.SANDBOX_EXECUTE,
            tool_name="step2_tool",
            tool_args={"param": "value"},
        )
        planner = AgentPlanner()
        plan = planner.create_plan(
            task_goal="E3 Idempotency Test Plan",
            steps=[step1, step2],
            plan_id="e3-test-plan",
        )

        checkpoint = ckpt_manager.create_checkpoint(
            checkpoint_id="e3-checkpoint",
            step_index=0,
            turn_history=[],
        )

        # --- FIRST EXECUTION (FAILS AT STEP 2) ---
        first_result = await loop.run_plan(
            plan=plan,
            checkpoint=checkpoint,
            checkpoint_manager=ckpt_manager,
        )

        assert first_result.success is False
        assert "Simulated network interrupt" in str(first_result.error)
        assert side_effect_execution_count == 1
        assert step2_execution_count == 1

        # Verify step 1 effect is recorded in the checkpoint
        step1_key = compute_idempotency_key("sandbox_execute", "effectful_tool", {"id": "record-123"})
        assert checkpoint.has_effect(step1_key) is True
        assert checkpoint.step_index == 1

        # Restore checkpoint from disk to simulate a fresh process restart
        resumed_checkpoint = ckpt_manager.restore_checkpoint("e3-checkpoint")
        assert resumed_checkpoint is not None
        assert resumed_checkpoint.has_effect(step1_key) is True

        # --- RESUME EXECUTION (STEP 2 SUCCEEDS) ---
        should_fail_step2 = False  # Transient issue resolved

        second_result = await loop.run_plan(
            plan=plan,
            checkpoint=resumed_checkpoint,
            checkpoint_manager=ckpt_manager,
        )

        assert second_result.success is True
        assert second_result.error is None
        assert len(second_result.step_results) == 2
        # Step 1 was served from cache, NOT re-executed!
        assert side_effect_execution_count == 1
        # Step 2 was executed
        assert step2_execution_count == 2
        assert second_result.final_output["status"] == "step2_complete"


def test_no_os_cr_claims_in_code():
    """Verify no misleading 'OS C/R' or 'Crab' claims exist in the state module."""
    state_file = Path("src/state/checkpoint_manager.py").read_text(encoding="utf-8")
    init_file = Path("src/state/__init__.py").read_text(encoding="utf-8")

    forbidden_terms = ["CRIU", "Crab", "OS C/R", "OS-level checkpoint", "checkpoint/restore at the OS"]
    for term in forbidden_terms:
        assert term.lower() not in state_file.lower(), f"Forbidden term '{term}' found in checkpoint_manager.py"
        assert term.lower() not in init_file.lower(), f"Forbidden term '{term}' found in state/__init__.py"
