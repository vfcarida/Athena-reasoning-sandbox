"""Conversation State Checkpoint & Idempotent Execution Management.

Provides structured, application-level conversation and turn-state persistence.
This module records dialogue turns, working memory, and a side-effect ledger to
enable safe resumption of interrupted workflows.

Important:
This module manages application-level state (conversation history, tool memory,
and effect ledgers) serialized as JSON. It does NOT perform low-level kernel
or system execution state snapshotting. External side effects (e.g., database writes,
network calls) are not rolled back automatically upon restore; instead, they
are tracked via idempotency keys in the effect ledger so that resumed executions
can avoid re-executing already-completed side effects.

Algorithmic Complexity:
    - Checkpoint Save/Restore: O(S) where S is size of serializable turn history state.
    - Idempotency lookup: O(1) hash table lookup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_idempotency_key(action_type: str, tool_name: str, tool_args: dict[str, Any]) -> str:
    """Compute a deterministic hash for a tool action and its arguments.

    Args:
        action_type: Category of action (e.g. 'athena_query', 'sandbox_execute').
        tool_name: Specific tool name invoked.
        tool_args: Dictionary of arguments supplied to the tool.

    Returns:
        Hexadecimal SHA-256 digest string representing the unique action invocation.
    """
    try:
        serialized_args = json.dumps(tool_args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized_args = str(sorted(tool_args.items()))

    raw_payload = f"{action_type}:{tool_name}:{serialized_args}"
    return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()


@dataclass
class ConversationStateCheckpoint:
    """Serializable snapshot container for conversation turn state and side-effect ledger.

    Attributes:
        checkpoint_id: Unique string checkpoint identifier.
        step_index: Current turn or step sequence number reached.
        turn_history: List of conversation turn records (prompts, think traces, observations).
        tool_memory: Key-value dictionary of persistent tool execution artifacts.
        effect_ledger: Map of idempotency keys to their recorded execution results.
        schema_version: Version identifier for backwards-compatible migrations (default 2).
        timestamp: Unix epoch timestamp when checkpoint was created.
    """

    checkpoint_id: str
    step_index: int
    turn_history: list[dict[str, Any]] = field(default_factory=list)
    tool_memory: dict[str, Any] = field(default_factory=dict)
    effect_ledger: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema_version: int = 2
    timestamp: float = field(default_factory=time.time)

    def has_effect(self, key: str) -> bool:
        """Check whether an effect with the given idempotency key was already recorded."""
        return key in self.effect_ledger

    def record_effect(self, key: str, effect_data: dict[str, Any]) -> None:
        """Record a completed side effect by its idempotency key.

        Args:
            key: Deterministic idempotency key for the action.
            effect_data: Output data or metadata from the executed action.
        """
        self.effect_ledger[key] = {
            "recorded_at": time.time(),
            "data": effect_data,
        }

    def get_effect(self, key: str) -> dict[str, Any] | None:
        """Retrieve the recorded result of a prior effect, or None if not recorded."""
        entry = self.effect_ledger.get(key)
        if entry is not None:
            return entry.get("data")
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a JSON-compatible dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationStateCheckpoint:
        """Deserialize dictionary payload into ConversationStateCheckpoint with migration support.

        Gracefully handles legacy v1 format (containing 'os_effects' instead of 'effect_ledger').
        """
        data_copy = dict(data)

        # Handle legacy schema v1 migration
        schema_ver = data_copy.get("schema_version", 1)
        data_copy["schema_version"] = schema_ver

        if "effect_ledger" not in data_copy:
            data_copy["effect_ledger"] = {}

        # Remove legacy os_effects if present and not in dataclass fields
        if "os_effects" in data_copy:
            legacy_effects = data_copy.pop("os_effects", [])
            # If any legacy string effects exist, store them in the ledger
            for effect in legacy_effects:
                if isinstance(effect, str) and effect not in data_copy["effect_ledger"]:
                    data_copy["effect_ledger"][effect] = {"data": {"legacy": effect}}

        # Ensure valid arguments for cls constructor
        valid_fields = {
            "checkpoint_id",
            "step_index",
            "turn_history",
            "tool_memory",
            "effect_ledger",
            "schema_version",
            "timestamp",
        }
        filtered_data = {k: v for k, v in data_copy.items() if k in valid_fields}

        return cls(**filtered_data)


# Backward-compatibility alias
CheckpointState = ConversationStateCheckpoint


class ConversationCheckpointManager:
    """Manager for saving, restoring, and querying conversation checkpoints.

    Attributes:
        storage_dir: Local directory path for checkpoint persistence.
    """

    def __init__(self, storage_dir: str = "checkpoints") -> None:
        """Initialize Conversation Checkpoint Manager.

        Args:
            storage_dir: Path to directory storing JSON checkpoints.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory_cache: dict[str, ConversationStateCheckpoint] = {}
        logger.info("ConversationCheckpointManager initialized (storage_dir='%s')", self.storage_dir)

    def create_checkpoint(
        self,
        checkpoint_id: str,
        step_index: int,
        turn_history: list[dict[str, Any]],
        tool_memory: dict[str, Any] | None = None,
        effect_ledger: dict[str, dict[str, Any]] | None = None,
        os_effects: list[str] | None = None,
    ) -> ConversationStateCheckpoint:
        """Create and persist a new conversation state snapshot.

        Args:
            checkpoint_id: Identifier string.
            step_index: Current step index.
            turn_history: History of interaction turns.
            tool_memory: Persistent tool state dictionary.
            effect_ledger: Optional map of recorded side-effects.
            os_effects: Deprecated parameter kept for backward compatibility.

        Returns:
            Saved ConversationStateCheckpoint instance.
        """
        ledger = effect_ledger.copy() if effect_ledger else {}
        if os_effects:
            for eff in os_effects:
                if eff not in ledger:
                    ledger[eff] = {"data": {"legacy": eff}}

        state = ConversationStateCheckpoint(
            checkpoint_id=checkpoint_id,
            step_index=step_index,
            turn_history=turn_history,
            tool_memory=tool_memory or {},
            effect_ledger=ledger,
            schema_version=2,
            timestamp=time.time(),
        )

        self.save_checkpoint(state)
        return state

    def save_checkpoint(self, checkpoint: ConversationStateCheckpoint) -> None:
        """Persist a ConversationStateCheckpoint to disk and in-memory cache.

        Args:
            checkpoint: The checkpoint instance to save.
        """
        self._in_memory_cache[checkpoint.checkpoint_id] = checkpoint
        file_path = self.storage_dir / f"{checkpoint.checkpoint_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        logger.info("Persisted checkpoint '%s' (step=%d) to %s", checkpoint.checkpoint_id, checkpoint.step_index, file_path)

    def restore_checkpoint(self, checkpoint_id: str) -> ConversationStateCheckpoint | None:
        """Restore a previously saved checkpoint by ID.

        Args:
            checkpoint_id: Identifier string.

        Returns:
            Restored ConversationStateCheckpoint or None if not found.
        """
        if checkpoint_id in self._in_memory_cache:
            return self._in_memory_cache[checkpoint_id]

        file_path = self.storage_dir / f"{checkpoint_id}.json"
        if not file_path.exists():
            logger.warning("Checkpoint file '%s' not found.", file_path)
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        state = ConversationStateCheckpoint.from_dict(data)
        self._in_memory_cache[checkpoint_id] = state
        logger.info("Restored checkpoint '%s' (step=%d) from %s", checkpoint_id, state.step_index, file_path)
        return state

    def has_effect(self, checkpoint_id: str, key: str) -> bool:
        """Check if an effect key has already been executed in the specified checkpoint."""
        ckpt = self.restore_checkpoint(checkpoint_id)
        if ckpt is None:
            return False
        return ckpt.has_effect(key)

    def record_effect(self, checkpoint_id: str, key: str, effect_data: dict[str, Any]) -> None:
        """Record an effect in the specified checkpoint and persist it."""
        ckpt = self.restore_checkpoint(checkpoint_id)
        if ckpt is None:
            ckpt = self.create_checkpoint(checkpoint_id=checkpoint_id, step_index=0, turn_history=[])
        ckpt.record_effect(key, effect_data)
        self.save_checkpoint(ckpt)


# Backward-compatibility alias
CheckpointManager = ConversationCheckpointManager
