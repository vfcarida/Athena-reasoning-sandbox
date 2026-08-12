"""DeltaState / Crab Inspired Checkpoint & Restore (C/R) State Management Engine.

Preserves long-horizon reasoning turn logs, OS side-effects, tool execution memory,
and context states to allow efficient resume capability without re-executing previous turns.

Algorithmic Complexity:
    - Checkpoint Save/Restore: O(S) where S is size of serializable turn history state.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Serializable snapshot container for agent trajectory state.

    Attributes:
        checkpoint_id: Unique string checkpoint identifier.
        step_index: Current turn or step sequence number.
        turn_history: List of conversation turn records (prompts, think traces, observations).
        tool_memory: Key-value dictionary of persistent tool execution artifacts.
        os_effects: List of recorded filesystem/side-effects.
        timestamp: Unix epoch timestamp when checkpoint was created.
    """
    checkpoint_id: str
    step_index: int
    turn_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_memory: Dict[str, Any] = field(default_factory=dict)
    os_effects: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointState:
        """Deserialize dictionary payload into CheckpointState instance."""
        return cls(**data)


class CheckpointManager:
    """Manager for saving, restoring, and diffing agent trajectory checkpoints.

    Attributes:
        storage_dir: Local directory path for checkpoint persistence.
    """

    def __init__(self, storage_dir: str = "checkpoints") -> None:
        """Initialize Checkpoint Manager.

        Args:
            storage_dir: Path to directory storing JSON checkpoints.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory_cache: Dict[str, CheckpointState] = {}
        logger.info("CheckpointManager initialized (storage_dir='%s')", self.storage_dir)

    def create_checkpoint(
        self,
        checkpoint_id: str,
        step_index: int,
        turn_history: List[Dict[str, Any]],
        tool_memory: Optional[Dict[str, Any]] = None,
        os_effects: Optional[List[str]] = None,
    ) -> CheckpointState:
        """Create and persist a new state snapshot checkpoint.

        Args:
            checkpoint_id: Identifier string.
            step_index: Current step index.
            turn_history: History of interaction turns.
            tool_memory: Persistent tool state dictionary.
            os_effects: Recorded side-effects list.

        Returns:
            Saved CheckpointState instance.
        """
        state = CheckpointState(
            checkpoint_id=checkpoint_id,
            step_index=step_index,
            turn_history=turn_history,
            tool_memory=tool_memory or {},
            os_effects=os_effects or [],
            timestamp=time.time(),
        )

        # Cache in memory
        self._in_memory_cache[checkpoint_id] = state

        # Write to JSON file
        file_path = self.storage_dir / f"{checkpoint_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)

        logger.info("Persisted checkpoint '%s' (step=%d) to %s", checkpoint_id, step_index, file_path)
        return state

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointState]:
        """Restore a previously saved snapshot checkpoint by ID.

        Args:
            checkpoint_id: Identifier string.

        Returns:
            Restored CheckpointState or None if not found.
        """
        if checkpoint_id in self._in_memory_cache:
            return self._in_memory_cache[checkpoint_id]

        file_path = self.storage_dir / f"{checkpoint_id}.json"
        if not file_path.exists():
            logger.warning("Checkpoint file '%s' not found.", file_path)
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        state = CheckpointState.from_dict(data)
        self._in_memory_cache[checkpoint_id] = state
        logger.info("Restored checkpoint '%s' (step=%d) from %s", checkpoint_id, state.step_index, file_path)
        return state
