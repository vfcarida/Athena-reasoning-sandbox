"""Tests for ENG-02: unified entry point and CLI.

Verifies that:
  - ``--list`` prints all available demo keys and exits with code 0.
  - ``--demo agent`` runs the AutonomousDataAgent demo without error.
  - ``--demo retrieval`` runs the RetrievalSearch demo without error.
  - ``--demo checkpoint`` runs the ConversationCheckpointManager demo without error.
  - Running without ``--demo`` executes all Tier-1 demos.
  - ``--demo ml`` falls back gracefully when PyTorch is not installed.
  - A demo that raises an unexpected exception returns exit code 1 and
    does not crash the entire entry-point.
  - The ``main()`` function returns 0 on success and 1 on partial failure.

All tests are Lane-1 (offline, deterministic, no external services).
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from src.main import DEMO_MAP, main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """Run main() with the given argv and return (exit_code, stdout, stderr).

    Captures output within this call so tests can inspect it from the returned
    values without a second ``capsys.readouterr()`` that would return empty strings.
    """
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---------------------------------------------------------------------------
# ENG-02-A: --list flag
# ---------------------------------------------------------------------------

class TestEng02List:
    """Verify --list prints all demo keys and exits cleanly."""

    def test_list_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _out, _ = _run(["--list"], capsys)
        assert code == 0

    def test_list_contains_all_demo_keys(self, capsys: pytest.CaptureFixture[str]) -> None:
        _code, out, _ = _run(["--list"], capsys)
        for key in DEMO_MAP:
            assert key in out, f"Key '{key}' not found in --list output"

    def test_list_contains_descriptions(self, capsys: pytest.CaptureFixture[str]) -> None:
        _code, out, _ = _run(["--list"], capsys)
        # At least one description keyword must appear
        assert "offline" in out or "requires" in out


# ---------------------------------------------------------------------------
# ENG-02-B: Individual demo selection
# ---------------------------------------------------------------------------

class TestEng02DemoAgent:
    """Verify --demo agent runs without crashing."""

    def test_agent_demo_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _out, _ = _run(["--demo", "agent"], capsys)
        assert code == 0

    def test_agent_demo_prints_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        _code, out, _ = _run(["--demo", "agent"], capsys)
        assert "AutonomousDataAgent" in out or "success" in out.lower() or "\u2713" in out


class TestEng02DemoRetrieval:
    """Verify --demo retrieval runs without crashing."""

    def test_retrieval_demo_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _, _ = _run(["--demo", "retrieval"], capsys)
        assert code == 0

    def test_retrieval_demo_prints_hits(self, capsys: pytest.CaptureFixture[str]) -> None:
        _code, out, _ = _run(["--demo", "retrieval"], capsys)
        assert "Rank" in out or "hits" in out.lower() or "\u2713" in out


class TestEng02DemoCheckpoint:
    """Verify --demo checkpoint runs without crashing."""

    def test_checkpoint_demo_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _, _ = _run(["--demo", "checkpoint"], capsys)
        assert code == 0

    def test_checkpoint_demo_prints_state_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        _code, out, _ = _run(["--demo", "checkpoint"], capsys)
        assert "Checkpoint" in out or "step" in out.lower() or "\u2713" in out


class TestEng02DemoMcp:
    """Verify --demo mcp runs without crashing."""

    def test_mcp_demo_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _, _ = _run(["--demo", "mcp"], capsys)
        assert code == 0

    def test_mcp_demo_prints_handshake_and_tool_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        _code, out, _ = _run(["--demo", "mcp"], capsys)
        assert "Model Context Protocol" in out or "MCP Server simulation" in out
        assert "duckdb_query" in out
        assert "\u2713" in out


# ---------------------------------------------------------------------------
# ENG-02-C: ML demo graceful skip when torch absent
# ---------------------------------------------------------------------------

class TestEng02DemoMl:
    """Verify --demo ml gracefully warns if PyTorch is unavailable."""

    def test_ml_demo_warns_without_torch(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Simulate missing torch by temporarily hiding it from the import system."""
        with patch.dict(sys.modules, {"torch": None}):
            code, _out, _ = _run(["--demo", "ml"], capsys)
        # Should exit cleanly (exit code 0) even when torch is unavailable
        assert code == 0


# ---------------------------------------------------------------------------
# ENG-02-D: Default run (no --demo flag) executes all Tier-1 demos
# ---------------------------------------------------------------------------

class TestEng02DefaultRun:
    """Verify that running without --demo executes agent, retrieval, and checkpoint."""

    def test_default_run_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, _, _ = _run([], capsys)
        assert code == 0

    def test_default_run_covers_all_tier1_demos(self, capsys: pytest.CaptureFixture[str]) -> None:
        _code, out, _ = _run([], capsys)
        # All three offline demo headers must appear in output
        assert "AutonomousDataAgent" in out
        assert "Retrieval" in out
        assert "Checkpoint" in out


# ---------------------------------------------------------------------------
# ENG-02-E: Error isolation — a failing demo returns exit code 1
# ---------------------------------------------------------------------------

class TestEng02ErrorIsolation:
    """Verify that a crashing demo sets exit code 1 but does not propagate the exception."""

    def test_failing_demo_returns_exit_code_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Inject a crashing demo function and verify main() returns 1."""
        def _crashing_demo() -> None:
            raise RuntimeError("Intentional crash for testing.")

        with patch.dict("src.main.DEMO_MAP", {"agent": ("crashed", _crashing_demo)}, clear=False):
            code, _, _ = _run(["--demo", "agent"], capsys)
        assert code == 1

    def test_failing_demo_prints_error_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        def _crashing_demo() -> None:
            raise ValueError("Simulated demo failure.")

        with patch.dict("src.main.DEMO_MAP", {"agent": ("crashed", _crashing_demo)}, clear=False):
            _code, out, _ = _run(["--demo", "agent"], capsys)
        assert "\u2717" in out or "error" in out.lower() or "Simulated" in out


# ---------------------------------------------------------------------------
# ENG-02-F: DEMO_MAP completeness
# ---------------------------------------------------------------------------

class TestEng02DemoMap:
    """Verify DEMO_MAP structure is correct."""

    def test_all_expected_keys_present(self) -> None:
        assert "agent" in DEMO_MAP
        assert "retrieval" in DEMO_MAP
        assert "checkpoint" in DEMO_MAP
        assert "ml" in DEMO_MAP

    def test_all_entries_have_description_and_callable(self) -> None:
        for key, (desc, fn) in DEMO_MAP.items():
            assert isinstance(desc, str) and len(desc) > 0, f"Missing description for '{key}'"
            assert callable(fn), f"Demo function for '{key}' is not callable"
