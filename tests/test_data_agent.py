"""Integration and unit tests for AutonomousDataAgent with multi-turn adaptive reflection.

Verifies that the agent reasons over FinOps AST query rejections (partition pruning omission,
unbounded SELECT * projections, missing LIMIT clauses) and autonomously self-corrects its SQL
trajectories across the Parallax boundary without human intervention.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.athena.duckdb_client import DuckDBClient
from src.engine.data_agent import AutonomousDataAgent
from src.state.checkpoint_manager import ConversationCheckpointManager


@pytest.fixture
def test_duckdb_client() -> DuckDBClient:
    """Fixture providing an in-memory DuckDB client seeded with analytical sales data."""
    client = DuckDBClient(database=":memory:")
    client.con.execute("""
        CREATE TABLE sales (
            order_id INTEGER,
            customer_id VARCHAR,
            amount DOUBLE,
            dt VARCHAR
        );
        INSERT INTO sales VALUES
            (1, 'cust_101', 99.5, '2026-09-01'),
            (2, 'cust_102', 150.0, '2026-09-01'),
            (3, 'cust_103', 45.2, '2026-09-02'),
            (4, 'cust_104', 210.0, '2026-09-03');
    """)
    yield client
    client.close()


@pytest.mark.asyncio
async def test_data_agent_zero_shot_success(test_duckdb_client: DuckDBClient):
    """Ensure a fully-compliant query succeeds on Turn 1 without requiring reflection."""
    agent = AutonomousDataAgent(
        backend="duckdb",
        duckdb_client=test_duckdb_client,
        max_reflection_turns=3,
    )

    sql = "SELECT order_id, amount FROM sales WHERE dt = '2026-09-01' LIMIT 10;"
    summary = await agent.run(
        goal="Retrieve sales orders for September 1st, 2026",
        initial_sql=sql,
        required_partition_keys=["dt"],
    )

    assert summary.success is True
    assert summary.turns == 1
    assert summary.error is None
    assert summary.result_data.get("row_count") == 2
    assert len(summary.thinking_log) == 2
    assert "Initial Formulation" in summary.thinking_log[0]
    assert "Success" in summary.thinking_log[1]


@pytest.mark.asyncio
async def test_data_agent_reflects_missing_partition(test_duckdb_client: DuckDBClient):
    """Ensure agent reflects on missing partition error and self-corrects on Turn 2."""
    agent = AutonomousDataAgent(
        backend="duckdb",
        duckdb_client=test_duckdb_client,
        max_reflection_turns=3,
    )

    # Initial SQL violates partition requirement (missing WHERE dt = ...)
    bad_sql = "SELECT order_id, amount FROM sales LIMIT 10;"
    summary = await agent.run(
        goal="Fetch order transactions",
        initial_sql=bad_sql,
        required_partition_keys=["dt"],
    )

    assert summary.success is True
    assert summary.turns == 2
    assert summary.error is None
    assert summary.result_data.get("row_count") == 2
    assert "dt = '2026-09-01'" in summary.final_sql
    # Reflection thought logged
    assert any("Reflection: The query was rejected because it omits partition pruning" in t for t in summary.thinking_log)


@pytest.mark.asyncio
async def test_data_agent_reflects_unbounded_select(test_duckdb_client: DuckDBClient):
    """Ensure agent reflects on unbounded SELECT * and self-corrects with explicit projection."""
    agent = AutonomousDataAgent(
        backend="duckdb",
        duckdb_client=test_duckdb_client,
        max_reflection_turns=3,
    )

    # Initial SQL violates projection requirement (SELECT *)
    bad_sql = "SELECT * FROM sales WHERE dt = '2026-09-01' LIMIT 10;"
    summary = await agent.run(
        goal="Fetch all columns from sales",
        initial_sql=bad_sql,
        required_partition_keys=["dt"],
    )

    assert summary.success is True
    assert summary.turns == 2
    assert summary.error is None
    assert summary.result_data.get("row_count") == 2
    assert "order_id, amount" in summary.final_sql
    assert any("unbounded 'SELECT *' projection" in t for t in summary.thinking_log)


@pytest.mark.asyncio
async def test_data_agent_reflects_multi_stage_rejection(test_duckdb_client: DuckDBClient):
    """Ensure agent recovers from compound errors across multiple reflection turns."""
    agent = AutonomousDataAgent(
        backend="duckdb",
        duckdb_client=test_duckdb_client,
        max_reflection_turns=3,
    )

    # Violates BOTH partition pruning and column projection
    compound_bad_sql = "SELECT * FROM sales LIMIT 10;"
    summary = await agent.run(
        goal="Fetch raw sales records",
        initial_sql=compound_bad_sql,
        required_partition_keys=["dt"],
    )

    assert summary.success is True
    assert summary.turns == 3
    assert summary.error is None
    assert summary.result_data.get("row_count") == 2
    assert "dt = '2026-09-01'" in summary.final_sql
    assert "order_id, amount" in summary.final_sql


@pytest.mark.asyncio
async def test_data_agent_terminal_failure_when_turns_exhausted(test_duckdb_client: DuckDBClient):
    """Ensure agent reports failure when max reflection turns are reached without resolution."""
    agent = AutonomousDataAgent(
        backend="duckdb",
        duckdb_client=test_duckdb_client,
        max_reflection_turns=2,
    )

    # Non-existent table cannot be resolved by standard reflection heuristics
    invalid_sql = "SELECT order_id, amount FROM non_existent_table WHERE dt = '2026-09-01' LIMIT 10;"
    summary = await agent.run(
        goal="Query missing table",
        initial_sql=invalid_sql,
        required_partition_keys=["dt"],
    )

    assert summary.success is False
    assert summary.turns == 2
    assert summary.error is not None
    assert "Catalog Error" in summary.error or "Table with name non_existent_table does not exist" in summary.error
    assert summary.result_data == {}


@pytest.mark.asyncio
async def test_data_agent_with_checkpoint_manager(tmp_path: Path, test_duckdb_client: DuckDBClient):
    """Ensure agent execution properly creates and updates state checkpoints."""
    ckpt_mgr = ConversationCheckpointManager(storage_dir=str(tmp_path))
    agent = AutonomousDataAgent(
        backend="duckdb",
        duckdb_client=test_duckdb_client,
        checkpoint_manager=ckpt_mgr,
        max_reflection_turns=2,
    )

    sql = "SELECT order_id, amount FROM sales WHERE dt = '2026-09-01' LIMIT 5;"
    summary = await agent.run(
        goal="Query with checkpointing",
        initial_sql=sql,
        required_partition_keys=["dt"],
    )

    assert summary.success is True
    # Verify checkpoint files were written to disk
    checkpoints = list(tmp_path.glob("*.json"))
    assert len(checkpoints) >= 1
