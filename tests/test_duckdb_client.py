"""Unit and integration tests for local DuckDB analytical engine with FinOps AST query governance.

Verifies that DuckDBClient strictly enforces partition pruning and explicit projections
while providing reliable local SQL data lake query execution without AWS dependencies.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import pytest

from src.athena.duckdb_client import DuckDBClient
from src.athena.query_guard import UnboundedSelectError, UnpartitionedQueryError


@pytest.fixture
def duckdb_client() -> DuckDBClient:
    """Fixture providing an in-memory DuckDB client populated with test data."""
    client = DuckDBClient(database=":memory:")
    # Seed sample sales data
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


def test_duckdb_client_unpartitioned_query_rejected(duckdb_client: DuckDBClient):
    """Ensure queries lacking mandatory partition filters in WHERE are rejected."""
    query = "SELECT order_id, amount FROM sales LIMIT 10;"
    with pytest.raises(UnpartitionedQueryError, match="Query lacks a WHERE clause"):
        duckdb_client.execute_query(query, required_partition_keys=["dt"])


def test_duckdb_client_unbounded_select_rejected(duckdb_client: DuckDBClient):
    """Ensure unbounded SELECT * queries are rejected by AST guard."""
    query = "SELECT * FROM sales WHERE dt = '2026-09-01' LIMIT 10;"
    with pytest.raises(UnboundedSelectError, match="Unbounded 'SELECT \\*' queries are prohibited"):
        duckdb_client.execute_query(query, required_partition_keys=["dt"])


def test_duckdb_client_missing_limit_rejected(duckdb_client: DuckDBClient):
    """Ensure queries omitting a LIMIT clause are rejected."""
    query = "SELECT order_id, amount FROM sales WHERE dt = '2026-09-01';"
    with pytest.raises(UnboundedSelectError, match="Query lacks an explicit LIMIT clause"):
        duckdb_client.execute_query(query, required_partition_keys=["dt"], enforce_limit=True)


def test_duckdb_client_valid_query_execution(duckdb_client: DuckDBClient):
    """Verify that a compliant query executes successfully and returns expected data."""
    query = "SELECT order_id, amount FROM sales WHERE dt = '2026-09-01' LIMIT 10;"
    res = duckdb_client.execute_query(query, required_partition_keys=["dt"])

    assert res["status"] == "SUCCEEDED"
    assert res["mode"] == "DUCKDB_LOCAL"
    assert res["row_count"] == 2
    assert res["columns"] == ["order_id", "amount"]
    assert len(res["rows"]) == 2
    assert res["rows"][0][0] == 1


def test_duckdb_client_dry_run(duckdb_client: DuckDBClient):
    """Verify that dry_run validates AST without executing query."""
    query = "SELECT order_id, amount FROM sales WHERE dt = '2026-09-01' LIMIT 10;"
    res = duckdb_client.execute_query(query, required_partition_keys=["dt"], dry_run=True)

    assert res["status"] == "SUCCEEDED"
    assert res["mode"] == "DRY_RUN"
    assert res["row_count"] == 0


@pytest.mark.asyncio
async def test_duckdb_client_async_execution(duckdb_client: DuckDBClient):
    """Verify that async_execute_query runs non-blocking in an asyncio loop."""
    query = "SELECT customer_id, amount FROM sales WHERE dt = '2026-09-02' LIMIT 5;"
    res = await duckdb_client.async_execute_query(query, required_partition_keys=["dt"])

    assert res["status"] == "SUCCEEDED"
    assert res["row_count"] == 1
    assert res["rows"][0][0] == "cust_103"


def test_duckdb_client_parquet_integration():
    """Verify table creation and query execution directly from a Parquet file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_file = Path(tmpdir) / "events.parquet"

        # Create a sample parquet file using standalone duckdb
        temp_con = duckdb.connect()
        parquet_str = str(parquet_file).replace("\\", "/")
        temp_con.execute(f"""
            COPY (
                SELECT 101 AS event_id, 'click' AS action, '2026-09-01' AS ds
                UNION ALL
                SELECT 102 AS event_id, 'view' AS action, '2026-09-01' AS ds
            ) TO '{parquet_str}' (FORMAT PARQUET);
        """)
        temp_con.close()

        # Load into DuckDBClient
        client = DuckDBClient(database=":memory:")
        client.create_table_from_parquet("events", parquet_file)

        # Query with AST validation
        query = "SELECT event_id, action FROM events WHERE ds = '2026-09-01' LIMIT 5;"
        res = client.execute_query(query, required_partition_keys=["ds"])

        assert res["status"] == "SUCCEEDED"
        assert res["row_count"] == 2
        assert res["columns"] == ["event_id", "action"]
        client.close()
