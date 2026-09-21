"""Unit test suite for AWS Athena FinOps Cost Guard and Execution Hardening.

Verifies:
- sqlglot AST query parsing and validation.
- AST rejection of E2 (b) evasion queries (IS NOT NULL, function-wrapped keys, non-pruning OR, subquery-only keys).
- Acceptance of valid pruning equality, range, IN, and BETWEEN predicates.
- Rejection of unbounded SELECT * and missing LIMIT hygiene checks.
- CTAS query generation and AST validation.
- config.athena_max_scan_bytes wiring to AthenaClient and WorkGroup settings.
- Offline query execution, polling, and DataScannedInBytes measurement via Moto.
- Cancellation and stop_query_execution dispatch on scan bytes cutoff breach and query timeout.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

from src.athena.athena_client import (
    AthenaClient,
    CTASPipeline,
    QueryExecutionError,
    QueryTimeoutError,
    ScanBytesLimitExceededError,
)
from src.athena.query_guard import (
    AthenaQueryGuard,
    UnboundedSelectError,
    UnpartitionedQueryError,
)
from src.utils.config import AppConfig, config

# ==============================================================================
# 1. AST Partition Filter Guard Tests (Valid Pruning Predicates)
# ==============================================================================


def test_partition_filter_guard_success_equality():
    """Verify partition guard allows queries with equality partition filters."""
    guard = AthenaQueryGuard()
    query = "SELECT user_id, event_type FROM user_events WHERE dt = '2026-08-12' LIMIT 100;"
    guard.validate_partition_filter(query, required_partition_keys=["dt"])


def test_partition_filter_guard_range_and_in_predicates():
    """Verify partition guard allows range (>=, <=, BETWEEN) and list (IN) partition predicates."""
    guard = AthenaQueryGuard()

    # Range with AND
    q1 = "SELECT user_id FROM user_events WHERE dt >= '2026-08-01' AND dt <= '2026-08-31' LIMIT 100;"
    guard.validate_partition_filter(q1, required_partition_keys=["dt"])

    # IN predicate
    q2 = "SELECT user_id FROM user_events WHERE dt IN ('2026-08-01', '2026-08-02') LIMIT 100;"
    guard.validate_partition_filter(q2, required_partition_keys=["dt"])

    # BETWEEN predicate
    q3 = "SELECT user_id FROM user_events WHERE dt BETWEEN '2026-08-01' AND '2026-08-31' LIMIT 100;"
    guard.validate_partition_filter(q3, required_partition_keys=["dt"])

    # Pruning OR branches (both branches filter on partition key)
    q4 = "SELECT user_id FROM user_events WHERE (dt = '2026-08-01' OR dt = '2026-08-02') AND status = 'A' LIMIT 100;"
    guard.validate_partition_filter(q4, required_partition_keys=["dt"])


# ==============================================================================
# 2. E2 (b) Evasion Set Rejections (0 False-Accepts via AST Validation)
# ==============================================================================


def test_partition_filter_guard_missing_where():
    """Verify partition guard rejects queries missing WHERE clause."""
    guard = AthenaQueryGuard()
    query = "SELECT user_id, event_type FROM user_events LIMIT 100;"
    with pytest.raises(UnpartitionedQueryError) as exc_info:
        guard.validate_partition_filter(query, required_partition_keys=["dt"])
    assert "lacks a WHERE clause" in str(exc_info.value)


def test_partition_filter_guard_missing_key():
    """Verify partition guard rejects queries where WHERE clause omits target partition key."""
    guard = AthenaQueryGuard()
    query = "SELECT user_id FROM user_events WHERE status = 'ACTIVE' LIMIT 100;"
    with pytest.raises(UnpartitionedQueryError) as exc_info:
        guard.validate_partition_filter(query, required_partition_keys=["dt"])
    assert "WHERE clause does not filter on required partition keys" in str(exc_info.value)


def test_partition_filter_guard_rejects_is_not_null():
    """Verify AST guard rejects 'dt IS NOT NULL' (previously allowed by regex, causes full scan)."""
    guard = AthenaQueryGuard()
    query = "SELECT user_id FROM user_events WHERE dt IS NOT NULL LIMIT 100;"
    with pytest.raises(UnpartitionedQueryError) as exc_info:
        guard.validate_partition_filter(query, required_partition_keys=["dt"])
    assert "WHERE clause does not filter on required partition keys" in str(exc_info.value)


def test_partition_filter_guard_rejects_function_wrapped_key():
    """Verify AST guard rejects function-wrapped partition keys (e.g. YEAR(dt) = 2026)."""
    guard = AthenaQueryGuard()
    query = "SELECT user_id FROM user_events WHERE YEAR(dt) = 2026 LIMIT 100;"
    with pytest.raises(UnpartitionedQueryError) as exc_info:
        guard.validate_partition_filter(query, required_partition_keys=["dt"])
    assert "WHERE clause does not filter on required partition keys" in str(exc_info.value)


def test_partition_filter_guard_rejects_non_pruning_or():
    """Verify AST guard rejects OR clauses where one branch does not filter on partition key."""
    guard = AthenaQueryGuard()
    query = "SELECT user_id FROM user_events WHERE status = 'ACTIVE' OR dt = '2026-08-12' LIMIT 100;"
    with pytest.raises(UnpartitionedQueryError) as exc_info:
        guard.validate_partition_filter(query, required_partition_keys=["dt"])
    assert "WHERE clause does not filter on required partition keys" in str(exc_info.value)


def test_partition_filter_guard_rejects_subquery_only_key():
    """Verify AST guard rejects partition keys present only inside nested subqueries."""
    guard = AthenaQueryGuard()
    query = (
        "SELECT user_id FROM user_events "
        "WHERE status = 'ACTIVE' AND (SELECT count(*) FROM t WHERE dt = '2026-08-12') > 0 "
        "LIMIT 100;"
    )
    with pytest.raises(UnpartitionedQueryError) as exc_info:
        guard.validate_partition_filter(query, required_partition_keys=["dt"])
    assert "WHERE clause does not filter on required partition keys" in str(exc_info.value)


def test_partition_filter_guard_rejects_tautology():
    """Verify AST guard rejects tautological comparisons like dt = dt."""
    guard = AthenaQueryGuard()
    query = "SELECT user_id FROM user_events WHERE dt = dt LIMIT 100;"
    with pytest.raises(UnpartitionedQueryError) as exc_info:
        guard.validate_partition_filter(query, required_partition_keys=["dt"])
    assert "WHERE clause does not filter on required partition keys" in str(exc_info.value)


# ==============================================================================
# 3. Projection and LIMIT Hygiene Checks
# ==============================================================================


def test_select_star_guard_failure():
    """Verify select star guard blocks SELECT * queries."""
    guard = AthenaQueryGuard()
    query = "SELECT * FROM user_events WHERE dt = '2026-08-12' LIMIT 100;"
    with pytest.raises(UnboundedSelectError) as exc_info:
        guard.validate_projections_and_limits(query)
    assert "SELECT *' queries are prohibited" in str(exc_info.value)


def test_select_table_star_guard_failure():
    """Verify select star guard blocks SELECT t.* table-scoped star projections."""
    guard = AthenaQueryGuard()
    query = "SELECT u.* FROM user_events u WHERE dt = '2026-08-12' LIMIT 100;"
    with pytest.raises(UnboundedSelectError) as exc_info:
        guard.validate_projections_and_limits(query)
    assert "SELECT *' queries are prohibited" in str(exc_info.value)


def test_missing_limit_guard_failure():
    """Verify limit guard blocks queries missing an explicit LIMIT clause."""
    guard = AthenaQueryGuard()
    query = "SELECT user_id, event_type FROM user_events WHERE dt = '2026-08-12';"
    with pytest.raises(UnboundedSelectError) as exc_info:
        guard.validate_projections_and_limits(query, enforce_limit=True)
    assert "lacks an explicit LIMIT clause" in str(exc_info.value)


# ==============================================================================
# 4. CTAS Statement Generation & AST Parsing
# ==============================================================================


def test_ctas_pipeline_statement_generation():
    """Verify CTAS statement generation formats Parquet and Snappy compression properly."""
    inner_sql = "SELECT user_id, count(*) as cnt FROM user_events WHERE dt = '2026-08-12' GROUP BY user_id"
    ctas_sql = CTASPipeline.build_ctas_query(
        target_table="user_events_summary",
        select_query=inner_sql,
        s3_location="s3://my-bucket/parquet-out/",
        output_format="PARQUET",
        compression="SNAPPY",
        partition_by=["dt"],
    )
    assert "CREATE TABLE user_events_summary" in ctas_sql
    assert "format = 'PARQUET'" in ctas_sql
    assert "parquet_compression = 'SNAPPY'" in ctas_sql
    assert "partitioned_by = ARRAY['dt']" in ctas_sql

    # Verify that the generated CTAS query passes full AST validation
    guard = AthenaQueryGuard()
    guard.validate_query(ctas_sql, required_partition_keys=["dt"], enforce_limit=False)


# ==============================================================================
# 5. Configuration & WorkGroup Cutoff Wiring
# ==============================================================================


def test_app_config_validation():
    """Verify AppConfig dynamic settings initialization."""
    cfg = AppConfig()
    assert cfg.aws_region is not None
    assert cfg.athena_max_scan_bytes > 0


def test_athena_max_scan_bytes_wiring():
    """Verify that config.athena_max_scan_bytes is wired into AthenaClient."""
    # Default wires from config
    client_default = AthenaClient()
    assert client_default.max_scan_bytes == config.athena_max_scan_bytes
    assert client_default.workgroup == "primary"

    # Custom override alters the client cutoff and workgroup
    custom_cutoff = 5 * 1024 * 1024 * 1024
    client_custom = AthenaClient(workgroup="finops-capped", max_scan_bytes=custom_cutoff)
    assert client_custom.max_scan_bytes == custom_cutoff
    assert client_custom.workgroup == "finops-capped"


def test_configure_workgroup_cutoff_mock():
    """Verify configure_workgroup_cutoff dispatches update_workgroup or create_workgroup."""
    mock_boto = MagicMock()
    client = AthenaClient(boto_client=mock_boto, workgroup="test-wg", max_scan_bytes=5000)

    # 1. Update workgroup path
    success = client.configure_workgroup_cutoff()
    assert success is True
    mock_boto.update_workgroup.assert_called_once_with(
        WorkGroup="test-wg",
        ConfigurationUpdates={
            "BytesScannedCutoffPerQuery": 5000,
            "EnforceWorkGroupConfiguration": True,
        },
    )

    # 2. Fallback to create_workgroup if update fails
    mock_boto.update_workgroup.side_effect = Exception("WorkGroup not found")
    success_create = client.configure_workgroup_cutoff(cutoff_bytes=8000)
    assert success_create is True
    mock_boto.create_workgroup.assert_called_once_with(
        Name="test-wg",
        Configuration={
            "BytesScannedCutoffPerQuery": 8000,
            "EnforceWorkGroupConfiguration": True,
        },
    )


# ==============================================================================
# 6. Dry Run & Offline Moto Query Execution & Measurement
# ==============================================================================


def test_athena_client_dry_run_execution():
    """Verify AthenaClient validates query and returns dry-run execution status with wired limits."""
    client = AthenaClient(s3_staging_dir="s3://my-staging-bucket/")
    sql = "SELECT user_id, timestamp FROM logs WHERE dt = '2026-08-12' LIMIT 500;"
    res = client.execute_query(sql, required_partition_keys=["dt"], dry_run=True)
    assert res["status"] == "SUCCEEDED"
    assert res["mode"] == "DRY_RUN"
    assert "dry-run-" in res["query_execution_id"]
    assert res["workgroup"] == "primary"
    assert res["max_scan_bytes"] == config.athena_max_scan_bytes
    assert res["data_scanned_in_bytes"] == 0


@mock_aws
def test_moto_athena_execution_and_polling():
    """Verify offline query dispatch and measurement via Moto without real AWS calls."""
    boto_client = boto3.client("athena", region_name="us-east-1")
    client = AthenaClient(
        database="analytics_db",
        s3_staging_dir="s3://test-staging-bucket/out/",
        workgroup="primary",
        boto_client=boto_client,
    )

    sql = "SELECT user_id, action FROM events WHERE dt = '2026-08-12' LIMIT 50;"

    # 1. Asynchronous dispatch (wait=False)
    res_async = client.execute_query(sql, required_partition_keys=["dt"], wait=False)
    assert res_async["status"] == "SUBMITTED"
    assert res_async["mode"] == "AWS_BOTO3"
    assert res_async["workgroup"] == "primary"
    assert "query_execution_id" in res_async

    # 2. Synchronous / polled execution (wait=True)
    res_sync = client.execute_query(sql, required_partition_keys=["dt"], wait=True, poll_interval=0.1)
    assert res_sync["status"] == "SUCCEEDED"
    assert res_sync["mode"] == "AWS_BOTO3"
    assert "data_scanned_in_bytes" in res_sync
    assert res_sync["workgroup"] == "primary"


# ==============================================================================
# 7. Polling Cancellation on Scan Bytes Limit Breach & Timeout
# ==============================================================================


def test_athena_client_scan_limit_exceeded_cancels_query():
    """Verify client calls stop_query_execution and raises ScanBytesLimitExceededError on scan breach."""
    mock_boto = MagicMock()
    mock_boto.start_query_execution.return_value = {"QueryExecutionId": "query-breach-123"}
    # Mock status returning scanned bytes exceeding 1000 limit
    mock_boto.get_query_execution.return_value = {
        "QueryExecution": {
            "QueryExecutionId": "query-breach-123",
            "Status": {"State": "RUNNING"},
            "Statistics": {"DataScannedInBytes": 5000},
            "WorkGroup": "primary",
        }
    }

    client = AthenaClient(boto_client=mock_boto, max_scan_bytes=1000)
    sql = "SELECT user_id FROM logs WHERE dt = '2026-08-12' LIMIT 10;"

    with pytest.raises(ScanBytesLimitExceededError) as exc_info:
        client.execute_query(sql, required_partition_keys=["dt"], wait=True, poll_interval=0.01)

    assert "exceeding max allowed limit of 1000 bytes" in str(exc_info.value)
    # Assert stop_query_execution was called with the breached query ID
    mock_boto.stop_query_execution.assert_called_once_with(QueryExecutionId="query-breach-123")


def test_athena_client_timeout_cancels_query():
    """Verify client calls stop_query_execution and raises QueryTimeoutError on timeout."""
    mock_boto = MagicMock()
    mock_boto.start_query_execution.return_value = {"QueryExecutionId": "query-timeout-456"}
    # Mock status remaining RUNNING indefinitely
    mock_boto.get_query_execution.return_value = {
        "QueryExecution": {
            "QueryExecutionId": "query-timeout-456",
            "Status": {"State": "RUNNING"},
            "Statistics": {"DataScannedInBytes": 10},
            "WorkGroup": "primary",
        }
    }

    client = AthenaClient(boto_client=mock_boto, max_scan_bytes=100000)
    sql = "SELECT user_id FROM logs WHERE dt = '2026-08-12' LIMIT 10;"

    with pytest.raises(QueryTimeoutError) as exc_info:
        client.execute_query(
            sql,
            required_partition_keys=["dt"],
            wait=True,
            poll_interval=0.05,
            timeout_seconds=0.1,
        )

    assert "timed out after" in str(exc_info.value)
    # Assert stop_query_execution was called with the timed out query ID
    mock_boto.stop_query_execution.assert_called_once_with(QueryExecutionId="query-timeout-456")


def test_athena_client_query_failure_raises_error():
    """Verify client raises QueryExecutionError if query enters FAILED state."""
    mock_boto = MagicMock()
    mock_boto.start_query_execution.return_value = {"QueryExecutionId": "query-failed-789"}
    mock_boto.get_query_execution.return_value = {
        "QueryExecution": {
            "QueryExecutionId": "query-failed-789",
            "Status": {
                "State": "FAILED",
                "StateChangeReason": "SYNTAX_ERROR: Line 1, Col 1: Table not found",
            },
            "Statistics": {"DataScannedInBytes": 0},
        }
    }

    client = AthenaClient(boto_client=mock_boto)
    sql = "SELECT user_id FROM logs WHERE dt = '2026-08-12' LIMIT 10;"

    with pytest.raises(QueryExecutionError) as exc_info:
        client.execute_query(sql, required_partition_keys=["dt"], wait=True, poll_interval=0.01)

    assert "Table not found" in str(exc_info.value)


def test_query_guard_invalid_sql_parse_error():
    """Verify query guard raises UnpartitionedQueryError on invalid SQL syntax."""
    guard = AthenaQueryGuard()
    with pytest.raises(UnpartitionedQueryError) as exc_info:
        guard.validate_query("SELECT FROM WHERE ;")
    assert "Failed to parse SQL query" in str(exc_info.value)


def test_partition_filter_guard_union_queries():
    """Verify query guard extracts and checks all branches in UNION queries."""
    guard = AthenaQueryGuard()

    # Both branches partitioned
    q_valid = (
        "SELECT user_id FROM t1 WHERE dt = '2026-08-12' "
        "UNION ALL "
        "SELECT user_id FROM t2 WHERE dt = '2026-08-12' LIMIT 10;"
    )
    guard.validate_partition_filter(q_valid, required_partition_keys=["dt"])

    # One branch unpartitioned
    q_invalid = (
        "SELECT user_id FROM t1 WHERE dt = '2026-08-12' "
        "UNION ALL "
        "SELECT user_id FROM t2 WHERE status = 'ACTIVE' LIMIT 10;"
    )
    with pytest.raises(UnpartitionedQueryError) as exc_info:
        guard.validate_partition_filter(q_invalid, required_partition_keys=["dt"])
    assert "WHERE clause does not filter on required partition keys" in str(exc_info.value)


def test_athena_client_execute_ctas_dry_run():
    """Verify execute_ctas_ingestion generates and validates CTAS SQL in dry-run mode."""
    client = AthenaClient(s3_staging_dir="s3://test-bucket/staging/")
    res = client.execute_ctas_ingestion(
        target_table="daily_summary",
        select_query="SELECT user_id, count(*) as c FROM events WHERE dt = '2026-08-12' GROUP BY user_id",
        s3_location="s3://test-bucket/parquet/",
        partition_by=["dt"],
        dry_run=True,
    )
    assert res["status"] == "SUCCEEDED"
    assert res["mode"] == "DRY_RUN"
    assert "CREATE TABLE daily_summary" in res["sql"]


def test_athena_client_wait_for_completion_dry_run():
    """Verify wait_for_completion immediately returns succeeded for dry-run IDs."""
    client = AthenaClient()
    res = client.wait_for_completion("dry-run-99999")
    assert res["status"] == "SUCCEEDED"
    assert res["mode"] == "DRY_RUN"
    assert res["data_scanned_in_bytes"] == 0


def test_configure_workgroup_cutoff_without_boto():
    """Verify configure_workgroup_cutoff returns False when no boto client is configured."""
    client = AthenaClient(boto_client=None)
    assert client.configure_workgroup_cutoff() is False


def test_athena_client_stop_query_exception_handled():
    """Verify that exceptions raised by stop_query_execution are logged and do not mask breach error."""
    mock_boto = MagicMock()
    mock_boto.start_query_execution.return_value = {"QueryExecutionId": "query-stop-fail"}
    mock_boto.get_query_execution.return_value = {
        "QueryExecution": {
            "QueryExecutionId": "query-stop-fail",
            "Status": {"State": "RUNNING"},
            "Statistics": {"DataScannedInBytes": 999999},
            "WorkGroup": "primary",
        }
    }
    mock_boto.stop_query_execution.side_effect = Exception("AWS StopQueryExecution internal error")

    client = AthenaClient(boto_client=mock_boto, max_scan_bytes=100)
    sql = "SELECT user_id FROM logs WHERE dt = '2026-08-12' LIMIT 10;"

    with pytest.raises(ScanBytesLimitExceededError) as exc_info:
        client.execute_query(sql, required_partition_keys=["dt"], wait=True, poll_interval=0.01)

    assert "exceeding max allowed limit of 100 bytes" in str(exc_info.value)


@pytest.mark.asyncio
async def test_athena_client_async_execute_query():
    """Verify async_execute_query completes and returns dry-run response asynchronously."""
    client = AthenaClient(s3_staging_dir="s3://my-staging-bucket/")
    sql = "SELECT user_id, timestamp FROM logs WHERE dt = '2026-08-12' LIMIT 500;"
    res = await client.async_execute_query(sql, required_partition_keys=["dt"], dry_run=True)
    assert res["status"] == "SUCCEEDED"
    assert res["mode"] == "DRY_RUN"


@pytest.mark.asyncio
async def test_athena_client_async_wait_for_completion_mock():
    """Verify async_wait_for_completion polls and returns succeeded result asynchronously."""
    mock_boto = MagicMock()
    mock_boto.start_query_execution.return_value = {"QueryExecutionId": "async-query-123"}
    mock_boto.get_query_execution.return_value = {
        "QueryExecution": {
            "QueryExecutionId": "async-query-123",
            "Status": {"State": "SUCCEEDED"},
            "Statistics": {"DataScannedInBytes": 1024},
            "WorkGroup": "primary",
        }
    }

    client = AthenaClient(boto_client=mock_boto)
    res = await client.async_wait_for_completion("async-query-123", poll_interval=0.01)
    assert res["status"] == "SUCCEEDED"
    assert res["data_scanned_in_bytes"] == 1024

