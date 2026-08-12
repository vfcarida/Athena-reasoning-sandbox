"""Unit test suite for Phase 2: AWS Athena & Cloud Cost Optimization.

Verifies query AST parsing, partition key enforcement, SELECT * guard, CTAS statement generation,
and environment configuration loading.
"""

import pytest
from src.athena.query_guard import AthenaQueryGuard, UnpartitionedQueryError, UnboundedSelectError
from src.athena.athena_client import AthenaClient, CTASPipeline
from src.utils.config import AppConfig


def test_partition_filter_guard_success():
    """Verify partition guard allows queries with WHERE clause filtering partition keys."""
    guard = AthenaQueryGuard()
    query = "SELECT user_id, event_type FROM user_events WHERE dt = '2026-08-12' LIMIT 100;"
    # Should not raise exception
    guard.validate_partition_filter(query, required_partition_keys=["dt"])


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


def test_select_star_guard_failure():
    """Verify select star guard blocks SELECT * queries."""
    guard = AthenaQueryGuard()
    query = "SELECT * FROM user_events WHERE dt = '2026-08-12' LIMIT 100;"
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


def test_athena_client_dry_run_execution():
    """Verify AthenaClient validates query and returns dry-run execution status."""
    client = AthenaClient(s3_staging_dir="s3://my-staging-bucket/")
    sql = "SELECT user_id, timestamp FROM logs WHERE dt = '2026-08-12' LIMIT 500;"
    res = client.execute_query(sql, required_partition_keys=["dt"], dry_run=True)
    assert res["status"] == "SUCCEEDED"
    assert res["mode"] == "DRY_RUN"
    assert "dry-run-" in res["query_execution_id"]


def test_app_config_validation():
    """Verify AppConfig dynamic settings initialization."""
    cfg = AppConfig()
    assert cfg.aws_region is not None
    assert cfg.athena_max_scan_bytes > 0
