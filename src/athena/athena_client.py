"""AWS Athena Boto3 Client and CTAS Pipeline Engine.

Orchestrates cost-optimized queries against AWS Athena using Boto3.
Enforces pre-execution query validation via AthenaQueryGuard (AST validation),
attaches Athena WorkGroups configured with BytesScannedCutoffPerQuery,
and monitors execution via wait_for_completion to measure DataScannedInBytes,
enforce timeout policies, and cancel runaway scans.

Cost Control Architecture:
    - Pre-execution AST guard: structural filter ensuring partition pruning.
    - WorkGroup Cutoff: server-side AWS Athena BytesScannedCutoffPerQuery cap.
    - Client-side Monitor: polls get_query_execution, tracks DataScannedInBytes,
      and triggers stop_query_execution if scan limits or timeouts are breached.

Algorithmic Complexity:
    - Query Validation: O(N) AST query parsing.
    - CTAS Transformation: O(1) SQL rewrite string template rendering.
    - Execution Polling: O(T/P) where T is execution duration and P is poll interval.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.athena.query_guard import AthenaQueryGuard
from src.utils.config import config

logger = logging.getLogger(__name__)


class ScanBytesLimitExceededError(Exception):
    """Raised when Athena query DataScannedInBytes exceeds the maximum configured threshold."""


class QueryTimeoutError(Exception):
    """Raised when Athena query execution exceeds the configured timeout seconds."""


class QueryExecutionError(Exception):
    """Raised when an Athena query fails or is cancelled."""


class CTASPipeline:
    """Creates CTAS (Create Table As Select) statements for columnar storage optimization."""

    @staticmethod
    def build_ctas_query(
        target_table: str,
        select_query: str,
        s3_location: str,
        output_format: str = "PARQUET",
        compression: str = "SNAPPY",
        partition_by: list[str] | None = None,
    ) -> str:
        """Construct a Create Table As Select (CTAS) query formatted as Parquet/ORC with Snappy.

        Args:
            target_table: Name of the table to create in Athena metastore.
            select_query: Inner validated SELECT query string.
            s3_location: S3 destination URI directory.
            output_format: Target format ('PARQUET' or 'ORC'). Default: PARQUET.
            compression: Codec compression ('SNAPPY' or 'GZIP'). Default: SNAPPY.
            partition_by: Optional list of partition column names.

        Returns:
            Formatted CTAS SQL query string.
        """
        format_clause = f"format = '{output_format.upper()}'"
        compression_clause = f"{output_format.lower()}_compression = '{compression.upper()}'"
        location_clause = f"external_location = '{s3_location.rstrip('/')}/'"

        with_properties = [format_clause, compression_clause, location_clause]

        if partition_by:
            partition_str = ", ".join([f"'{col}'" for col in partition_by])
            with_properties.append(f"partitioned_by = ARRAY[{partition_str}]")

        properties_str = ",\n    ".join(with_properties)

        ctas_sql = (
            f"CREATE TABLE {target_table}\n"
            f"WITH (\n"
            f"    {properties_str}\n"
            f") AS\n"
            f"{select_query};"
        )
        return ctas_sql


class AthenaClient:
    """High-performance cost-guarded AWS Athena client.

    Attributes:
        database: Target Athena database name.
        s3_staging_dir: S3 bucket URI for query execution output.
        workgroup: Target Athena WorkGroup name (e.g., 'primary').
        max_scan_bytes: Maximum allowed scan bytes threshold per query.
        query_guard: AthenaQueryGuard instance for FinOps AST verification.
        boto_client: Optional boto3 Athena client instance.
    """

    def __init__(
        self,
        database: str = "default",
        s3_staging_dir: str | None = None,
        workgroup: str = "primary",
        max_scan_bytes: int | None = None,
        query_guard: AthenaQueryGuard | None = None,
        boto_client: Any = None,
    ) -> None:
        """Initialize the Athena Client.

        Args:
            database: Athena database name.
            s3_staging_dir: S3 output directory URI.
            workgroup: Athena WorkGroup name with configured cost bounds.
            max_scan_bytes: Query scan bytes safety threshold (defaults to config.athena_max_scan_bytes).
            query_guard: Custom AthenaQueryGuard instance.
            boto_client: Injected boto3 athena client (useful for testing/mocking).
        """
        self.database = database
        self.s3_staging_dir = s3_staging_dir or config.athena_s3_staging_dir
        self.workgroup = workgroup
        self.max_scan_bytes = (
            max_scan_bytes if max_scan_bytes is not None else config.athena_max_scan_bytes
        )
        self.query_guard = query_guard or AthenaQueryGuard()
        self.boto_client = boto_client

        if self.boto_client is None and config.validate_aws_credentials():
            try:
                import boto3
                self.boto_client = boto3.client(
                    "athena",
                    aws_access_key_id=config.aws_access_key_id,
                    aws_secret_access_key=config.aws_secret_access_key,
                    region_name=config.aws_region,
                )
                logger.info(
                    "Initialized live AWS Boto3 Athena client (region=%s, workgroup=%s)",
                    config.aws_region,
                    self.workgroup,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not initialize live Boto3 client: %s. Using dry-run mode.", exc)

    def configure_workgroup_cutoff(self, cutoff_bytes: int | None = None) -> bool:
        """Attempt to configure the WorkGroup BytesScannedCutoffPerQuery on AWS.

        Args:
            cutoff_bytes: Maximum scan bytes cutoff. Defaults to self.max_scan_bytes.

        Returns:
            True if successfully updated/verified, False if unsupported or failed.
        """
        if self.boto_client is None:
            logger.debug("Cannot configure workgroup cutoff: no boto_client present.")
            return False

        effective_cutoff = cutoff_bytes if cutoff_bytes is not None else self.max_scan_bytes
        config_dict = {
            "BytesScannedCutoffPerQuery": effective_cutoff,
            "EnforceWorkGroupConfiguration": True,
        }

        try:
            if hasattr(self.boto_client, "update_workgroup"):
                self.boto_client.update_workgroup(
                    WorkGroup=self.workgroup,
                    ConfigurationUpdates={
                        "BytesScannedCutoffPerQuery": effective_cutoff,
                        "EnforceWorkGroupConfiguration": True,
                    },
                )
                logger.info(
                    "Updated WorkGroup '%s' BytesScannedCutoffPerQuery to %d bytes",
                    self.workgroup,
                    effective_cutoff,
                )
                return True
        except Exception as update_err:  # noqa: BLE001
            logger.debug("update_workgroup failed: %s; trying create_workgroup...", update_err)

        try:
            if hasattr(self.boto_client, "create_workgroup"):
                self.boto_client.create_workgroup(
                    Name=self.workgroup,
                    Configuration=config_dict,
                )
                logger.info(
                    "Created WorkGroup '%s' with BytesScannedCutoffPerQuery=%d",
                    self.workgroup,
                    effective_cutoff,
                )
                return True
        except Exception as create_err:  # noqa: BLE001
            logger.warning(
                "Could not set BytesScannedCutoffPerQuery on WorkGroup '%s': %s. "
                "Relying on client-side polling and cancellation.",
                self.workgroup,
                create_err,
            )
        return False

    def execute_query(
        self,
        sql: str,
        required_partition_keys: list[str] | None = None,
        enforce_limit: bool = True,
        dry_run: bool = False,
        wait: bool = False,
        poll_interval: float = 0.5,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Validate and execute a SQL query against AWS Athena.

        Args:
            sql: Raw SQL query string.
            required_partition_keys: Required partition key names in WHERE clause.
            enforce_limit: Whether to enforce explicit LIMIT clause presence.
            dry_run: If True, validates query without sending to AWS Athena.
            wait: If True, blocks until query finishes, measuring scan bytes and enforcing limits.
            poll_interval: Polling frequency in seconds when wait=True.
            timeout_seconds: Timeout threshold in seconds when wait=True.

        Returns:
            Dictionary containing query execution status, QueryExecutionId, and execution metadata.
        """
        # 1. Enforce strict FinOps AST cost validation checks
        self.query_guard.validate_query(
            query=sql,
            required_partition_keys=required_partition_keys,
            enforce_limit=enforce_limit,
        )

        if dry_run or self.boto_client is None:
            logger.info("Executing Athena query in DRY-RUN mode (Cost Validated).")
            return {
                "status": "SUCCEEDED",
                "mode": "DRY_RUN",
                "query_execution_id": f"dry-run-{int(time.time())}",
                "sql": sql,
                "s3_output": self.s3_staging_dir,
                "workgroup": self.workgroup,
                "max_scan_bytes": self.max_scan_bytes,
                "data_scanned_in_bytes": 0,
            }

        # 2. Execute query via AWS Boto3 API with attached WorkGroup
        response = self.boto_client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.s3_staging_dir},
            WorkGroup=self.workgroup,
        )
        execution_id = response["QueryExecutionId"]
        logger.info(
            "Dispatched Athena query execution (ID: %s, WorkGroup: %s)",
            execution_id,
            self.workgroup,
        )

        if wait:
            return self.wait_for_completion(
                query_execution_id=execution_id,
                poll_interval=poll_interval,
                timeout_seconds=timeout_seconds,
            )

        return {
            "status": "SUBMITTED",
            "mode": "AWS_BOTO3",
            "query_execution_id": execution_id,
            "sql": sql,
            "s3_output": self.s3_staging_dir,
            "workgroup": self.workgroup,
            "max_scan_bytes": self.max_scan_bytes,
        }

    def wait_for_completion(
        self,
        query_execution_id: str,
        poll_interval: float = 0.5,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Poll Athena query execution, measure data scanned, enforce limits, and cancel on breach.

        Args:
            query_execution_id: AWS Athena QueryExecutionId.
            poll_interval: Interval in seconds between status polls.
            timeout_seconds: Hard timeout in seconds before query cancellation.

        Returns:
            Dict containing execution status, measured data_scanned_in_bytes, elapsed_time_seconds, and statistics.

        Raises:
            ScanBytesLimitExceededError: If DataScannedInBytes exceeds self.max_scan_bytes.
            QueryTimeoutError: If execution time exceeds timeout_seconds.
            QueryExecutionError: If query enters FAILED or CANCELLED state.
        """
        if self.boto_client is None or query_execution_id.startswith("dry-run-"):
            return {
                "status": "SUCCEEDED",
                "mode": "DRY_RUN",
                "query_execution_id": query_execution_id,
                "data_scanned_in_bytes": 0,
                "elapsed_time_seconds": 0.0,
                "workgroup": self.workgroup,
                "max_scan_bytes": self.max_scan_bytes,
            }

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                try:
                    self.boto_client.stop_query_execution(QueryExecutionId=query_execution_id)
                except Exception as stop_err:  # noqa: BLE001
                    logger.warning(
                        "Failed to stop timed out query %s: %s",
                        query_execution_id,
                        stop_err,
                    )
                raise QueryTimeoutError(
                    f"Athena query '{query_execution_id}' timed out after {elapsed:.2f}s "
                    f"(limit: {timeout_seconds}s) and was cancelled."
                )

            resp = self.boto_client.get_query_execution(QueryExecutionId=query_execution_id)
            query_exec = resp.get("QueryExecution", {})
            status_info = query_exec.get("Status", {})
            state = status_info.get("State")
            statistics = query_exec.get("Statistics", {})
            data_scanned = statistics.get("DataScannedInBytes", 0)

            # Enforce hard scan bytes limit
            if data_scanned > self.max_scan_bytes:
                try:
                    self.boto_client.stop_query_execution(QueryExecutionId=query_execution_id)
                except Exception as stop_err:  # noqa: BLE001
                    logger.warning(
                        "Failed to stop query %s breaching scan limit: %s",
                        query_execution_id,
                        stop_err,
                    )
                raise ScanBytesLimitExceededError(
                    f"Athena query '{query_execution_id}' scanned {data_scanned} bytes, "
                    f"exceeding max allowed limit of {self.max_scan_bytes} bytes. Query was cancelled."
                )

            if state == "SUCCEEDED":
                return {
                    "status": "SUCCEEDED",
                    "mode": "AWS_BOTO3",
                    "query_execution_id": query_execution_id,
                    "data_scanned_in_bytes": data_scanned,
                    "elapsed_time_seconds": elapsed,
                    "statistics": statistics,
                    "workgroup": query_exec.get("WorkGroup", self.workgroup),
                    "max_scan_bytes": self.max_scan_bytes,
                }

            if state in ("FAILED", "CANCELLED"):
                reason = status_info.get("StateChangeReason", f"Query entered {state} state")
                raise QueryExecutionError(
                    f"Athena query '{query_execution_id}' failed ({state}): {reason}"
                )

            time.sleep(poll_interval)

    async def async_wait_for_completion(
        self,
        query_execution_id: str,
        poll_interval: float = 0.5,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Asynchronously poll Athena query execution, measure data scanned, and cancel on breach.

        Args:
            query_execution_id: AWS Athena QueryExecutionId.
            poll_interval: Interval in seconds between status polls.
            timeout_seconds: Hard timeout in seconds before query cancellation.

        Returns:
            Dict containing execution status, measured data_scanned_in_bytes, elapsed_time_seconds, and statistics.
        """
        if self.boto_client is None or query_execution_id.startswith("dry-run-"):
            return {
                "status": "SUCCEEDED",
                "mode": "DRY_RUN",
                "query_execution_id": query_execution_id,
                "data_scanned_in_bytes": 0,
                "elapsed_time_seconds": 0.0,
                "workgroup": self.workgroup,
                "max_scan_bytes": self.max_scan_bytes,
            }

        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                try:
                    await asyncio.to_thread(self.boto_client.stop_query_execution, QueryExecutionId=query_execution_id)
                except Exception as stop_err:  # noqa: BLE001
                    logger.warning("Failed to cancel timed out Athena query %s: %s", query_execution_id, stop_err)
                raise QueryTimeoutError(
                    f"Athena query '{query_execution_id}' timed out after {elapsed:.2f}s "
                    f"(limit: {timeout_seconds}s). Query was cancelled."
                )

            res = await asyncio.to_thread(self.boto_client.get_query_execution, QueryExecutionId=query_execution_id)
            query_exec = res["QueryExecution"]
            status_info = query_exec["Status"]
            state = status_info["State"]
            statistics = query_exec.get("Statistics", {})
            data_scanned = statistics.get("DataScannedInBytes", 0)

            if data_scanned > self.max_scan_bytes:
                try:
                    await asyncio.to_thread(self.boto_client.stop_query_execution, QueryExecutionId=query_execution_id)
                except Exception as cancel_err:  # noqa: BLE001
                    logger.warning("Failed to cancel query exceeding scan limit: %s", cancel_err)
                raise ScanBytesLimitExceededError(
                    f"Athena query '{query_execution_id}' scanned {data_scanned} bytes, "
                    f"exceeding max allowed limit of {self.max_scan_bytes} bytes. Query was cancelled."
                )

            if state == "SUCCEEDED":
                return {
                    "status": "SUCCEEDED",
                    "mode": "AWS_BOTO3",
                    "query_execution_id": query_execution_id,
                    "data_scanned_in_bytes": data_scanned,
                    "elapsed_time_seconds": elapsed,
                    "statistics": statistics,
                    "workgroup": query_exec.get("WorkGroup", self.workgroup),
                    "max_scan_bytes": self.max_scan_bytes,
                }

            if state in ("FAILED", "CANCELLED"):
                reason = status_info.get("StateChangeReason", f"Query entered {state} state")
                raise QueryExecutionError(
                    f"Athena query '{query_execution_id}' failed ({state}): {reason}"
                )

            await asyncio.sleep(poll_interval)

    async def async_execute_query(
        self,
        sql: str,
        required_partition_keys: list[str] | None = None,
        enforce_limit: bool = True,
        dry_run: bool = False,
        wait: bool = False,
        poll_interval: float = 0.5,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Asynchronously validate and execute a SQL query against AWS Athena.

        Args:
            sql: Raw SQL query string.
            required_partition_keys: Required partition key names in WHERE clause.
            enforce_limit: Whether to enforce explicit LIMIT clause presence.
            dry_run: If True, validates query without sending to AWS Athena.
            wait: If True, awaits query completion asynchronously.
            poll_interval: Polling frequency in seconds when wait=True.
            timeout_seconds: Timeout threshold in seconds when wait=True.

        Returns:
            Dictionary containing query execution status, QueryExecutionId, and execution metadata.
        """
        res = await asyncio.to_thread(
            self.execute_query,
            sql=sql,
            required_partition_keys=required_partition_keys,
            enforce_limit=enforce_limit,
            dry_run=dry_run,
            wait=False,
        )

        if wait and res.get("status") == "SUBMITTED":
            return await self.async_wait_for_completion(
                query_execution_id=res["query_execution_id"],
                poll_interval=poll_interval,
                timeout_seconds=timeout_seconds,
            )

        return res

    def execute_ctas_ingestion(
        self,
        target_table: str,
        select_query: str,
        s3_location: str,
        partition_by: list[str] | None = None,
        output_format: str = "PARQUET",
        compression: str = "SNAPPY",
        dry_run: bool = False,
        wait: bool = False,
        poll_interval: float = 0.5,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Transform inner select query into CTAS statement and execute ingestion.

        Args:
            target_table: Target table name to create.
            select_query: Inner SELECT query string.
            s3_location: Destination S3 directory for Parquet data.
            partition_by: Partition column names.
            output_format: Codec format (PARQUET/ORC).
            compression: Codec compression (SNAPPY/GZIP).
            dry_run: Execute validation only.
            wait: Block until completion and return measured scan statistics.
            poll_interval: Poll frequency when wait=True.
            timeout_seconds: Execution timeout when wait=True.

        Returns:
            Query execution status payload.
        """
        ctas_sql = CTASPipeline.build_ctas_query(
            target_table=target_table,
            select_query=select_query,
            s3_location=s3_location,
            output_format=output_format,
            compression=compression,
            partition_by=partition_by,
        )

        return self.execute_query(
            sql=ctas_sql,
            required_partition_keys=partition_by,
            enforce_limit=False,  # CTAS queries do not require LIMIT
            dry_run=dry_run,
            wait=wait,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )
