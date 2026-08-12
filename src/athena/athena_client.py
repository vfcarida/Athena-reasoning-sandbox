"""AWS Athena Boto3 Client and CTAS Pipeline Engine.

Orchestrates cost-optimized queries against AWS Athena using Boto3.
Enforces pre-execution query validation via AthenaQueryGuard and provides CTAS
(Create Table As Select) transformation pipelines that automatically format data into
Snappy-compressed Apache Parquet or ORC columnar formats.

Algorithmic Complexity:
    - Query Validation: O(N) query parsing.
    - CTAS Transformation: O(1) SQL rewrite string template rendering.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

from src.athena.query_guard import AthenaQueryGuard
from src.utils.config import config

logger = logging.getLogger(__name__)


class CTASPipeline:
    """Creates CTAS (Create Table As Select) statements for columnar storage optimization."""

    @staticmethod
    def build_ctas_query(
        target_table: str,
        select_query: str,
        s3_location: str,
        output_format: str = "PARQUET",
        compression: str = "SNAPPY",
        partition_by: Optional[List[str]] = None,
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
        query_guard: AthenaQueryGuard instance for FinOps verification.
        boto_client: Optional boto3 Athena client instance.
    """

    def __init__(
        self,
        database: str = "default",
        s3_staging_dir: Optional[str] = None,
        query_guard: Optional[AthenaQueryGuard] = None,
        boto_client: Any = None,
    ) -> None:
        """Initialize the Athena Client.

        Args:
            database: Athena database name.
            s3_staging_dir: S3 output directory URI.
            query_guard: Custom AthenaQueryGuard instance.
            boto_client: Injected boto3 athena client (useful for testing/mocking).
        """
        self.database = database
        self.s3_staging_dir = s3_staging_dir or config.athena_s3_staging_dir
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
                logger.info("Initialized live AWS Boto3 Athena client (region=%s)", config.aws_region)
            except Exception as exc:
                logger.warning("Could not initialize live Boto3 client: %s. Using dry-run mode.", exc)

    def execute_query(
        self,
        sql: str,
        required_partition_keys: Optional[List[str]] = None,
        enforce_limit: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Validate and execute a SQL query against AWS Athena.

        Args:
            sql: Raw SQL query string.
            required_partition_keys: Required partition key names in WHERE clause.
            enforce_limit: Whether to enforce explicit LIMIT clause presence.
            dry_run: If True, validates query without sending to AWS Athena.

        Returns:
            Dictionary containing query execution status, QueryExecutionId, and execution metadata.
        """
        # 1. Enforce strict FinOps cost validation checks
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
            }

        # 2. Execute query via AWS Boto3 API
        response = self.boto_client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.s3_staging_dir},
        )
        execution_id = response["QueryExecutionId"]
        logger.info("Dispatched Athena query execution (ID: %s)", execution_id)

        return {
            "status": "SUBMITTED",
            "mode": "AWS_BOTO3",
            "query_execution_id": execution_id,
            "sql": sql,
            "s3_output": self.s3_staging_dir,
        }

    def execute_ctas_ingestion(
        self,
        target_table: str,
        select_query: str,
        s3_location: str,
        partition_by: Optional[List[str]] = None,
        output_format: str = "PARQUET",
        compression: str = "SNAPPY",
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Transform inner select query into CTAS statement and execute ingestion.

        Args:
            target_table: Target table name to create.
            select_query: Inner SELECT query string.
            s3_location: Destination S3 directory for Parquet data.
            partition_by: Partition column names.
            output_format: Codec format (PARQUET/ORC).
            compression: Codec compression (SNAPPY/GZIP).
            dry_run: Execute validation only.

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
        )
