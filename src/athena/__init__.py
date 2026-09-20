"""AWS Athena FinOps and Cost Optimization module.

Implements query AST pre-execution verification, partition key enforcement,
unbounded SELECT * guards, Athena WorkGroup cutoff integration, runtime scan measurement,
and Snappy-compressed Apache Parquet CTAS pipelines.
"""

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

__all__ = [
    "AthenaClient",
    "AthenaQueryGuard",
    "CTASPipeline",
    "QueryExecutionError",
    "QueryTimeoutError",
    "ScanBytesLimitExceededError",
    "UnboundedSelectError",
    "UnpartitionedQueryError",
]
