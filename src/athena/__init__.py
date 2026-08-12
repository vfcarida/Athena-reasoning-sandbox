"""AWS Athena FinOps and Cost Optimization module.

Implements query AST pre-execution verification, partition key enforcement,
unbounded SELECT * guards, and Snappy-compressed Apache Parquet CTAS pipelines.
"""

from src.athena.query_guard import AthenaQueryGuard, UnpartitionedQueryError, UnboundedSelectError
from src.athena.athena_client import AthenaClient, CTASPipeline

__all__ = [
    "AthenaQueryGuard",
    "UnpartitionedQueryError",
    "UnboundedSelectError",
    "AthenaClient",
    "CTASPipeline",
]
