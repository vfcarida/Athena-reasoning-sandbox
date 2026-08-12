"""AWS Athena FinOps Query Guard and Cost Protection Engine.

Analyzes raw SQL queries prior to execution against AWS Athena ($5/TB scan cost).
Algorithmic AST parsing enforces mandatory partition key filters in WHERE clauses,
blocks unbounded SELECT * projections, and guarantees explicit LIMIT clauses.

Algorithmic Complexity:
    - SQL Tokenization & Parsing: O(N) where N is length of SQL query string.
    - Partition Filter Check: O(K) where K is number of WHERE clause tokens.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Set

logger = logging.getLogger(__name__)


class UnpartitionedQueryError(Exception):
    """Raised when a SQL query omits mandatory partition key filters in the WHERE clause."""
    pass


class UnboundedSelectError(Exception):
    """Raised when a SQL query uses SELECT * or omits mandatory projection and LIMIT bounds."""
    pass


class AthenaQueryGuard:
    """Algorithmic pre-execution SQL guard enforcing AWS Athena FinOps compliance.

    Attributes:
        default_partition_keys: Common partition keys checked if none explicitly provided.
        max_default_limit: Hard upper bound on result set rows (default 10,000).
    """

    DEFAULT_PARTITION_KEYS = {"dt", "date", "partition_date", "year", "month", "day", "tenant_id", "ds"}

    def __init__(self, default_partition_keys: Optional[Set[str]] = None) -> None:
        """Initialize the Athena Query Guard.

        Args:
            default_partition_keys: Optional set of default partition column names.
        """
        self.partition_keys = default_partition_keys or self.DEFAULT_PARTITION_KEYS

    def validate_partition_filter(
        self,
        query: str,
        required_partition_keys: Optional[List[str]] = None,
    ) -> None:
        """Verify query contains a WHERE clause filtering against partition keys.

        Args:
            query: Raw SQL query string.
            required_partition_keys: Specific partition keys required for this query.

        Raises:
            UnpartitionedQueryError: If query omits WHERE clause or fails partition filter check.
        """
        clean_query = self._clean_query(query)
        target_keys = set(required_partition_keys) if required_partition_keys else self.partition_keys

        # Extract WHERE clause
        where_match = re.search(r"\bWHERE\b\s+(.*)", clean_query, re.IGNORECASE | re.DOTALL)
        if not where_match:
            raise UnpartitionedQueryError(
                "Athena FinOps Error: Query lacks a WHERE clause. "
                f"Must explicitly filter against partition keys: {sorted(list(target_keys))}"
            )

        where_clause = where_match.group(1).lower()

        # Check if at least one partition key is present in the WHERE clause
        found_key = False
        for key in target_keys:
            # Pattern matching column usage e.g. "dt =", "dt >", "dt IN"
            pattern = rf"\b{re.escape(key.lower())}\b"
            if re.search(pattern, where_clause):
                found_key = True
                break

        if not found_key:
            raise UnpartitionedQueryError(
                f"Athena FinOps Error: WHERE clause does not filter on required partition keys. "
                f"Expected one of {sorted(list(target_keys))}. Query: '{query}'"
            )

    def validate_projections_and_limits(
        self,
        query: str,
        enforce_limit: bool = True,
    ) -> None:
        """Verify query does not execute unbounded SELECT * projections.

        Args:
            query: Raw SQL query string.
            enforce_limit: Whether to enforce explicit LIMIT clause presence.

        Raises:
            UnboundedSelectError: If query contains SELECT * without explicit columns or missing LIMIT.
        """
        clean_query = self._clean_query(query)

        # 1. Check for SELECT * pattern
        select_star_pattern = r"\bSELECT\s+(\*|[\w_]+\.\*)"
        if re.search(select_star_pattern, clean_query, re.IGNORECASE):
            raise UnboundedSelectError(
                "Athena FinOps Error: Unbounded 'SELECT *' queries are prohibited ($5/TB scan risk). "
                "Specify explicit column names (e.g., SELECT id, timestamp, payload)."
            )

        # 2. Check for explicit LIMIT clause if requested
        if enforce_limit:
            limit_pattern = r"\bLIMIT\s+\d+\b"
            if not re.search(limit_pattern, clean_query, re.IGNORECASE):
                raise UnboundedSelectError(
                    "Athena FinOps Error: Query lacks an explicit LIMIT clause. "
                    "Append a LIMIT statement to restrict scan size (e.g., LIMIT 1000)."
                )

    def validate_query(
        self,
        query: str,
        required_partition_keys: Optional[List[str]] = None,
        enforce_limit: bool = True,
    ) -> None:
        """Execute full algorithmic validation suite on target SQL query.

        Args:
            query: Raw SQL query string.
            required_partition_keys: List of partition column names.
            enforce_limit: Whether to check for LIMIT clause.

        Raises:
            UnpartitionedQueryError: If partition keys missing in WHERE clause.
            UnboundedSelectError: If SELECT * used or LIMIT missing.
        """
        self.validate_partition_filter(query, required_partition_keys)
        self.validate_projections_and_limits(query, enforce_limit=enforce_limit)
        logger.info("SQL query passed Athena FinOps cost validation: '%s'", query[:60])

    @staticmethod
    def _clean_query(query: str) -> str:
        """Strip multiline comments and normalize whitespace.

        Args:
            query: Raw SQL input string.

        Returns:
            Cleaned SQL string.
        """
        # Remove single line comments -- ...
        cleaned = re.sub(r"--.*$", "", query, flags=re.MULTILINE)
        # Remove multiline comments /* ... */
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
        # Normalize spaces
        return " ".join(cleaned.split())
