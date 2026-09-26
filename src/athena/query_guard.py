"""AWS Athena FinOps Query Guard and Cost Protection Engine.

Analyzes raw SQL queries prior to execution against AWS Athena ($5/TB scan cost).
Uses sqlglot AST parsing to enforce mandatory partition key filters in WHERE clauses,
block unbounded SELECT * projections, and guarantee explicit LIMIT clauses.

Cost Guard Architectural Note:
    - AST validation is a structural pre-execution filter (technical hygiene control).
    - An explicit LIMIT clause restricts client result buffer sizes, but does NOT bound
      Athena scan volume in columnar storage (Parquet/ORC scan entire column blocks).
    - Real spending bounds must be enforced at execution time via Athena WorkGroup
      BytesScannedCutoffPerQuery and client-side DataScannedInBytes polling/cancellation.

Algorithmic Complexity:
    - SQL AST Parsing: O(N) where N is length of SQL query string.
    - Partition Filter AST Traversal: O(K) where K is number of AST predicate nodes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


class UnpartitionedQueryError(Exception):
    """Raised when a SQL query omits mandatory partition key filters in the WHERE clause."""


class UnboundedSelectError(Exception):
    """Raised when a SQL query uses SELECT * or omits mandatory projection and LIMIT bounds."""


class AthenaQueryGuard:
    """AST-based pre-execution SQL guard enforcing AWS Athena FinOps compliance.

    Attributes:
        default_partition_keys: Common partition keys checked if none explicitly provided.
        max_default_limit: Hard upper bound on result set rows (default 10,000).
    """

    DEFAULT_PARTITION_KEYS: ClassVar[set[str]] = {
        "dt",
        "date",
        "partition_date",
        "year",
        "month",
        "day",
        "tenant_id",
        "ds",
    }

    METADATA_COMMANDS: ClassVar[tuple[str, ...]] = (
        "describe",
        "desc",
        "show",
        "explain",
        "pragma",
    )

    METADATA_SCHEMAS: ClassVar[tuple[str, ...]] = (
        "information_schema",
        "system",
        "pg_catalog",
    )

    def __init__(
        self,
        default_partition_keys: set[str] | None = None,
        dialect: str = "trino",
    ) -> None:
        """Initialize the Athena Query Guard.

        Args:
            default_partition_keys: Optional set of default partition column names.
            dialect: Default SQL dialect for parsing (default "trino").
        """
        self.partition_keys = default_partition_keys or self.DEFAULT_PARTITION_KEYS
        self.dialect = dialect

    def is_metadata_query(self, query: str) -> bool:
        """Check whether a query is a metadata discovery or execution plan operation.

        Metadata queries (e.g. SHOW TABLES, DESCRIBE tbl, EXPLAIN query, PRAGMA table_info)
        do not trigger large table partition scans and are exempt from partition pruning rules.

        Args:
            query: Raw SQL query string.

        Returns:
            True if the query is a metadata exploration command, False otherwise.
        """
        clean = self._clean_query(query).strip()
        if not clean:
            return False
        first_token = clean.split()[0].lower()
        if first_token in self.METADATA_COMMANDS:
            return True

        # Check for queries targeting system / information_schema tables
        lower_query = clean.lower()
        if any(f"{schema}." in lower_query or f"from {schema}" in lower_query for schema in self.METADATA_SCHEMAS):
            return True

        # Check for scalar/constant queries without FROM clause (e.g. SELECT 1, SELECT version())
        try:
            parsed = self._parse_sql(clean)
            selects = self._extract_select_nodes(parsed)
            if selects and all(sel.find(exp.From) is None for sel in selects):
                return True
        except Exception:
            pass

        return False

    def estimate_query_cost(
        self,
        query: str,
        table_size_bytes: int = 1_000_000_000,
        total_columns: int = 20,
    ) -> dict[str, Any]:
        """Estimate projected data scan volume and cost in USD prior to execution.

        Calculates cost using standard AWS Athena pricing ($5.00 per TB scanned,
        subject to a 10MB minimum charge per query on non-zero scans).

        Args:
            query: Raw SQL query string.
            table_size_bytes: Estimated baseline table unpartitioned storage size (default 1GB).
            total_columns: Estimated total columns in the target table (default 20).

        Returns:
            Dictionary with projected_scan_bytes, projected_cost_usd, selectivity, and cost_tier.
        """
        if self.is_metadata_query(query):
            return {
                "projected_scan_bytes": 0,
                "projected_cost_usd": 0.0,
                "is_metadata": True,
                "projection_ratio": 0.0,
                "selectivity": 0.0,
                "cost_tier": "FREE",
            }

        parsed = self._parse_sql(query)
        selects = self._extract_select_nodes(parsed)
        if not selects:
            return {
                "projected_scan_bytes": table_size_bytes,
                "projected_cost_usd": round((table_size_bytes / (1024**4)) * 5.0, 6),
                "is_metadata": False,
                "projection_ratio": 1.0,
                "selectivity": 1.0,
                "cost_tier": "HIGH",
            }

        sel = selects[0]
        # Calculate projection ratio
        has_star = any(
            isinstance(expr, exp.Star)
            or (isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star))
            for expr in sel.expressions
        )
        if has_star:
            projection_ratio = 1.0
        else:
            col_count = len(sel.expressions)
            projection_ratio = min(max(col_count / max(total_columns, 1), 0.05), 1.0)

        # Calculate partition selectivity
        where_node = sel.args.get("where")
        selectivity = 1.0
        if where_node is not None:
            pred = where_node.this
            if isinstance(pred, exp.EQ):
                selectivity = 0.01  # Point partition filter: ~1% of dataset
            elif isinstance(pred, (exp.Between, exp.In)):
                selectivity = 0.05  # Range or discrete list: ~5% of dataset
            elif isinstance(pred, (exp.And, exp.Or)):
                selectivity = 0.02  # Compound predicate

        raw_scanned = int(table_size_bytes * projection_ratio * selectivity)
        # AWS Athena enforces a 10MB minimum scan charge per query
        athena_min_bytes = 10 * 1024 * 1024
        effective_scanned = max(raw_scanned, athena_min_bytes) if raw_scanned > 0 else 0
        cost_usd = round((effective_scanned / (1024**4)) * 5.0, 6)

        tier = "LOW"
        if cost_usd > 0.50:
            tier = "HIGH"
        elif cost_usd > 0.05:
            tier = "MEDIUM"

        return {
            "projected_scan_bytes": effective_scanned,
            "projected_cost_usd": cost_usd,
            "is_metadata": False,
            "projection_ratio": round(projection_ratio, 3),
            "selectivity": round(selectivity, 3),
            "cost_tier": tier,
        }

    def _parse_sql(self, query: str, dialect: str | None = None) -> exp.Expression:
        """Parse raw SQL query into sqlglot AST expression.

        Args:
            query: Raw SQL query string.
            dialect: Optional SQL dialect override (defaults to self.dialect).

        Returns:
            sqlglot AST root node.

        Raises:
            UnpartitionedQueryError: If query fails to parse.
        """
        clean_query = self._clean_query(query)
        target_dialect = dialect or self.dialect
        try:
            parsed = sqlglot.parse_one(clean_query, read=target_dialect)
        except Exception as exc:
            raise UnpartitionedQueryError(f"Athena FinOps Error: Failed to parse SQL query: {exc}") from exc
        return parsed  # type: ignore[return-value]

    @classmethod
    def _extract_select_nodes(cls, root: exp.Expression) -> list[exp.Select]:
        """Extract top-level SELECT nodes requiring partition filtering (including CTEs / CTAS)."""
        if isinstance(root, exp.Select):
            return [root]
        if isinstance(root, exp.Union):
            selects: list[exp.Select] = []
            for branch in (root.this, root.expression):
                selects.extend(cls._extract_select_nodes(branch))
            return selects
        sel = root.find(exp.Select)
        return [sel] if sel else []

    @classmethod
    def _is_pruning_predicate(cls, node: Any, partition_keys: set[str]) -> bool:
        """Recursively verify whether AST predicate enforces partition pruning on partition_keys.

        Rejects:
            - NULL checks (IS NULL, IS NOT NULL) which scan all data blocks.
            - Function-wrapped partition keys (e.g. YEAR(dt) = 2026).
            - Disjunctions (OR) where any branch omits a partition filter.
            - Nested subqueries that do not filter the scanned table.
        """
        if node is None:
            return False

        if isinstance(node, exp.Paren):
            return cls._is_pruning_predicate(node.this, partition_keys)

        if isinstance(node, exp.And):
            # In an AND conjunction, if either side prunes partition keys, the query prunes
            return cls._is_pruning_predicate(node.left, partition_keys) or cls._is_pruning_predicate(
                node.right, partition_keys
            )

        if isinstance(node, exp.Or):
            # In an OR disjunction, EVERY branch must prune to prevent a full table scan
            return cls._is_pruning_predicate(node.left, partition_keys) and cls._is_pruning_predicate(
                node.right, partition_keys
            )

        if isinstance(node, (exp.Between, exp.In)):
            return isinstance(node.this, exp.Column) and node.this.name.lower() in partition_keys

        if isinstance(node, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            left_is_col = isinstance(node.left, exp.Column) and node.left.name.lower() in partition_keys
            right_is_col = isinstance(node.right, exp.Column) and node.right.name.lower() in partition_keys

            # Reject column-to-column comparisons (e.g. dt = dt or dt = other_col)
            if left_is_col and not isinstance(node.right, exp.Column) and not node.right.find(exp.Column):
                return True
            return bool(right_is_col and not isinstance(node.left, exp.Column) and not node.left.find(exp.Column))

        # All other AST nodes (exp.Is, exp.Func, exp.Anonymous, exp.Subquery, etc.) do NOT prune
        return False

    def validate_partition_filter(
        self,
        query: str,
        required_partition_keys: list[str] | None = None,
    ) -> None:
        """Verify query contains a WHERE clause filtering against partition keys via AST analysis.

        Args:
            query: Raw SQL query string.
            required_partition_keys: Specific partition keys required for this query.

        Raises:
            UnpartitionedQueryError: If query omits WHERE clause or fails AST partition filter check.
        """
        target_keys = {k.lower() for k in (required_partition_keys or self.partition_keys)}
        parsed = self._parse_sql(query)
        select_nodes = self._extract_select_nodes(parsed)

        if not select_nodes:
            raise UnpartitionedQueryError(
                "Athena FinOps Error: Query lacks a WHERE clause. "
                f"Must explicitly filter against partition keys: {sorted(target_keys)}"
            )

        for sel in select_nodes:
            where_node = sel.args.get("where")
            if where_node is None:
                raise UnpartitionedQueryError(
                    "Athena FinOps Error: Query lacks a WHERE clause. "
                    f"Must explicitly filter against partition keys: {sorted(target_keys)}"
                )

            if not self._is_pruning_predicate(where_node.this, target_keys):
                raise UnpartitionedQueryError(
                    f"Athena FinOps Error: WHERE clause does not filter on required partition keys. "
                    f"Expected one of {sorted(target_keys)}. Query: '{query}'"
                )

    def validate_projections_and_limits(
        self,
        query: str,
        enforce_limit: bool = True,
    ) -> None:
        """Verify query does not execute unbounded SELECT * projections and includes explicit LIMIT.

        Note:
            LIMIT is a client memory hygiene check, not an Athena scan-cost boundary. Athena charges
            for the full volume of data scanned by column blocks regardless of LIMIT.

        Args:
            query: Raw SQL query string.
            enforce_limit: Whether to enforce explicit LIMIT clause presence.

        Raises:
            UnboundedSelectError: If query contains SELECT * or lacks explicit LIMIT when required.
        """
        parsed = self._parse_sql(query)
        select_nodes = self._extract_select_nodes(parsed)

        for sel in select_nodes:
            # 1. Check for SELECT * pattern in AST projections
            has_star = any(
                isinstance(expr, exp.Star)
                or (isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star))
                for expr in sel.expressions
            )
            if has_star:
                raise UnboundedSelectError(
                    "Athena FinOps Error: Unbounded 'SELECT *' queries are prohibited ($5/TB scan risk). "
                    "Specify explicit column names (e.g., SELECT id, timestamp, payload)."
                )

            # 2. Check for explicit LIMIT clause if requested
            if enforce_limit:
                has_limit = sel.args.get("limit") is not None or sel.find(exp.Limit) is not None
                if not has_limit:
                    raise UnboundedSelectError(
                        "Athena FinOps Error: Query lacks an explicit LIMIT clause. "
                        "Append a LIMIT statement to restrict scan size (e.g., LIMIT 1000)."
                    )

    def validate_query(
        self,
        query: str,
        required_partition_keys: list[str] | None = None,
        enforce_limit: bool = True,
    ) -> None:
        """Execute full AST validation suite on target SQL query.

        Args:
            query: Raw SQL query string.
            required_partition_keys: List of partition column names.
            enforce_limit: Whether to check for LIMIT clause.

        Raises:
            UnpartitionedQueryError: If partition keys missing in WHERE clause.
            UnboundedSelectError: If SELECT * used or LIMIT missing.
        """
        if self.is_metadata_query(query):
            logger.info(
                "SQL query is a metadata discovery operation, exempt from partition checks: '%s'",
                query[:60],
            )
            return

        self.validate_partition_filter(query, required_partition_keys)
        self.validate_projections_and_limits(query, enforce_limit=enforce_limit)
        logger.info("SQL query passed Athena FinOps AST cost validation: '%s'", query[:60])

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


# Aliases and public exports
QueryCostGuard = AthenaQueryGuard

__all__ = [
    "AthenaQueryGuard",
    "QueryCostGuard",
    "UnpartitionedQueryError",
    "UnboundedSelectError",
]
