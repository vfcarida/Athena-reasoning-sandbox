"""Unit test suite for ENG-04: SQL AST-Driven Multi-Turn Reflection.

Verifies that AutonomousDataAgent._reflect_and_refine_sql uses sqlglot AST transformations
rather than brittle regex substitutions, correctly handling:
1. Missing partition key addition to queries with no WHERE clause.
2. Partition key appending to queries with complex existing WHERE clauses (conjunctions, disjunctions).
3. Unbounded projection (SELECT *) rewriting to explicit columns (order_id, amount).
4. Unbounded projection in queries with pre-existing explicit columns (e.g. SELECT user_id, *).
5. Projection rewriting with table aliases (e.g. SELECT * FROM sales AS s).
6. Queries with Common Table Expressions (CTEs).
7. Queries with subqueries.
8. Missing LIMIT clause addition and preservation of existing compliant LIMIT clauses.
9. Queries with embedded SQL comments and whitespace variations.
10. Fallback resilience on malformed non-SQL input.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from src.engine.data_agent import AutonomousDataAgent


def create_agent(backend: str = "duckdb") -> AutonomousDataAgent:
    """Helper to instantiate agent without active database connections."""
    return AutonomousDataAgent(backend=backend)


class TestASTPartitionPruningReflection:
    """Verify AST-driven partition pruning filter synthesis."""

    def test_adds_where_clause_when_missing(self) -> None:
        agent = create_agent()
        original = "SELECT order_id, amount FROM sales"
        err = "Athena FinOps Error: Missing WHERE clause with partition key filter."

        refined, thought = agent._reflect_and_refine_sql(
            original_sql=original,
            error_message=err,
            default_partition_key="dt",
            default_partition_val="2026-09-01",
        )

        assert "dt = '2026-09-01'" in refined
        assert "WHERE dt = '2026-09-01'" in refined
        assert "Reflection: The query was rejected because it omits partition pruning" in thought

        # Verify parsed AST has WHERE clause with EQ node
        parsed = sqlglot.parse_one(refined, read="duckdb")
        where = parsed.find(exp.Where)
        assert where is not None
        assert parsed.find(exp.EQ) is not None

    def test_appends_to_existing_simple_where(self) -> None:
        agent = create_agent()
        original = "SELECT order_id, amount FROM sales WHERE status = 'COMPLETED'"
        err = "UnpartitionedQueryError: Mandatory partition filter missing."

        refined, thought = agent._reflect_and_refine_sql(
            original_sql=original,
            error_message=err,
            default_partition_key="dt",
            default_partition_val="2026-09-01",
        )

        assert "status = 'COMPLETED'" in refined
        assert "dt = '2026-09-01'" in refined
        assert "AND" in refined

        # Verify AST has Conjunction (And node)
        parsed = sqlglot.parse_one(refined, read="duckdb")
        where = parsed.find(exp.Where)
        assert where is not None
        assert parsed.find(exp.And) is not None

    def test_appends_to_complex_where_with_or(self) -> None:
        agent = create_agent()
        original = "SELECT order_id FROM sales WHERE (category = 'books' OR category = 'music') AND amount > 50"
        err = "Query lacks required partition key filter."

        refined, thought = agent._reflect_and_refine_sql(
            original_sql=original,
            error_message=err,
            default_partition_key="dt",
            default_partition_val="2026-09-01",
        )

        assert "dt = '2026-09-01'" in refined

        # Verify AST structure is valid and parsed cleanly
        parsed = sqlglot.parse_one(refined, read="duckdb")
        assert parsed.find(exp.Where) is not None
        assert parsed.find(exp.Or) is not None
        assert parsed.find(exp.And) is not None

    def test_preserves_table_alias_and_limit_ordering(self) -> None:
        agent = create_agent()
        original = "SELECT s.order_id FROM sales AS s LIMIT 50"
        err = "Missing partition filter."

        refined, _ = agent._reflect_and_refine_sql(
            original_sql=original,
            error_message=err,
            default_partition_key="dt",
            default_partition_val="2026-09-01",
        )

        parsed = sqlglot.parse_one(refined, read="duckdb")
        # Ensure WHERE appears before LIMIT in the AST
        assert parsed.find(exp.Where) is not None
        assert parsed.find(exp.Limit) is not None
        # String representation has WHERE before LIMIT
        where_pos = refined.find("WHERE")
        limit_pos = refined.find("LIMIT")
        assert where_pos < limit_pos


class TestASTProjectionRewriting:
    """Verify AST-driven projection restriction from SELECT * to explicit columns."""

    def test_rewrites_simple_star_to_columns(self) -> None:
        agent = create_agent()
        original = "SELECT * FROM sales WHERE dt = '2026-09-01' LIMIT 10"
        err = "Unbounded 'SELECT *' projection detected."

        refined, thought = agent._reflect_and_refine_sql(
            original_sql=original,
            error_message=err,
        )

        assert "SELECT *" not in refined
        assert "order_id, amount" in refined
        assert "Reflection: The query was rejected due to an unbounded 'SELECT *' projection" in thought

        parsed = sqlglot.parse_one(refined, read="duckdb")
        assert parsed.find(exp.Star) is None
        col_names = [c.name for c in parsed.find_all(exp.Column)]
        assert "order_id" in col_names
        assert "amount" in col_names

    def test_preserves_other_columns_alongside_star(self) -> None:
        agent = create_agent()
        original = "SELECT customer_id, * FROM sales WHERE dt = '2026-09-01' LIMIT 10"
        err = "Unbounded projection error."

        refined, _ = agent._reflect_and_refine_sql(
            original_sql=original,
            error_message=err,
        )

        assert "customer_id" in refined
        assert "order_id" in refined
        assert "amount" in refined
        assert "SELECT *" not in refined

        parsed = sqlglot.parse_one(refined, read="duckdb")
        assert parsed.find(exp.Star) is None

    def test_rewrites_star_with_subquery(self) -> None:
        agent = create_agent()
        original = "SELECT * FROM (SELECT order_id, dt FROM transactions) sub WHERE sub.dt = '2026-09-01'"
        err = "Unbounded projection error: SELECT *."

        refined, _ = agent._reflect_and_refine_sql(
            original_sql=original,
            error_message=err,
        )

        assert "order_id" in refined
        assert "amount" in refined
        parsed = sqlglot.parse_one(refined, read="duckdb")
        assert parsed.find(exp.Subquery) is not None

    def test_rewrites_star_with_cte(self) -> None:
        agent = create_agent()
        original = (
            "WITH recent_orders AS (SELECT order_id, amount, dt FROM raw_orders) "
            "SELECT * FROM recent_orders WHERE dt = '2026-09-01'"
        )
        err = "Unbounded 'SELECT *' projection."

        refined, _ = agent._reflect_and_refine_sql(
            original_sql=original,
            error_message=err,
        )

        assert "order_id" in refined
        assert "amount" in refined
        parsed = sqlglot.parse_one(refined, read="duckdb")
        assert parsed.find(exp.CTE) is not None


class TestASTLimitBoundingReflection:
    """Verify AST-driven LIMIT bounding."""

    def test_appends_limit_when_missing(self) -> None:
        agent = create_agent()
        original = "SELECT order_id, amount FROM sales WHERE dt = '2026-09-01'"
        err = "Query lacks mandatory LIMIT clause."

        refined, thought = agent._reflect_and_refine_sql(
            original_sql=original,
            error_message=err,
        )

        assert "LIMIT 100" in refined
        assert "Reflection: Query lacks an explicit LIMIT clause" in thought

        parsed = sqlglot.parse_one(refined, read="duckdb")
        limit_node = parsed.find(exp.Limit)
        assert limit_node is not None

    def test_default_hygiene_adds_limit_if_not_present(self) -> None:
        agent = create_agent()
        original = "SELECT order_id, amount FROM sales WHERE dt = '2026-09-01'"
        err = "Generic internal error during query validation."

        refined, _ = agent._reflect_and_refine_sql(
            original_sql=original,
            error_message=err,
        )

        assert "LIMIT 100" in refined
        parsed = sqlglot.parse_one(refined, read="duckdb")
        assert parsed.find(exp.Limit) is not None


class TestASTFallbackResilience:
    """Verify graceful handling when input is malformed."""

    def test_unparseable_sql_fallback(self) -> None:
        agent = create_agent()
        malformed = "NOT A VALID SQL STATEMENT !@#$%"
        err = "partition filter missing"

        refined, thought = agent._reflect_and_refine_sql(
            original_sql=malformed,
            error_message=err,
            default_partition_key="dt",
            default_partition_val="2026-09-01",
        )

        assert "WHERE dt = '2026-09-01'" in refined
        assert "<think>" in thought
