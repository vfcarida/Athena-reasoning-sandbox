# FinOps AST Cost Guard & Schema Discovery

> **Deterministic Pre-Execution Governance, Cost Estimation, and Schema Discovery for Data Lake AI Agents**

In cloud data lake architectures (AWS Athena, Trino, Presto), queries are billed primarily by the volume of data scanned—typically **$5.00 per Terabyte (TB)**. Autonomous AI agents writing dynamic SQL queries can easily trigger catastrophic runaway cloud bills if they execute full-table scans, wildcard projections (`SELECT *`), or unpartitioned queries across petabyte-scale object stores (S3).

Athena Reasoning Sandbox solves this at compile-time with **`AthenaQueryGuard`**: a deterministic, AST-based FinOps firewall powered by `sqlglot`.

---

## 🛡️ Core FinOps Safeguards

```
              ┌──────────────────────────────────────┐
              │      Generated SQL from Agent        │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
                     sqlglot AST Parse Tree
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
  Metadata Query?        Partition Filter?       Projection Clean?
  (SHOW / DESCRIBE)       (WHERE dt = '...')      (No SELECT *)
         │                       │                       │
      [BYPASS]             [VERIFY AST]             [VERIFY AST]
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                   Pre-Execution Cost Estimate
                   (Scanned Bytes & Projected USD)
                                 │
                                 ▼
                    Execution Engine (Athena / DuckDB)
```

### 1. Partition Pruning Enforcement
Data lake tables are partitioned on disk by specific keys (e.g., `dt`, `year`, `region`). A query without a predicate on the partition key forces the query engine to read every file in every S3 prefix.

`AthenaQueryGuard` traverses the SQL Abstract Syntax Tree (`sqlglot.expressions.Where`) to ensure that:
- A `WHERE` clause is present.
- The designated partition key (e.g., `dt`) is actively compared with an equality or comparison operator.
- The predicate is part of a top-level conjunction (`AND`), preventing bypassing via `OR TRUE`.

### 2. Wildcard Projection Prevention
Columnar formats (Parquet, ORC) allow analytical engines to scan only the specific columns requested. Using `SELECT *` defeats columnar optimization and scans 100% of column files.

`AthenaQueryGuard` inspects `exp.Select` expressions and rejects any occurrence of `exp.Star`, forcing the agent to project only the explicit columns needed for reasoning.

### 3. Explicit Row Limits
Unbounded queries can overwhelm agent memory contexts and client processes. `AthenaQueryGuard` validates that an explicit `LIMIT` clause is present (or injects one during automated reflection).

### 4. Metadata Query Bypass
Autonomous data exploration requires schema discovery commands (`SHOW TABLES`, `SHOW COLUMNS`, `DESCRIBE <table>`, `EXPLAIN ...`) or utility queries (`SELECT 1`). 

`AthenaQueryGuard` detects metadata and introspection AST roots, safely exempting them from partition pruning and column projection checks while maintaining strict syntax and injection validation.

---

## 💰 Pre-Execution Query Cost Estimation

Before dispatching an analytical query to AWS Athena, `AthenaQueryGuard` can estimate the bytes that will be scanned and compute the projected cost in USD.

### Formula

$$\text{Estimated Bytes} = \text{Table Size} \times \text{Projection Ratio} \times \text{Partition Pruning Factor}$$

- **Projection Ratio**: $\frac{\text{Projected Columns}}{\text{Total Table Columns}}$ (for columnar formats like Parquet/ORC).
- **Partition Pruning Factor**: Default $0.05$ (5%) when a valid partition filter is present; $1.0$ (100% full scan) if unpartitioned.
- **Cost (USD)**: $\frac{\text{Estimated Bytes}}{10^{12}} \times \$5.00$

### Code Example

```python
from src.athena.query_guard import AthenaQueryGuard, TableStatistics

# Initialize guard
guard = AthenaQueryGuard(
    required_partition_key="dt",
    require_partition_filter=True,
    forbid_star_projection=True,
    require_limit=True,
    dialect="trino",
)

# Define known table metadata
stats = TableStatistics(
    table_name="orders",
    total_bytes=10 * (1024**4),  # 10 TB raw parquet dataset
    total_columns=20,
    partition_column="dt",
)

query = "SELECT order_id, amount FROM orders WHERE dt = '2026-09-01' LIMIT 100"

# 1. Validate AST
result = guard.validate(query)
assert result.is_valid is True

# 2. Estimate Scan Volume & Cost
cost_est = guard.estimate_query_cost(query, stats, price_per_tb_usd=5.0)
print(f"Projected Scan: {cost_est.estimated_bytes_scanned / (1024**3):.2f} GB")
print(f"Projected Cost: ${cost_est.estimated_cost_usd:.4f} USD")
# Projected Scan: 51.20 GB
# Projected Cost: $0.2500 USD (instead of $50.00 for an unpartitioned scan!)
```

---

## 🔍 Autonomous Schema Discovery

Agents should not guess table columns. Athena provides native schema discovery across the executive engine and the Model Context Protocol (MCP) server.

### Available Discovery Tools

1. **`describe_table` (Executive Tool & MCP Endpoint)**:
   ```json
   {
     "jsonrpc": "2.0",
     "id": 1,
     "method": "tools/call",
     "params": {
       "name": "describe_table",
       "arguments": {"table_name": "ecommerce.orders"}
     }
   }
   ```
   **Returns**:
   ```json
   {
     "table": "ecommerce.orders",
     "columns": [
       {"name": "order_id", "type": "VARCHAR"},
       {"name": "customer_id", "type": "VARCHAR"},
       {"name": "amount", "type": "DOUBLE"},
       {"name": "dt", "type": "VARCHAR", "is_partition": true}
     ],
     "partition_keys": ["dt"]
   }
   ```

2. **Programmatic Python Discovery**:
   ```python
   from src.engine.data_agent import AutonomousDataAgent
   from src.engine.executor import ExecutiveEngine

   agent = AutonomousDataAgent(model_name="mock-model")
   schema = agent.discover_schema("ecommerce.orders")
   print(schema["columns"])
   ```

---

## 🔄 Self-Healing Multi-Turn Reflection

If an LLM agent produces an invalid query (e.g. `SELECT * FROM orders`), Athena does not crash. It leverages `sqlglot` AST rewriting:

```
[Agent Output]  SELECT * FROM orders
       │
       ▼
[QueryGuard]    ❌ StarProjectionError: Wildcard '*' not allowed
       │
       ▼
[Reflection]    Agent synthesizes `<think>` reflection trace.
       │
       ▼
[AST Rewriter]  - Discovers table schema
                - Replaces exp.Star with explicit columns (e.g. order_id, amount)
                - Injects WHERE dt = '2026-09-01'
                - Injects LIMIT 100
       │
       ▼
[QueryGuard]    ✅ AST Validated -> Dispatched to Engine
```

This ensures zero runaway queries reach the cloud data lake while maintaining autonomous agent self-direction.
