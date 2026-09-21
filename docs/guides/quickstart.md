# Quickstart Guide

This guide walks you through executing analytical data tasks using the **Athena Reasoning Sandbox** with local DuckDB and AST query governance.

---

## 1. Installation

Install the package and core dependencies in an isolated virtual environment:

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 2. Using `AutonomousDataAgent` for Self-Correcting Analytics

The `AutonomousDataAgent` coordinates cognitive reasoning with FinOps pre-execution AST validation. If an initial query is non-compliant (e.g., forgets to filter by partition or issues an unbounded `SELECT *`), the agent reflects on the AST error and automatically repairs the query.

```python
import asyncio
from src.athena.duckdb_client import DuckDBClient
from src.engine.data_agent import AutonomousDataAgent

async def main():
    # 1. Initialize local DuckDB analytical client
    client = DuckDBClient(database=":memory:")
    
    # 2. Seed sample data
    client.con.execute("""
        CREATE TABLE daily_sales (
            order_id INTEGER,
            customer_id VARCHAR,
            amount DOUBLE,
            dt VARCHAR
        );
        INSERT INTO daily_sales VALUES
            (101, 'cust_a', 120.50, '2026-09-01'),
            (102, 'cust_b', 89.00, '2026-09-01'),
            (103, 'cust_c', 450.00, '2026-09-02');
    """)

    # 3. Instantiate AutonomousDataAgent
    agent = AutonomousDataAgent(
        backend="duckdb",
        duckdb_client=client,
        max_reflection_turns=3,
    )

    # 4. Propose an initial query that violates partition pruning rules
    initial_flawed_sql = "SELECT order_id, amount FROM daily_sales LIMIT 10;"

    print("[*] Running AutonomousDataAgent...")
    summary = await agent.run(
        goal="Retrieve sales order amounts",
        initial_sql=initial_flawed_sql,
        required_partition_keys=["dt"],
    )

    print(f"Success: {summary.success}")
    print(f"Total Turns: {summary.turns}")
    print(f"Final SQL: {summary.final_sql}")
    print(f"Returned Rows: {summary.result_data.get('rows')}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Output:
```text
[*] Running AutonomousDataAgent...
Success: True
Total Turns: 2
Final SQL: SELECT order_id, amount FROM daily_sales WHERE dt = '2026-09-01' LIMIT 10
Returned Rows: [[101, 120.5], [102, 89.0]]
```

---

## 3. Direct Query Governance with `AthenaQueryGuard`

You can also use the AST guard directly to validate queries before submitting them to cloud services or execution engines:

```python
from src.athena.query_guard import AthenaQueryGuard, UnpartitionedQueryError

guard = AthenaQueryGuard()

sql = "SELECT order_id, amount FROM orders LIMIT 10;"

try:
    # AST validation checks partition predicates and limits
    guard.validate_query(sql, required_partition_keys=["dt"])
except UnpartitionedQueryError as exc:
    print(f"Query rejected: {exc}")
```

---

## 4. Conversation State Checkpointing

Athena includes a conversation-level checkpoint manager with SHA-256 side-effect ledgers to ensure re-entrant, idempotent execution:

```python
from src.state.checkpoint_manager import ConversationCheckpointManager

ckpt_mgr = ConversationCheckpointManager(storage_dir="checkpoints")

# Create a checkpoint for the current conversation turn
checkpoint = ckpt_mgr.create_checkpoint(
    checkpoint_id="session-001",
    step_index=1,
    turn_history=[{"role": "user", "content": "Fetch metrics"}],
    effect_ledger={"sha256_hash": {"status": "executed"}},
)
print(f"Checkpoint persisted: {checkpoint.checkpoint_id}")
```
