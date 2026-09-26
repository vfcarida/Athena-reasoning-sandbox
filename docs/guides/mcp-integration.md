# Model Context Protocol (MCP) Integration Guide

This guide describes how to configure and connect external AI assistants and IDEs (such as Claude Desktop, Cursor, and custom Python agents) to the **Athena Reasoning Sandbox** via the Model Context Protocol (MCP - JSON-RPC 2.0).

---

## 1. Overview

The Model Context Protocol (MCP) enables frontier models and AI development environments to discover and invoke local executive tools safely. When Athena runs in MCP mode, it acts as an executive tool server exposing:

- **`duckdb_query`**: Local analytical SQL with automatic FinOps partition pruning.
- **`athena_query`**: AWS Athena analytical queries with pre-execution cost guards.
- **`retrieval_search`**: Stateful hybrid BM25 + dense vector document search with JSON persistence.
- **`sandbox_execute`**: Isolated subprocess execution with memory and CPU quotas.
- **`ping` / `echo`**: Health check and roundtrip latency verification.

```
┌────────────────────────────┐
│ External MCP Host (Client) │  (Claude Desktop, Cursor, Custom Agent)
└──────────────┬─────────────┘
               │ JSON-RPC 2.0 over stdio
┌──────────────▼─────────────┐
│      AthenaMCPServer       │  (src/engine/mcp_server.py)
└──────────────┬─────────────┘
               │ Parallax Boundary
┌──────────────▼─────────────┐
│   ExecutiveEngineProcess   │  (Allowlist, AST Guards, Isolation)
└────────────────────────────┘
```

---

## 2. Launching the MCP Server

You can run the MCP server directly using the CLI entrypoint:

```bash
# Using installed package entrypoint:
athena-agent --mcp

# Or using the Python module:
python -m src.main --mcp
```

The server listens on `stdin` for JSON-RPC 2.0 messages and emits newline-delimited responses on `stdout`.

---

## 3. Host Configuration

### Claude Desktop

To integrate Athena tools into Claude Desktop, add the following to your `claude_desktop_config.json` (located at `%APPDATA%\Claude\claude_desktop_config.json` on Windows or `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "athena-reasoning-sandbox": {
      "command": "python",
      "args": [
        "-m",
        "src.main",
        "--mcp"
      ],
      "cwd": "/path/to/Athena-reasoning-sandbox"
    }
  }
}
```

### Cursor / IDE Configuration

In Cursor or VS Code MCP settings:
1. Navigate to **Settings** > **Features** > **MCP Servers**.
2. Add a new server entry:
   - **Name**: `athena-sandbox`
   - **Command**: `python -m src.main --mcp`
   - **Working Directory**: The workspace root directory.

---

## 4. Programmatic Python Client Example

You can interact with `AthenaMCPServer` in-memory or over subprocess pipes:

```python
import asyncio
from src.engine.mcp_server import AthenaMCPServer

async def main():
    server = AthenaMCPServer()

    # 1. Initialize Handshake
    init_res = await server.handle_message({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "my-client", "version": "1.0"},
        },
    })
    print("Capabilities:", init_res["result"]["capabilities"])

    # 2. List Authorized Tools
    tools_res = await server.handle_message({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
    })
    print("Available Tools:", [t["name"] for t in tools_res["result"]["tools"]])

    # 3. Call duckdb_query
    query_res = await server.handle_message({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "duckdb_query",
            "arguments": {
                "query": "SELECT 'Hello from MCP' AS greeting, '2026-09-01' AS dt;"
            },
        },
    })
    print("Query Output:", query_res["result"]["content"][0]["text"])

asyncio.run(main())
```

---

## 5. Testing & Verification

To verify the MCP server functionality without an external client:

```bash
# Run the interactive simulation demo:
python -m src.main --demo mcp

# Run the automated MCP test suite:
pytest tests/test_mcp_server.py -v
```
