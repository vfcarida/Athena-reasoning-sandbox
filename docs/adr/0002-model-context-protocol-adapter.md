# ADR 0002: Model Context Protocol (MCP) Executive Adapter Gateway

- **Status**: Accepted
- **Date**: 2026-09-25
- **Deciders**: Athena Reasoning Sandbox Architecture Team
- **Consulted**: Security, FinOps Engineering, AI Agent Core
- **Informed**: Framework Maintainers

---

## Context

The Athena Reasoning Sandbox architecture is grounded upon the **Parallax Principle** (Cognitive-Executive Separation), which establishes a strict security boundary between stochastic reasoning environments (LLMs, prompt orchestrators) and deterministic execution engines (AWS Athena, DuckDB, sandboxed code execution, file systems).

With the rapid emergence and industry adoption of the **Model Context Protocol (MCP)** specification (JSON-RPC 2.0 over stdio), external frontier agent platforms (such as Claude Desktop, Cursor, Windsurf, and custom autonomous agents) require a standardized mechanism to discover, inspect, and invoke domain-specific tools without bespoke integration code.

We need a clean architectural adapter that exposes Athena's FinOps-guarded tools (including SQL query execution, hybrid document retrieval, and sandboxed computation) to MCP clients while strictly upholding:
1. Deny-by-default allowlist enforcement.
2. AST partition pruning and cost governance.
3. Subprocess isolation and execution quotas.
4. OpenTelemetry distributed tracing across boundary transitions.

---

## Decision

We introduce `AthenaMCPServer` in `src/engine/mcp_server.py` as an adapter layer operating at the Parallax Executive boundary:

1. **Protocol Compliance**:
   - Implements JSON-RPC 2.0 per the MCP specification (`version 2024-11-05`).
   - Handles standard protocol lifecycle methods: `initialize`, `notifications/initialized`, `ping`, `tools/list`, and `tools/call`.

2. **Executive Layer Delegation**:
   - The MCP server does not execute operations directly. Instead, every `tools/call` is packaged into an immutable `ToolCallPayload` and dispatched to `ParallaxToolDispatcher` and `ExecutiveEngineProcess`.
   - The Executive process enforces allowlists, AST validation, and query guards before executing any operation.

3. **Dynamic Schema Reflection**:
   - The MCP server filters advertised tools against `ExecutiveEngineProcess.authorized_tools`. If an operator revokes authorization for a tool, it is automatically omitted from `tools/list` and rejected on invocation.

4. **CLI & Stdio Integration**:
   - Added `--mcp` / `--mcp-serve` flags to `src/main.py` and entrypoint `athena-agent --mcp`, enabling seamless plug-and-play configuration with MCP host applications.

---

## Consequences

### Positive
- **Ecosystem Interoperability**: Any MCP-compliant host can leverage Athena's analytical data lake capabilities and hybrid RAG tools with zero custom code.
- **Security Invariance**: The security posture remains uncompromising: the MCP adapter is purely a protocol translator and cannot bypass AST guards or allowlists.
- **Developer Ergonomics**: Operators can run Athena in server mode or test tools via interactive simulations (`python -m src.main --demo mcp`).

### Trade-offs & Mitigations
- **Transport Limitations**: Initial release focuses on stdio transport. Network-based transports (SSE / WebSocket) can be added in subsequent phases by wrapping the same `handle_message` asynchronous core.
- **Encoding Hazards**: Windows stdio can encounter character encoding mismatches; solved by forcing UTF-8 reconfiguration on process startup.
