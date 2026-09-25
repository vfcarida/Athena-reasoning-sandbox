"""Unit and Integration Test Suite for Model Context Protocol (MCP) Server.

Verifies that AthenaMCPServer conforms to JSON-RPC 2.0 MCP standards:
1. Protocol negotiation and initialization ('initialize').
2. Client handshake acknowledgment ('notifications/initialized').
3. Liveness check ('ping').
4. Tool catalog inspection with schema reflection ('tools/list').
5. Deny-by-default allowlist enforcement.
6. Tool invocation execution ('tools/call').
7. Error propagation for unauthorized tools, invalid params, and execution failures.
8. JSON-RPC 2.0 protocol error handling.
"""

from __future__ import annotations

import pytest

from src.athena.duckdb_client import DuckDBClient
from src.engine.executor import ExecutiveEngineProcess
from src.engine.mcp_server import AthenaMCPServer


@pytest.fixture
def mcp_server() -> AthenaMCPServer:
    """Fixture providing a standard AthenaMCPServer instance with seeded DuckDB data."""
    client = DuckDBClient(database=":memory:")
    client.con.execute("""
        CREATE TABLE sales (
            order_id INTEGER,
            customer_id VARCHAR,
            amount DOUBLE,
            dt VARCHAR
        );
        INSERT INTO sales VALUES
            (1, 'cust_101', 99.5, '2026-09-01'),
            (2, 'cust_102', 150.0, '2026-09-01');
    """)
    executor = ExecutiveEngineProcess(duckdb_client=client)
    return AthenaMCPServer(executor=executor)


class TestMCPProtocolLifecycle:
    """Verifies handshake and core RPC lifecycle methods."""

    @pytest.mark.asyncio
    async def test_initialize_handshake(self, mcp_server: AthenaMCPServer) -> None:
        """Verify initialize negotiates protocol version and returns capabilities."""
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }
        res = await mcp_server.handle_message(req)
        assert res is not None
        assert res["jsonrpc"] == "2.0"
        assert res["id"] == 1
        assert "result" in res

        result = res["result"]
        assert result["protocolVersion"] == "2024-11-05"
        assert result["serverInfo"]["name"] == "athena-reasoning-sandbox"
        assert "tools" in result["capabilities"]

    @pytest.mark.asyncio
    async def test_notifications_initialized(self, mcp_server: AthenaMCPServer) -> None:
        """Verify client initialized notification is acknowledged with no response."""
        req = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        res = await mcp_server.handle_message(req)
        assert res is None
        assert mcp_server._is_initialized is True

    @pytest.mark.asyncio
    async def test_ping(self, mcp_server: AthenaMCPServer) -> None:
        """Verify ping returns empty result block."""
        req = {
            "jsonrpc": "2.0",
            "id": "ping-123",
            "method": "ping",
        }
        res = await mcp_server.handle_message(req)
        assert res is not None
        assert res["id"] == "ping-123"
        assert res["result"] == {}


class TestMCPToolDiscovery:
    """Verifies schema listing and allowlist filtering."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_schemas(self, mcp_server: AthenaMCPServer) -> None:
        """Verify tools/list returns authorized tool definitions with inputSchemas."""
        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        }
        res = await mcp_server.handle_message(req)
        assert res is not None
        tools = res["result"]["tools"]
        assert len(tools) > 0

        tool_names = [t["name"] for t in tools]
        assert "duckdb_query" in tool_names
        assert "athena_query" in tool_names
        assert "retrieval_search" in tool_names
        assert "ping" in tool_names

        duckdb_tool = next(t for t in tools if t["name"] == "duckdb_query")
        assert "inputSchema" in duckdb_tool
        assert "query" in duckdb_tool["inputSchema"]["properties"]

    @pytest.mark.asyncio
    async def test_list_tools_respects_custom_allowlist(self) -> None:
        """Verify only explicitly authorized tools are listed."""
        custom_executor = ExecutiveEngineProcess(authorized_tools={"ping", "echo"})
        server = AthenaMCPServer(executor=custom_executor)

        req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
        res = await server.handle_message(req)
        assert res is not None
        tools = res["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        assert sorted(tool_names) == ["echo", "ping"]
        assert "duckdb_query" not in tool_names


class TestMCPToolExecution:
    """Verifies execution of tool calls across the Parallax boundary."""

    @pytest.mark.asyncio
    async def test_tools_call_ping(self, mcp_server: AthenaMCPServer) -> None:
        """Verify ping tool execution."""
        req = {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "ping",
                "arguments": {},
            },
        }
        res = await mcp_server.handle_message(req)
        assert res is not None
        result = res["result"]
        assert result["isError"] is False
        assert len(result["content"]) == 1
        assert "status" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tools_call_duckdb_success(self, mcp_server: AthenaMCPServer) -> None:
        """Verify valid duckdb_query executes and returns JSON content."""
        req = {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "duckdb_query",
                "arguments": {
                    "query": "SELECT order_id, amount FROM sales WHERE dt = '2026-09-01' LIMIT 10",
                },
            },
        }
        res = await mcp_server.handle_message(req)
        assert res is not None
        result = res["result"]
        assert result["isError"] is False
        assert "99.5" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tools_call_unauthorized_tool(self, mcp_server: AthenaMCPServer) -> None:
        """Verify unauthorized tool is blocked and returns isError: True."""
        req = {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "unauthorized_admin_drop_db",
                "arguments": {},
            },
        }
        res = await mcp_server.handle_message(req)
        assert res is not None
        result = res["result"]
        assert result["isError"] is True
        assert "not authorized" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tools_call_execution_failure_reports_error(self, mcp_server: AthenaMCPServer) -> None:
        """Verify execution errors (e.g. malformed SQL) are surfaced with isError: True."""
        req = {
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "duckdb_query",
                "arguments": {
                    "query": "SELECT FROM INVALID SYNTAX",
                },
            },
        }
        res = await mcp_server.handle_message(req)
        assert res is not None
        result = res["result"]
        assert result["isError"] is True
        assert "Execution Error" in result["content"][0]["text"]


class TestMCPProtocolErrors:
    """Verifies JSON-RPC 2.0 error responses."""

    @pytest.mark.asyncio
    async def test_parse_error(self, mcp_server: AthenaMCPServer) -> None:
        """Verify malformed JSON string returns -32700."""
        res = await mcp_server.handle_message("{malformed json:")
        assert res is not None
        assert res["error"]["code"] == -32700

    @pytest.mark.asyncio
    async def test_invalid_request_missing_jsonrpc(self, mcp_server: AthenaMCPServer) -> None:
        """Verify missing jsonrpc version returns -32600."""
        res = await mcp_server.handle_message({"id": 1, "method": "ping"})
        assert res is not None
        assert res["error"]["code"] == -32600

    @pytest.mark.asyncio
    async def test_method_not_found(self, mcp_server: AthenaMCPServer) -> None:
        """Verify unknown method returns -32601."""
        req = {"jsonrpc": "2.0", "id": 99, "method": "unknown_rpc_op"}
        res = await mcp_server.handle_message(req)
        assert res is not None
        assert res["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_invalid_tools_call_params(self, mcp_server: AthenaMCPServer) -> None:
        """Verify missing tool name returns -32602."""
        req = {"jsonrpc": "2.0", "id": 100, "method": "tools/call", "params": {}}
        res = await mcp_server.handle_message(req)
        assert res is not None
        assert res["error"]["code"] == -32602
