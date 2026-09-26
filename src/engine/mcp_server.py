"""Model Context Protocol (MCP) Server Adapter for Parallax Boundary.

Exposes Athena's FinOps tools (duckdb_query, athena_query, retrieval_search,
sandbox_execute, ping, echo) over the Model Context Protocol (MCP - JSON-RPC 2.0).

Enables standard agent frameworks (Claude Desktop, Cursor, LangChain MCP adapters,
CrewAI, and custom agents) to connect to Athena Reasoning Sandbox as an external
tool provider over stdio or programmatic message passing.

Algorithmic Complexity:
    - Message Parsing & Handling: O(N) where N is JSON message byte length.
    - Tool Dispatch: Delegated to ParallaxToolDispatcher / ExecutiveEngineProcess.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from typing import Any

from src.engine.dispatcher import ParallaxToolDispatcher
from src.engine.executor import ExecutiveEngineProcess
from src.reasoning.schemas import ToolCallPayload

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "athena-reasoning-sandbox"
SERVER_VERSION = "0.1.0"

# Standard JSON Schemas for Athena tools exposed via MCP
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "duckdb_query": {
        "name": "duckdb_query",
        "description": (
            "Execute local analytical SQL queries with FinOps AST partition pruning "
            "and cost governance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute against local tables/Parquet files.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Validate and estimate cost without executing.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
    "describe_table": {
        "name": "describe_table",
        "description": "Inspect column names, data types, and metadata of a data lake table.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Target table name to inspect schema.",
                },
            },
            "required": ["table_name"],
        },
    },
    "athena_query": {
        "name": "athena_query",
        "description": (
            "Execute analytical SQL queries on AWS Athena with FinOps AST query guard "
            "preventing unbudgeted full table scans."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute.",
                },
                "database": {
                    "type": "string",
                    "description": "Athena database name.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Simulate and validate query without AWS spend.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
    "retrieval_search": {
        "name": "retrieval_search",
        "description": (
            "Hybrid document search combining BM25 keyword matching and dense embeddings "
            "via Reciprocal Rank Fusion (RRF). Supports indexing, querying, and JSON persistence."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["search", "index", "clear", "export", "get", "delete", "save", "load"],
                    "description": "Operation type (default 'search').",
                    "default": "search",
                },
                "query": {
                    "type": "string",
                    "description": "Search query text (required for 'search' op).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return.",
                    "default": 5,
                },
                "doc_id": {
                    "type": "string",
                    "description": "Document identifier (required for 'get' and 'delete' ops).",
                },
                "filepath": {
                    "type": "string",
                    "description": "JSON file path (required for 'save' and 'load' ops).",
                },
                "documents": {
                    "type": "array",
                    "description": "List of document objects to ingest (required for 'index' op).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doc_id": {"type": "string"},
                            "content": {"type": "string"},
                            "metadata": {"type": "object"},
                        },
                        "required": ["doc_id", "content"],
                    },
                },
                "filter_metadata": {
                    "type": "object",
                    "description": "Optional key-value metadata dictionary to filter search results.",
                },
            },
        },
    },
    "sandbox_execute": {
        "name": "sandbox_execute",
        "description": (
            "Execute isolated code in an E2B sandbox environment with cpu/memory quotas."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code string to execute.",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "bash"],
                    "description": "Language runtime for execution.",
                    "default": "python",
                },
            },
            "required": ["code"],
        },
    },
    "ping": {
        "name": "ping",
        "description": "Health check ping to verify Executive Engine availability.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    "echo": {
        "name": "echo",
        "description": "Echo payload back for connection latency and framing verification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message string to echo.",
                },
            },
        },
    },
}


class AthenaMCPServer:
    """Model Context Protocol (MCP) JSON-RPC 2.0 server.

    Connects standard MCP clients to the ParallaxToolDispatcher and
    ExecutiveEngineProcess, providing tool reflection and secure tool execution.

    Attributes:
        dispatcher: Attached ParallaxToolDispatcher instance.
        protocol_version: Negotiated MCP protocol version string.
        server_name: Server identification string.
        server_version: Server semantic version string.
    """

    def __init__(
        self,
        dispatcher: ParallaxToolDispatcher | None = None,
        executor: ExecutiveEngineProcess | None = None,
        server_name: str = SERVER_NAME,
        server_version: str = SERVER_VERSION,
    ) -> None:
        """Initialize the Athena MCP Server.

        Args:
            dispatcher: Optional ParallaxToolDispatcher instance.
            executor: Optional ExecutiveEngineProcess instance (used if dispatcher is None).
            server_name: Name reported in serverInfo.
            server_version: Version reported in serverInfo.
        """
        if dispatcher is not None:
            self.dispatcher = dispatcher
        elif executor is not None:
            self.dispatcher = ParallaxToolDispatcher(executor=executor)
        else:
            self.dispatcher = ParallaxToolDispatcher()

        self.server_name = server_name
        self.server_version = server_version
        self.protocol_version = MCP_PROTOCOL_VERSION
        self._is_initialized = False

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the list of tool definitions currently authorized by the executor.

        Filters the canonical tool schemas against the Executive allowlist so unauthorized
        tools are not advertised to clients.

        Returns:
            List of MCP-compliant tool schema dictionaries.
        """
        authorized = self.dispatcher.executor.authorized_tools
        tools: list[dict[str, Any]] = []
        for name, schema in TOOL_SCHEMAS.items():
            if name in authorized:
                tools.append(schema)
        return tools

    async def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call to the ParallaxToolDispatcher.

        Args:
            name: Tool identifier.
            arguments: Tool arguments dictionary.

        Returns:
            MCP tool call result dictionary with content blocks and isError flag.
        """
        if not self.dispatcher.executor.is_authorized(name):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Tool '{name}' is not authorized on the Executive allowlist.",
                    }
                ],
                "isError": True,
            }

        call_id = f"mcp-{uuid.uuid4()}"
        payload = ToolCallPayload(
            call_id=call_id,
            tool_name=name,
            arguments=arguments,
            timeout_seconds=30.0,
        )

        obs = await self.dispatcher.dispatch_tool_call(payload)

        if obs.success:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(obs.output_data, ensure_ascii=False, indent=2),
                    }
                ],
                "isError": False,
            }
        else:
            err_msg = obs.error_message or "Tool execution failed."
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Execution Error: {err_msg}",
                    }
                ],
                "isError": True,
            }

    async def handle_message(self, message: dict[str, Any] | str) -> dict[str, Any] | None:
        """Handle a single MCP JSON-RPC 2.0 message.

        Args:
            message: Raw JSON string or parsed message dictionary.

        Returns:
            JSON-RPC response dictionary, or None for notifications.
        """
        if isinstance(message, str):
            try:
                data = json.loads(message)
            except json.JSONDecodeError as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {exc!s}",
                    },
                }
        else:
            data = message

        if not isinstance(data, dict) or data.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": data.get("id") if isinstance(data, dict) else None,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: Must be a JSON-RPC 2.0 object.",
                },
            }

        msg_id = data.get("id")
        method = data.get("method")
        params = data.get("params", {}) or {}

        # Handle notifications (no id)
        if msg_id is None:
            if method == "notifications/initialized":
                self._is_initialized = True
                logger.info("MCP client acknowledged initialization.")
            return None

        # Handle standard RPC methods
        if method == "initialize":
            client_version = params.get("protocolVersion", MCP_PROTOCOL_VERSION)
            self.protocol_version = client_version
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {
                        "tools": {
                            "listChanged": False,
                        },
                    },
                    "serverInfo": {
                        "name": self.server_name,
                        "version": self.server_version,
                    },
                },
            }

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {},
            }

        elif method == "tools/list":
            tools = self.list_tools()
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": tools,
                },
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            if not tool_name or not isinstance(arguments, dict):
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: 'name' (string) and 'arguments' (dict) are required.",
                    },
                }

            result = await self.handle_tool_call(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result,
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: '{method}'",
                },
            }

    async def run_stdio(self) -> None:
        """Run the MCP server over standard input and output (stdio transport).

        Reads newline-delimited JSON-RPC messages from sys.stdin, dispatches them
        asynchronously, and writes JSON-RPC responses to sys.stdout.
        """
        logger.info("Starting Athena MCP Server on stdio transport...")
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break  # EOF reached

                text = line.decode("utf-8").strip()
                if not text:
                    continue

                response = await self.handle_message(text)
                if response is not None:
                    out = json.dumps(response, ensure_ascii=False)
                    sys.stdout.write(out + "\n")
                    sys.stdout.flush()

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Error processing MCP stdio message: %s", exc)


def run_mcp_server() -> None:
    """Synchronous entry point to run the MCP server via asyncio."""
    server = AthenaMCPServer()
    try:
        asyncio.run(server.run_stdio())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_mcp_server()
