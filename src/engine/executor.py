"""Executive Engine Process for Isolated Tool Execution.

Implements the Executive Layer under the Cognitive-Executive Separation (Parallax) principle.
The LLM reasoning environment is prevented from making direct system/OS operations.
All requested actions pass across the Parallax boundary to this process, which enforces
security isolation, authorization allowlists, and structured output parsing.

Algorithmic Complexity:
    - Tool Dispatch: O(1) dictionary lookup for registered tool handles.
    - Authorization Check: O(1) set membership check.
    - Execution Overhead: O(T) where T is tool runtime.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from src.reasoning.schemas import ObservationPayload, ToolCallPayload

if TYPE_CHECKING:
    from src.athena.athena_client import AthenaClient
    from src.athena.duckdb_client import DuckDBClient
    from src.rag.retrieval_index import RetrievalIndex
    from src.sandbox.e2b_sandbox import E2BSandboxEngine

logger = logging.getLogger(__name__)

# Type definition for async tool handler callables
ToolHandlerCallable = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

DEFAULT_AUTHORIZED_TOOLS: set[str] = {
    "ping",
    "echo",
    "athena_query",
    "duckdb_query",
    "sandbox_execute",
    "retrieval_search",
}


class ExecutiveEngineProcess:
    """Isolated process executing authorized system tools on behalf of the Cognitive Layer.

    The Executive process maintains an explicit registry of tool handlers and enforces
    a deny-by-default authorization allowlist before dispatching any execution requests.

    Attributes:
        _registry: Internal map of tool identifiers to handler functions.
        _authorized_tools: Explicit set of tool names permitted for execution.
    """

    def __init__(
        self,
        athena_client: AthenaClient | None = None,
        duckdb_client: DuckDBClient | None = None,
        sandbox_engine: E2BSandboxEngine | None = None,
        retrieval_index: RetrievalIndex | None = None,
        authorized_tools: set[str] | None = None,
    ) -> None:
        """Initialize the Executive Engine Process with default tools and allowlist.

        Args:
            athena_client: Optional AthenaClient instance for executing Athena queries.
            duckdb_client: Optional DuckDBClient instance for executing local analytical queries.
            sandbox_engine: Optional E2BSandboxEngine instance for executing sandboxed code.
            retrieval_index: Optional RetrievalIndex for hybrid BM25 + dense document search.
                If None, a default in-process index is created lazily on first use.
            authorized_tools: Optional set of authorized tool names. If None, defaults to
                the standard system tools (ping, echo, athena_query, duckdb_query,
                sandbox_execute, retrieval_search). If an empty set is passed, all tools
                are denied by default.
        """
        self._registry: dict[str, ToolHandlerCallable] = {}
        self._athena_client = athena_client
        self._duckdb_client = duckdb_client
        self._sandbox_engine = sandbox_engine
        self._retrieval_index = retrieval_index

        if authorized_tools is not None:
            self._authorized_tools: set[str] = set(authorized_tools)
        else:
            self._authorized_tools = set(DEFAULT_AUTHORIZED_TOOLS)

        self._register_default_handlers()

    @property
    def authorized_tools(self) -> set[str]:
        """Return a read-only copy of currently authorized tools."""
        return set(self._authorized_tools)

    def is_authorized(self, tool_name: str) -> bool:
        """Check if a tool name is currently authorized on the allowlist."""
        return tool_name in self._authorized_tools

    def authorize_tool(self, tool_name: str) -> None:
        """Add a tool name to the authorized tools allowlist.

        Args:
            tool_name: Identifier of the tool to authorize.
        """
        self._authorized_tools.add(tool_name)
        logger.info("Authorized executive tool: '%s'", tool_name)

    def revoke_tool(self, tool_name: str) -> None:
        """Remove a tool name from the authorized tools allowlist.

        Args:
            tool_name: Identifier of the tool to revoke authorization for.
        """
        self._authorized_tools.discard(tool_name)
        logger.info("Revoked authorization for executive tool: '%s'", tool_name)

    def register_tool(
        self,
        tool_name: str,
        handler: ToolHandlerCallable,
        authorize: bool = True,
    ) -> None:
        """Register a new tool handler.

        Args:
            tool_name: Unique string name of the tool.
            handler: Asynchronous callable receiving arguments dict and returning result dict.
            authorize: If True, automatically adds the tool to the authorized allowlist.
        """
        self._registry[tool_name] = handler
        if authorize:
            self.authorize_tool(tool_name)
        logger.info("Registered executive tool handler: '%s' (authorized=%s)", tool_name, authorize)

    def _register_default_handlers(self) -> None:
        """Register built-in tool handlers for system tasks, Athena, and Sandbox."""
        async def ping_handler(args: dict[str, Any]) -> dict[str, Any]:
            return {"status": "pong", "received_args": args}

        async def echo_handler(args: dict[str, Any]) -> dict[str, Any]:
            return {"echo": args.get("message", "")}

        async def athena_query_handler(args: dict[str, Any]) -> dict[str, Any]:
            client = self._athena_client
            if client is None:
                from src.athena.athena_client import AthenaClient
                client = AthenaClient()
                self._athena_client = client

            sql = args.get("sql") or args.get("query")
            if not sql:
                raise ValueError("Missing required 'sql' or 'query' parameter in tool arguments.")

            required_partition_keys = args.get("required_partition_keys")
            enforce_limit = args.get("enforce_limit", True)
            dry_run = args.get("dry_run", False)
            wait = args.get("wait", False)
            poll_interval = args.get("poll_interval", 0.5)
            timeout_seconds = args.get("timeout_seconds", 30.0)

            # AthenaClient.execute_query is synchronous, run in thread pool
            result = await asyncio.to_thread(
                client.execute_query,
                sql=sql,
                required_partition_keys=required_partition_keys,
                enforce_limit=enforce_limit,
                dry_run=dry_run,
                wait=wait,
                poll_interval=poll_interval,
                timeout_seconds=timeout_seconds,
            )
            return result

        async def sandbox_execute_handler(args: dict[str, Any]) -> dict[str, Any]:
            engine = self._sandbox_engine
            if engine is None:
                from src.sandbox.e2b_sandbox import E2BSandboxEngine
                engine = E2BSandboxEngine()
                self._sandbox_engine = engine

            code = args.get("code")
            if not code:
                raise ValueError("Missing required 'code' parameter in tool arguments.")

            language = args.get("language", "python")
            env_vars = args.get("environment_vars")

            result = await engine.execute_code(
                code=code,
                language=language,
                environment_vars=env_vars,
            )

            # Require exit_code == 0 for clean success; non-zero exit codes propagate as errors
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Sandbox process failed with exit code {result.exit_code}: "
                    f"{result.stderr or result.stdout}"
                )

            return {
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time_ms": result.execution_time_ms,
                "backend_used": result.backend_used,
            }

        async def duckdb_query_handler(args: dict[str, Any]) -> dict[str, Any]:
            client = self._duckdb_client
            if client is None:
                from src.athena.duckdb_client import DuckDBClient
                client = DuckDBClient()
                self._duckdb_client = client

            sql = args.get("sql") or args.get("query")
            if not sql:
                raise ValueError("Missing required 'sql' or 'query' parameter in tool arguments.")

            required_partition_keys = args.get("required_partition_keys")
            enforce_limit = args.get("enforce_limit", True)
            dry_run = args.get("dry_run", False)

            return await client.async_execute_query(
                sql=sql,
                required_partition_keys=required_partition_keys,
                enforce_limit=enforce_limit,
                dry_run=dry_run,
            )

        async def retrieval_search_handler(args: dict[str, Any]) -> dict[str, Any]:
            """Handle retrieval_search tool calls.

            Supports three operations selected by the ``operation`` argument key:

            * ``"search"`` (default) — query the index.  Required: ``query`` (str).
              Optional: ``query_embedding`` (list[float]), ``top_k`` (int, default 5).
            * ``"index"`` — add documents to the index.  Required: ``documents``
              (list of dicts, each with ``doc_id`` and ``content`` keys).
            * ``"clear"`` — remove all documents from the index.

            Args:
                args: Tool argument payload from the Parallax boundary.

            Returns:
                JSON-safe result dict for ObservationPayload.output_data.

            Raises:
                ValueError: On missing required arguments or empty index search.
            """
            # Lazy-init in-process index
            index = self._retrieval_index
            if index is None:
                from src.rag.retrieval_index import RetrievalIndex
                index = RetrievalIndex()
                self._retrieval_index = index

            operation = args.get("operation", "search")

            if operation == "index":
                documents = args.get("documents")
                if not documents or not isinstance(documents, list):
                    raise ValueError(
                        "'retrieval_search' with operation='index' requires a non-empty "
                        "'documents' list argument."
                    )
                count = index.index_documents(documents)
                return {"operation": "index", "indexed_count": count, "total_docs": index.document_count}

            if operation == "clear":
                index.clear()
                return {"operation": "clear", "total_docs": 0}

            # Default: search
            query = args.get("query")
            if not query or not isinstance(query, str):
                raise ValueError(
                    "'retrieval_search' with operation='search' requires a non-empty 'query' string."
                )
            top_k = int(args.get("top_k", 5))
            query_embedding: list[float] | None = args.get("query_embedding")
            results = index.search(query=query, query_embedding=query_embedding, top_k=top_k)
            return index.to_tool_result(results)

        self._registry["ping"] = ping_handler
        self._registry["echo"] = echo_handler
        self._registry["athena_query"] = athena_query_handler
        self._registry["duckdb_query"] = duckdb_query_handler
        self._registry["sandbox_execute"] = sandbox_execute_handler
        self._registry["retrieval_search"] = retrieval_search_handler

    async def execute_tool_call(self, payload: ToolCallPayload) -> ObservationPayload:
        """Execute a tool call payload dispatched across the Parallax boundary.

        Enforces explicit authorization allowlists, hard timeouts, and structured output parsing.
        Denies execution by default if the tool is not in the authorized allowlist.

        Args:
            payload: Validated ToolCallPayload received from Cognitive Layer.

        Returns:
            ObservationPayload containing execution results or error details.
        """
        start_time = time.perf_counter()
        tool_name = payload.tool_name

        # 1. Registry check
        if tool_name not in self._registry:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            logger.warning("Attempted invocation of unregistered tool: '%s'", tool_name)
            return ObservationPayload(
                call_id=payload.call_id,
                success=False,
                output_data={},
                error_message=f"Unregistered executive tool: '{tool_name}'",
                execution_time_ms=elapsed,
            )

        # 2. Authorization check (deny-by-default)
        if not self.is_authorized(tool_name):
            elapsed = (time.perf_counter() - start_time) * 1000.0
            logger.warning("Unauthorized tool invocation attempted: '%s'", tool_name)
            return ObservationPayload(
                call_id=payload.call_id,
                success=False,
                output_data={},
                error_message=f"Unauthorized tool execution: '{tool_name}' is not in the authorized tools allowlist.",
                execution_time_ms=elapsed,
            )

        handler = self._registry[tool_name]

        try:
            # 3. Enforce hard timeout execution safety
            output_data = await asyncio.wait_for(
                handler(payload.arguments),
                timeout=payload.timeout_seconds,
            )
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ObservationPayload(
                call_id=payload.call_id,
                success=True,
                output_data=output_data,
                error_message=None,
                execution_time_ms=elapsed,
            )
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            logger.error("Tool execution timed out after %.2fs: '%s'", payload.timeout_seconds, tool_name)
            return ObservationPayload(
                call_id=payload.call_id,
                success=False,
                output_data={},
                error_message=f"Execution timed out after {payload.timeout_seconds} seconds.",
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            logger.exception("Executive error executing tool '%s'", tool_name)
            return ObservationPayload(
                call_id=payload.call_id,
                success=False,
                output_data={},
                error_message=f"{type(exc).__name__}: {exc!s}",
                execution_time_ms=elapsed,
            )
