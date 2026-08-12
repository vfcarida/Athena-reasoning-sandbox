"""Executive Engine Process for Isolated Tool Execution.

Implements the Executive Layer under the Cognitive-Executive Separation (Parallax) principle.
The LLM reasoning environment is prevented from making direct system/OS operations.
All requested actions pass across the Parallax boundary to this process, which enforces
security isolation, authorization, and structured output parsing.

Algorithmic Complexity:
    - Tool Dispatch: O(1) dictionary lookup for registered tool handles.
    - Execution Overhead: O(T) where T is tool runtime.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from src.reasoning.schemas import ObservationPayload, ToolCallPayload

logger = logging.getLogger(__name__)

# Type definition for async tool handler callables
ToolHandlerCallable = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class ExecutiveEngineProcess:
    """Isolated process executing authorized system tools on behalf of the Cognitive Layer.

    The Executive process maintains an explicit registry of authorized tool handlers
    and executes them within defined security and timeout parameters.

    Attributes:
        _registry: Internal map of tool identifiers to handler functions.
    """

    def __init__(self) -> None:
        """Initialize the Executive Engine Process with default system tools."""
        self._registry: Dict[str, ToolHandlerCallable] = {}
        self._register_default_handlers()

    def register_tool(self, tool_name: str, handler: ToolHandlerCallable) -> None:
        """Register a new authorized tool handler.

        Args:
            tool_name: Unique string name of the tool.
            handler: Asynchronous callable receiving arguments dict and returning result dict.
        """
        self._registry[tool_name] = handler
        logger.info("Registered executive tool handler: '%s'", tool_name)

    def _register_default_handlers(self) -> None:
        """Register initial built-in tool handlers for system tasks."""
        async def ping_handler(args: Dict[str, Any]) -> Dict[str, Any]:
            return {"status": "pong", "received_args": args}

        async def echo_handler(args: Dict[str, Any]) -> Dict[str, Any]:
            return {"echo": args.get("message", "")}

        self.register_tool("ping", ping_handler)
        self.register_tool("echo", echo_handler)

    async def execute_tool_call(self, payload: ToolCallPayload) -> ObservationPayload:
        """Execute a tool call payload dispatched across the Parallax boundary.

        Enforces strict execution timeouts, catches exceptions, and guarantees
        a structured ObservationPayload return.

        Args:
            payload: Validated ToolCallPayload received from Cognitive Layer.

        Returns:
            ObservationPayload containing execution results or error details.
        """
        start_time = time.perf_counter()
        tool_name = payload.tool_name

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

        handler = self._registry[tool_name]

        try:
            # Enforce hard timeout execution safety
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
            logger.exception("Executive error executing tool '%s': %s", tool_name, exc)
            return ObservationPayload(
                call_id=payload.call_id,
                success=False,
                output_data={},
                error_message=f"Executive engine failure: {str(exc)}",
                execution_time_ms=elapsed,
            )
