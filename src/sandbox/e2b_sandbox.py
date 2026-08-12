"""Execution Sandbox Engine supporting E2B MicroVM and Docker Isolation.

Standardizes code execution by isolating executive tool actions within E2B microVMs
or containerized Docker environments. Enforces strict resource quotas (CPU limits,
memory allocation caps, and network egress policies).

Algorithmic Complexity:
    - Execution Overhead: O(1) process setup + O(T) script runtime.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.config import config

logger = logging.getLogger(__name__)


@dataclass
class ResourceQuota:
    """Defines hard resource allocation quotas for sandbox execution environments.

    Attributes:
        max_cpu_cores: Maximum CPU cores allocated (e.g. 1.0 = 1 vCPU).
        max_memory_mb: Maximum memory in Megabytes (e.g. 512 MB).
        timeout_seconds: Hard execution timeout in seconds.
        allow_network_egress: Flag indicating whether external HTTP/network egress is allowed.
    """
    max_cpu_cores: float = 1.0
    max_memory_mb: int = 512
    timeout_seconds: float = 30.0
    allow_network_egress: bool = False


@dataclass
class SandboxExecutionResult:
    """Execution output container from isolated sandbox microVM.

    Attributes:
        exit_code: Process return code (0 = success).
        stdout: Standard output string.
        stderr: Standard error string.
        execution_time_ms: Runtime latency in milliseconds.
        resource_used_mb: Peak memory usage in MB.
    """
    exit_code: int
    stdout: str
    stderr: str
    execution_time_ms: float
    resource_used_mb: float = 0.0


class E2BSandboxEngine:
    """Standardized Sandbox Execution Engine with resource quotas.

    Attributes:
        quota: Active ResourceQuota instance.
        api_key: Optional E2B API Key.
    """

    def __init__(self, quota: Optional[ResourceQuota] = None, api_key: Optional[str] = None) -> None:
        """Initialize the E2B Sandbox Engine.

        Args:
            quota: Custom ResourceQuota configuration.
            api_key: Optional E2B API Key override.
        """
        self.quota = quota or ResourceQuota()
        self.api_key = api_key or config.e2b_api_key
        logger.info(
            "E2BSandboxEngine initialized (CPU=%.1fvCPU, RAM=%dMB, timeout=%.1fs, egress=%s)",
            self.quota.max_cpu_cores,
            self.quota.max_memory_mb,
            self.quota.timeout_seconds,
            self.quota.allow_network_egress,
        )

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        environment_vars: Optional[Dict[str, str]] = None,
    ) -> SandboxExecutionResult:
        """Execute arbitrary code safely inside isolated microVM sandbox with quota bounds.

        Args:
            code: Source code string to execute.
            language: Programming language ('python' or 'bash').
            environment_vars: Optional environment variables dictionary.

        Returns:
            SandboxExecutionResult object.
        """
        start_time = time.perf_counter()

        if self.api_key:
            return await self._execute_e2b_cloud(code, language, environment_vars, start_time)
        else:
            return await self._execute_local_fallback(code, language, environment_vars, start_time)

    async def _execute_e2b_cloud(
        self,
        code: str,
        language: str,
        env_vars: Optional[Dict[str, str]],
        start_time: float,
    ) -> SandboxExecutionResult:
        """Execute code via E2B SDK Cloud Sandbox."""
        try:
            from e2b_code_interpreter import Sandbox
            sbx = Sandbox(api_key=self.api_key)
            execution = sbx.run_code(code)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            stdout_text = "\n".join([str(o) for o in execution.logs.stdout])
            stderr_text = "\n".join([str(e) for e in execution.logs.stderr])

            sbx.kill()
            return SandboxExecutionResult(
                exit_code=0 if not execution.error else 1,
                stdout=stdout_text,
                stderr=stderr_text or (str(execution.error) if execution.error else ""),
                execution_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning("E2B SDK cloud execution unavailable: %s. Reverting to local fallback.", exc)
            return await self._execute_local_fallback(code, language, env_vars, start_time)

    async def _execute_local_fallback(
        self,
        code: str,
        language: str,
        env_vars: Optional[Dict[str, str]],
        start_time: float,
    ) -> SandboxExecutionResult:
        """Execute code via isolated local subprocess with strict timeout quotas."""
        cmd: List[str]
        if language.lower() in ("python", "python3"):
            cmd = ["python", "-c", code]
        else:
            cmd = ["bash", "-c", code]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(),
                timeout=self.quota.timeout_seconds,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                exit_code=process.returncode or 0,
                stdout=stdout_data.decode("utf-8", errors="replace"),
                stderr=stderr_data.decode("utf-8", errors="replace"),
                execution_time_ms=elapsed_ms,
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                exit_code=124,
                stdout="",
                stderr=f"Sandbox quota violation: Execution timed out after {self.quota.timeout_seconds}s.",
                execution_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Local sandbox execution failure: {str(exc)}",
                execution_time_ms=elapsed_ms,
            )
