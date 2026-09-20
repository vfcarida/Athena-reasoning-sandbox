"""Execution Sandbox Engine with Isolate-or-Refuse Honesty and Enforcement.

Standardizes code execution by isolating executive tool actions within real isolation backends
(E2B microVMs, local gVisor runsc, or OCI containers with CPU/memory quotas and network disabled).
Enforces an explicit isolate-or-refuse policy: if no isolated backend is available and healthy,
the engine refuses execution by raising SandboxUnavailableError rather than silently executing
unisolated commands on the host machine.

LOCAL_UNSAFE dev execution is strictly gated behind an explicit `allow_unsafe_local=True` flag
or `SANDBOX_ALLOW_UNSAFE_LOCAL=1` environment setting, and emits prominent security warnings.

Algorithmic Complexity:
    - Execution Overhead: O(1) container/microVM startup + O(T) script runtime.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.utils.config import config

try:
    import resource  # POSIX only
except ImportError:
    resource = None  # Windows or non-POSIX

logger = logging.getLogger(__name__)

SENSITIVE_ENV_PREFIXES = ("AWS_", "E2B_", "OPENAI_", "WANDB_", "HF_", "HUGGING_FACE_")
SENSITIVE_ENV_SUBSTRINGS = ("SECRET", "KEY", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH")


class IsolationBackend(str, Enum):
    """Supported sandbox isolation backends."""
    E2B = "e2b"                   # Cloud microVM via E2B SDK
    GVISOR = "gvisor"             # Local gVisor runsc runtime
    CONTAINER = "container"       # Local container runtime (Docker / Podman)
    LOCAL_UNSAFE = "local_unsafe" # Unisolated dev/testing fallback (strictly gated)


class SandboxUnavailableError(RuntimeError):
    """Raised when no verified isolation backend is available and unsafe local execution is not permitted."""


class SandboxSecurityError(RuntimeError):
    """Raised when an execution violates sandbox isolation or security boundaries."""


class SandboxQuotaViolationError(RuntimeError):
    """Raised when an execution violates configured CPU, memory, or timeout quotas."""


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
    """Execution output container from isolated sandbox execution.

    Attributes:
        exit_code: Process return code (0 = success).
        stdout: Standard output string.
        stderr: Standard error string.
        execution_time_ms: Runtime latency in milliseconds.
        resource_used_mb: Peak memory usage in MB.
        backend_used: Name of the isolation backend that executed the code.
    """
    exit_code: int
    stdout: str
    stderr: str
    execution_time_ms: float
    resource_used_mb: float = 0.0
    backend_used: str = ""


def sanitize_environment(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Construct a sanitized environment dictionary stripping host secrets.

    Args:
        extra_env: Optional caller-specified environment variables.

    Returns:
        Sanitized environment dictionary safe for sandboxed subprocesses.
    """
    allowed_keys = {
        "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
        "USERPROFILE", "HOME", "LANG", "LC_ALL", "PYTHONPATH",
    }
    clean_env: dict[str, str] = {}
    for k, v in os.environ.items():
        k_upper = k.upper()
        if k_upper in allowed_keys:
            clean_env[k] = v
        elif any(k_upper.startswith(prefix) for prefix in SENSITIVE_ENV_PREFIXES) or any(sub in k_upper for sub in SENSITIVE_ENV_SUBSTRINGS):
            continue

    if extra_env:
        clean_env.update(extra_env)
    return clean_env


class E2BSandboxEngine:
    """Standardized Sandbox Execution Engine with Isolate-or-Refuse enforcement.

    Attributes:
        quota: Active ResourceQuota instance.
        api_key: Optional E2B API Key.
        backend: Explicit or resolved IsolationBackend.
        allow_unsafe_local: Whether unisolated local execution is explicitly permitted.
    """

    def __init__(
        self,
        quota: ResourceQuota | None = None,
        api_key: str | None = None,
        backend: IsolationBackend | str | None = None,
        allow_unsafe_local: bool | None = None,
    ) -> None:
        """Initialize the Sandbox Execution Engine.

        Args:
            quota: Custom ResourceQuota configuration.
            api_key: Optional E2B API Key override.
            backend: Optional explicit IsolationBackend preference.
            allow_unsafe_local: Explicit flag to allow unisolated local fallback (dev only).
        """
        self.quota = quota or ResourceQuota()
        self.api_key = api_key or config.e2b_api_key

        # Resolve allow_unsafe_local flag
        if allow_unsafe_local is not None:
            self.allow_unsafe_local = allow_unsafe_local
        elif os.environ.get("SANDBOX_ALLOW_UNSAFE_LOCAL", "").lower() in ("1", "true", "yes"):
            self.allow_unsafe_local = True
        else:
            self.allow_unsafe_local = config.sandbox_allow_unsafe_local

        # Explicit backend override if requested
        self.configured_backend = IsolationBackend(backend) if backend else None

        logger.info(
            "SandboxEngine initialized (CPU=%.1fvCPU, RAM=%dMB, timeout=%.1fs, egress=%s, allow_unsafe_local=%s)",
            self.quota.max_cpu_cores,
            self.quota.max_memory_mb,
            self.quota.timeout_seconds,
            self.quota.allow_network_egress,
            self.allow_unsafe_local,
        )

    @staticmethod
    def is_gvisor_available() -> bool:
        """Check whether local gVisor runsc runtime is installed."""
        return shutil.which("runsc") is not None

    @staticmethod
    def is_container_available() -> bool:
        """Check whether a container runtime (docker/podman) is available and responsive."""
        runtime = shutil.which("docker") or shutil.which("podman")
        if not runtime:
            return False
        try:
            # Check if daemon is active with a short timeout
            res = subprocess.run(
                [runtime, "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.5,
                check=False,
            )
            return res.returncode == 0
        except Exception:  # noqa: BLE001
            return False

    def is_e2b_available(self) -> bool:
        """Check whether E2B microVM credentials are present."""
        return bool(self.api_key and self.api_key.strip())

    def resolve_backend(self) -> IsolationBackend:
        """Resolve the effective sandbox isolation backend using isolate-or-refuse policy.

        Resolution order:
            1. Explicit configured_backend (validated for availability).
            2. E2B microVM (if keyed).
            3. gVisor runsc (if installed).
            4. Container runtime (docker/podman, if daemon is responding).
            5. LOCAL_UNSAFE (only if allow_unsafe_local is True).
            6. Refuse with SandboxUnavailableError.

        Returns:
            Resolved IsolationBackend enum.

        Raises:
            SandboxUnavailableError: If no verified isolation backend is available.
        """
        if self.configured_backend:
            target = self.configured_backend
            if target == IsolationBackend.LOCAL_UNSAFE:
                if not self.allow_unsafe_local:
                    raise SandboxUnavailableError(
                        "LOCAL_UNSAFE execution requested but allow_unsafe_local is False. "
                        "Refusing unisolated code execution."
                    )
                return IsolationBackend.LOCAL_UNSAFE
            if target == IsolationBackend.E2B and not self.is_e2b_available():
                raise SandboxUnavailableError("E2B microVM backend requested but no E2B_API_KEY is configured.")
            if target == IsolationBackend.GVISOR and not self.is_gvisor_available():
                raise SandboxUnavailableError("gVisor backend requested but 'runsc' was not found on system PATH.")
            if target == IsolationBackend.CONTAINER and not self.is_container_available():
                raise SandboxUnavailableError("Container backend requested but container runtime is not running.")
            return target

        # Default automatic resolution
        if self.is_e2b_available():
            return IsolationBackend.E2B

        if self.is_gvisor_available():
            return IsolationBackend.GVISOR

        if self.is_container_available():
            return IsolationBackend.CONTAINER

        # No real isolation backend available
        if self.allow_unsafe_local:
            logger.warning(
                "SECURITY WARNING: No isolated backend available (E2B microVM, gVisor, or Container). "
                "Executing in LOCAL_UNSAFE mode because allow_unsafe_local=True."
            )
            return IsolationBackend.LOCAL_UNSAFE

        raise SandboxUnavailableError(
            "No isolated sandbox backend (E2B microVM, gVisor, or Container) is available and healthy. "
            "To prevent unisolated code execution on the host, sandbox execution is refused. "
            "To explicitly allow unisolated execution for local development, pass allow_unsafe_local=True "
            "or set SANDBOX_ALLOW_UNSAFE_LOCAL=1."
        )

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        environment_vars: dict[str, str] | None = None,
    ) -> SandboxExecutionResult:
        """Execute arbitrary code within an isolated sandbox environment.

        Args:
            code: Source code string to execute.
            language: Programming language ('python' or 'bash').
            environment_vars: Optional environment variables dictionary.

        Returns:
            SandboxExecutionResult object with execution metadata.

        Raises:
            SandboxUnavailableError: If no verified isolation backend is available.
        """
        start_time = time.perf_counter()
        backend = self.resolve_backend()

        if backend == IsolationBackend.E2B:
            return await self._execute_e2b_cloud(code, language, environment_vars, start_time)
        elif backend == IsolationBackend.CONTAINER:
            return await self._execute_container(code, language, environment_vars, start_time)
        elif backend == IsolationBackend.GVISOR:
            return await self._execute_gvisor(code, language, environment_vars, start_time)
        elif backend == IsolationBackend.LOCAL_UNSAFE:
            return await self._execute_local_unsafe(code, language, environment_vars, start_time)

        raise SandboxUnavailableError(f"Unsupported sandbox backend: {backend}")

    async def _execute_e2b_cloud(
        self,
        code: str,
        language: str,
        env_vars: dict[str, str] | None,
        start_time: float,
    ) -> SandboxExecutionResult:
        """Execute code via E2B Cloud MicroVM Sandbox."""
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
                backend_used=IsolationBackend.E2B.value,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error("E2B microVM execution failed: %s", exc)
            if self.allow_unsafe_local:
                logger.warning("E2B failed; falling back to LOCAL_UNSAFE because allow_unsafe_local=True.")
                return await self._execute_local_unsafe(code, language, env_vars, start_time)
            raise SandboxUnavailableError(
                f"E2B cloud microVM execution failed: {exc}. Refusing unisolated local execution."
            ) from exc

    async def _execute_container(
        self,
        code: str,
        language: str,
        env_vars: dict[str, str] | None,
        start_time: float,
    ) -> SandboxExecutionResult:
        """Execute code in an isolated OCI container with resource limits and network policy."""
        runtime = shutil.which("docker") or shutil.which("podman")
        if not runtime:
            raise SandboxUnavailableError("Container runtime (docker/podman) executable not found.")

        cmd = [
            runtime, "run", "--rm", "-i",
            f"--memory={self.quota.max_memory_mb}m",
            f"--cpus={self.quota.max_cpu_cores}",
        ]

        if not self.quota.allow_network_egress:
            cmd.extend(["--network", "none"])

        if language.lower() in ("python", "python3"):
            cmd.extend(["python:3.11-slim", "python", "-c", code])
        else:
            cmd.extend(["alpine:latest", "sh", "-c", code])

        clean_env = sanitize_environment(env_vars)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=clean_env,
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
                backend_used=IsolationBackend.CONTAINER.value,
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                exit_code=124,
                stdout="",
                stderr=f"Sandbox quota violation: Execution timed out after {self.quota.timeout_seconds}s.",
                execution_time_ms=elapsed_ms,
                backend_used=IsolationBackend.CONTAINER.value,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Container execution failure: {exc}",
                execution_time_ms=elapsed_ms,
                backend_used=IsolationBackend.CONTAINER.value,
            )

    async def _execute_gvisor(
        self,
        code: str,
        language: str,
        env_vars: dict[str, str] | None,
        start_time: float,
    ) -> SandboxExecutionResult:
        """Execute code via gVisor runsc runtime."""
        runsc = shutil.which("runsc")
        if not runsc:
            raise SandboxUnavailableError("gVisor runsc executable not found on system PATH.")

        cmd = [runsc, "do"]
        if not self.quota.allow_network_egress:
            cmd.extend(["--network", "none"])

        if language.lower() in ("python", "python3"):
            cmd.extend(["python", "-c", code])
        else:
            cmd.extend(["bash", "-c", code])

        clean_env = sanitize_environment(env_vars)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=clean_env,
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
                backend_used=IsolationBackend.GVISOR.value,
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                exit_code=124,
                stdout="",
                stderr=f"Sandbox quota violation: Execution timed out after {self.quota.timeout_seconds}s.",
                execution_time_ms=elapsed_ms,
                backend_used=IsolationBackend.GVISOR.value,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"gVisor execution failure: {exc}",
                execution_time_ms=elapsed_ms,
                backend_used=IsolationBackend.GVISOR.value,
            )

    async def _execute_local_unsafe(
        self,
        code: str,
        language: str,
        env_vars: dict[str, str] | None,
        start_time: float,
    ) -> SandboxExecutionResult:
        """Execute code via unisolated local subprocess (dev/testing mode ONLY).

        Raises:
            SandboxUnavailableError: If allow_unsafe_local is False.
        """
        # Hard security gate: Never run unisolated without explicit permission
        if not self.allow_unsafe_local:
            raise SandboxUnavailableError(
                "Refusing unisolated local execution: allow_unsafe_local is False. "
                "Pass allow_unsafe_local=True to explicitly permit local unisolated execution."
            )

        logger.warning(
            "SECURITY WARNING: Running in LOCAL_UNSAFE mode without process or network isolation. "
            "Host system resources and network are NOT protected."
        )

        cmd: list[str]
        if language.lower() in ("python", "python3"):
            cmd = ["python", "-c", code]
        else:
            cmd = ["bash", "-c", code]

        clean_env = sanitize_environment(env_vars)

        def preexec_limits() -> None:
            """Apply resource limits on POSIX systems if supported."""
            if resource is not None:
                try:
                    mem_bytes = self.quota.max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                except Exception as rlim_err:  # noqa: BLE001
                    logger.debug("Could not set RLIMIT_AS: %s", rlim_err)

        try:
            # Note: preexec_fn is supported only on POSIX systems
            kwargs: dict[str, Any] = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                "env": clean_env,
            }
            if os.name == "posix" and resource is not None:
                kwargs["preexec_fn"] = preexec_limits

            process = await asyncio.create_subprocess_exec(
                *cmd,
                **kwargs,
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
                backend_used=IsolationBackend.LOCAL_UNSAFE.value,
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                exit_code=124,
                stdout="",
                stderr=f"Sandbox quota violation: Execution timed out after {self.quota.timeout_seconds}s.",
                execution_time_ms=elapsed_ms,
                backend_used=IsolationBackend.LOCAL_UNSAFE.value,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"Local sandbox execution failure: {exc!s}",
                execution_time_ms=elapsed_ms,
                backend_used=IsolationBackend.LOCAL_UNSAFE.value,
            )
