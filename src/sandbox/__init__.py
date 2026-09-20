"""Standardized execution sandbox module with Isolate-or-Refuse enforcement.

Supports E2B microVMs, gVisor runsc, and container runtimes with strict CPU/memory/network quotas.
Refuses unisolated execution by default, requiring explicit allow_unsafe_local=True for dev testing.
"""

from src.sandbox.e2b_sandbox import (
    E2BSandboxEngine,
    IsolationBackend,
    ResourceQuota,
    SandboxExecutionResult,
    SandboxQuotaViolationError,
    SandboxSecurityError,
    SandboxUnavailableError,
)

__all__ = [
    "E2BSandboxEngine",
    "IsolationBackend",
    "ResourceQuota",
    "SandboxExecutionResult",
    "SandboxQuotaViolationError",
    "SandboxSecurityError",
    "SandboxUnavailableError",
]
