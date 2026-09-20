"""Unit test suite for Sandbox Isolation (Honesty + Enforcement) — ARS-T04.

Verifies:
- Isolate-or-Refuse policy: default unkeyed/unisolated environment refuses execution.
- E2B cloud failure does not silently fall back to host unisolated bash -c.
- Explicit IsolationBackend requests validate runtime health or raise SandboxUnavailableError.
- LOCAL_UNSAFE is strictly unreachable without allow_unsafe_local=True or env var.
- Environment sanitization strips AWS secrets and API keys from executed code.
- Container runtime passes CPU, memory, and network isolation flags.
- E1 Adversarial Suite: hostile snippets (network, disk, secret, fork bomb, memory) are blocked/refused.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.sandbox.e2b_sandbox import (
    E2BSandboxEngine,
    IsolationBackend,
    ResourceQuota,
    SandboxUnavailableError,
    sanitize_environment,
)

# ==============================================================================
# 1. Isolate-or-Refuse Default Policy Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_default_unkeyed_environment_refuses_execution():
    """Verify engine raises SandboxUnavailableError by default when unisolated."""
    with patch.object(E2BSandboxEngine, "is_container_available", return_value=False), \
         patch.object(E2BSandboxEngine, "is_gvisor_available", return_value=False):
        engine = E2BSandboxEngine(api_key="", allow_unsafe_local=False)
        with pytest.raises(SandboxUnavailableError) as exc_info:
            await engine.execute_code("print('Refuse me')", language="python")
        assert "No isolated sandbox backend" in str(exc_info.value)
        assert "refused" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_e2b_cloud_failure_refuses_silent_fallback():
    """Verify E2B failure does NOT silently fall back to host execution when allow_unsafe_local=False."""
    with patch.object(E2BSandboxEngine, "is_container_available", return_value=False), \
         patch.object(E2BSandboxEngine, "is_gvisor_available", return_value=False):
        engine = E2BSandboxEngine(api_key="mock-key-123", allow_unsafe_local=False)

        mock_e2b = MagicMock()
        mock_e2b.Sandbox.side_effect = Exception("E2B cloud API connection timed out")
        with patch.dict("sys.modules", {"e2b_code_interpreter": mock_e2b}):
            with pytest.raises(SandboxUnavailableError) as exc_info:
                await engine.execute_code("print('Hello')", language="python")
            assert "Refusing unisolated local execution" in str(exc_info.value)


@pytest.mark.asyncio
async def test_e2b_cloud_failure_allows_local_only_with_explicit_flag():
    """Verify E2B failure falls back to local execution ONLY when allow_unsafe_local=True."""
    engine = E2BSandboxEngine(api_key="mock-key-123", allow_unsafe_local=True)

    mock_e2b = MagicMock()
    mock_e2b.Sandbox.side_effect = Exception("E2B cloud outage")
    with patch.dict("sys.modules", {"e2b_code_interpreter": mock_e2b}):
        res = await engine.execute_code("print('Dev fallback output')", language="python")
        assert res.exit_code == 0
        assert "Dev fallback output" in res.stdout
        assert res.backend_used == IsolationBackend.LOCAL_UNSAFE.value


# ==============================================================================
# 2. Explicit Backend Selection & Availability Gating
# ==============================================================================


def test_explicit_e2b_backend_without_key_refuses():
    """Verify explicit E2B backend selection raises SandboxUnavailableError if unkeyed."""
    engine = E2BSandboxEngine(api_key="", backend=IsolationBackend.E2B)
    with pytest.raises(SandboxUnavailableError) as exc_info:
        engine.resolve_backend()
    assert "no E2B_API_KEY is configured" in str(exc_info.value)


def test_explicit_gvisor_backend_without_runsc_refuses():
    """Verify explicit gVisor backend selection raises SandboxUnavailableError if runsc is missing."""
    with patch.object(E2BSandboxEngine, "is_gvisor_available", return_value=False):
        engine = E2BSandboxEngine(backend=IsolationBackend.GVISOR)
        with pytest.raises(SandboxUnavailableError) as exc_info:
            engine.resolve_backend()
        assert "gVisor backend requested but 'runsc' was not found" in str(exc_info.value)


def test_explicit_container_backend_without_daemon_refuses():
    """Verify explicit container backend selection raises SandboxUnavailableError if daemon is unreachable."""
    with patch.object(E2BSandboxEngine, "is_container_available", return_value=False):
        engine = E2BSandboxEngine(backend=IsolationBackend.CONTAINER)
        with pytest.raises(SandboxUnavailableError) as exc_info:
            engine.resolve_backend()
        assert "Container backend requested but container runtime is not running" in str(exc_info.value)


def test_explicit_local_unsafe_without_permission_refuses():
    """Verify explicit LOCAL_UNSAFE backend selection raises SandboxUnavailableError if allow_unsafe_local=False."""
    engine = E2BSandboxEngine(backend=IsolationBackend.LOCAL_UNSAFE, allow_unsafe_local=False)
    with pytest.raises(SandboxUnavailableError) as exc_info:
        engine.resolve_backend()
    assert "allow_unsafe_local is False" in str(exc_info.value)


# ==============================================================================
# 3. Gating of LOCAL_UNSAFE Execution
# ==============================================================================


@pytest.mark.asyncio
async def test_execute_local_unsafe_hard_security_gate():
    """Verify _execute_local_unsafe raises immediately if allow_unsafe_local is False."""
    engine = E2BSandboxEngine(allow_unsafe_local=False)
    with pytest.raises(SandboxUnavailableError) as exc_info:
        await engine._execute_local_unsafe("print('attack')", "python", None, 0.0)
    assert "allow_unsafe_local is False" in str(exc_info.value)


def test_allow_unsafe_local_via_environment_variable():
    """Verify allow_unsafe_local can be enabled via SANDBOX_ALLOW_UNSAFE_LOCAL env var."""
    with patch.dict(os.environ, {"SANDBOX_ALLOW_UNSAFE_LOCAL": "1"}):
        engine = E2BSandboxEngine()
        assert engine.allow_unsafe_local is True

    with patch.dict(os.environ, {"SANDBOX_ALLOW_UNSAFE_LOCAL": "0"}):
        engine = E2BSandboxEngine()
        assert engine.allow_unsafe_local is False


# ==============================================================================
# 4. Environment Variable Secret Sanitization
# ==============================================================================


def test_sanitize_environment_strips_sensitive_credentials():
    """Verify sanitize_environment removes AWS, E2B, API tokens, and secrets."""
    dirty_env = {
        "PATH": "C:\\Windows\\System32;/usr/bin",
        "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "E2B_API_KEY": "e2b_sec_1234567890",
        "OPENAI_API_KEY": "sk-proj-abcdef123456",
        "DATABASE_PASSWORD": "supersecretpassword",
        "USER_AUTH_TOKEN": "eyJhbGciOi...",
        "NORMAL_SETTING": "enabled",
    }
    with patch.dict(os.environ, dirty_env, clear=True):
        clean = sanitize_environment(extra_env={"TEST_KEY": "test_val"})

        # Sensitive variables must be stripped
        assert "AWS_ACCESS_KEY_ID" not in clean
        assert "AWS_SECRET_ACCESS_KEY" not in clean
        assert "E2B_API_KEY" not in clean
        assert "OPENAI_API_KEY" not in clean
        assert "DATABASE_PASSWORD" not in clean
        assert "USER_AUTH_TOKEN" not in clean

        # Safe variables and explicit extra_env must be preserved
        assert "PATH" in clean
        assert clean["TEST_KEY"] == "test_val"


@pytest.mark.asyncio
async def test_executed_code_cannot_read_parent_aws_secrets():
    """Verify local executed code cannot inspect parent AWS secrets."""
    dirty_env = {
        "AWS_SECRET_ACCESS_KEY": "EXPOSED_SECRET_12345",
        "E2B_API_KEY": "EXPOSED_E2B_TOKEN",
    }
    with patch.dict(os.environ, dirty_env):
        engine = E2BSandboxEngine(allow_unsafe_local=True)
        code = (
            "import os\n"
            "s1 = os.environ.get('AWS_SECRET_ACCESS_KEY')\n"
            "s2 = os.environ.get('E2B_API_KEY')\n"
            "print(f'S1={s1},S2={s2}')\n"
        )
        res = await engine.execute_code(code, language="python")
        assert res.exit_code == 0
        assert "S1=None,S2=None" in res.stdout


# ==============================================================================
# 5. Container Runtime Quota and Network Flag Construction
# ==============================================================================


@pytest.mark.asyncio
async def test_container_command_construction_with_quotas_and_no_network():
    """Verify container backend passes memory, cpu, and network=none flags."""
    with patch("shutil.which", return_value="/usr/bin/docker"), \
         patch.object(E2BSandboxEngine, "is_container_available", return_value=True):
        engine = E2BSandboxEngine(
            quota=ResourceQuota(
                max_cpu_cores=2.0,
                max_memory_mb=1024,
                timeout_seconds=15.0,
                allow_network_egress=False,
            ),
            backend=IsolationBackend.CONTAINER,
        )

        # Mock asyncio subprocess creation
        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"container output\n", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            res = await engine.execute_code("print('In container')", language="python")

            # Verify subprocess call args
            assert mock_exec.called
            call_args = mock_exec.call_args[0]
            assert "/usr/bin/docker" in call_args
            assert "run" in call_args
            assert "--memory=1024m" in call_args
            assert "--cpus=2.0" in call_args
            assert "--network" in call_args
            assert "none" in call_args

            assert res.exit_code == 0
            assert "container output" in res.stdout
            assert res.backend_used == IsolationBackend.CONTAINER.value


# ==============================================================================
# 6. E1 Adversarial Suite (Hostile Snippets Blocked or Refused)
# ==============================================================================


@pytest.mark.asyncio
async def test_e1_adversarial_network_connect_refused():
    """E1 Snippet: Hostile network connect is refused under default isolate-or-refuse."""
    hostile_code = (
        "import urllib.request\n"
        "urllib.request.urlopen('https://example.com', timeout=1.0)\n"
    )
    with patch.object(E2BSandboxEngine, "is_container_available", return_value=False), \
         patch.object(E2BSandboxEngine, "is_gvisor_available", return_value=False):
        engine = E2BSandboxEngine(api_key="", allow_unsafe_local=False)
        with pytest.raises(SandboxUnavailableError):
            await engine.execute_code(hostile_code, language="python")


@pytest.mark.asyncio
async def test_e1_adversarial_filesystem_escape_refused():
    """E1 Snippet: Hostile filesystem write outside workdir is refused under default."""
    hostile_code = (
        "import tempfile, os\n"
        "with open('/tmp/evil_payload.txt', 'w') as f:\n"
        "    f.write('pwned')\n"
    )
    with patch.object(E2BSandboxEngine, "is_container_available", return_value=False), \
         patch.object(E2BSandboxEngine, "is_gvisor_available", return_value=False):
        engine = E2BSandboxEngine(api_key="", allow_unsafe_local=False)
        with pytest.raises(SandboxUnavailableError):
            await engine.execute_code(hostile_code, language="python")


@pytest.mark.asyncio
async def test_e1_adversarial_env_secret_harvest_refused():
    """E1 Snippet: Hostile environment secret harvesting is refused under default."""
    hostile_code = (
        "import os\n"
        "secrets = {k: v for k, v in os.environ.items() if 'KEY' in k or 'SECRET' in k}\n"
        "assert len(secrets) > 0\n"
    )
    with patch.object(E2BSandboxEngine, "is_container_available", return_value=False), \
         patch.object(E2BSandboxEngine, "is_gvisor_available", return_value=False):
        engine = E2BSandboxEngine(api_key="", allow_unsafe_local=False)
        with pytest.raises(SandboxUnavailableError):
            await engine.execute_code(hostile_code, language="python")


@pytest.mark.asyncio
async def test_e1_adversarial_fork_bomb_refused():
    """E1 Snippet: Hostile fork bomb / process flooding is refused under default."""
    hostile_code = (
        "import os\n"
        "while True:\n"
        "    os.fork()\n"
    )
    with patch.object(E2BSandboxEngine, "is_container_available", return_value=False), \
         patch.object(E2BSandboxEngine, "is_gvisor_available", return_value=False):
        engine = E2BSandboxEngine(api_key="", allow_unsafe_local=False)
        with pytest.raises(SandboxUnavailableError):
            await engine.execute_code(hostile_code, language="python")


@pytest.mark.asyncio
async def test_e1_adversarial_memory_hog_refused():
    """E1 Snippet: Hostile 10GB memory allocation is refused under default."""
    hostile_code = "x = bytearray(10 * 1024 * 1024 * 1024)\n"
    with patch.object(E2BSandboxEngine, "is_container_available", return_value=False), \
         patch.object(E2BSandboxEngine, "is_gvisor_available", return_value=False):
        engine = E2BSandboxEngine(api_key="", allow_unsafe_local=False)
        with pytest.raises(SandboxUnavailableError):
            await engine.execute_code(hostile_code, language="python")
