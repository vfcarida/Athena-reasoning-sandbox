# ADR 0001: Minimal Isolated Authorized Agent-Loop Wiring (Branch A)

## Status
Accepted

## Context & Problem Statement
Prior to this architectural change (ARS-R1), the repository lacked an end-to-end execution loop connecting reasoning trajectories to tool execution. Specifically:
- `src/reasoning/schemas.py` defined schemas (`AgentPlan`, `ReasoningStep`, `ActionType`, `ToolCallPayload`, `ObservationPayload`), but no loop existed to process plans.
- `src/engine/executor.py` only registered trivial `ping` and `echo` handlers.
- `ActionType.ATHENA_QUERY` and `ActionType.SANDBOX_EXECUTE` routed to no executive handlers.
- Tool invocation boundaries were in-process without explicit authorization allowlists.

Decision ARS-R1 offered two paths:
- **Branch A**: Wire a minimal, isolated, authorized `reasoning -> plan -> executor -> sandbox/athena -> result` loop with real tools under explicit deny-by-default authorization.
- **Branch B**: Move unwired modules (`engine/`, `sandbox/`, `athena/`, `state/`, `rag/`, `telemetry/`) under `experiments/` and stop presenting them as a working framework.

## Prerequisites Check
The prerequisites for Branch A have been fully satisfied:
1. **ARS-T01**: Python 3.10+ virtual environment and test harness split into Lane 1 (fast, torch-free, no paid cloud) and Lane 2 (torch, live cloud).
2. **ARS-T03**: Athena FinOps cost-guard hardened with `sqlglot` AST partition predicate validation, WorkGroup `BytesScannedCutoffPerQuery`, and `wait_for_completion` with cancellation.
3. **ARS-T04**: Sandbox hardened with isolate-or-refuse policy (`IsolationBackend`), host secret sanitization, and elimination of silent unisolated local fallbacks.

Because the underlying execution backends (`AthenaClient` and `E2BSandboxEngine`) are now verified, safe, and hardened, selecting Branch A delivers a working, secure agent loop rather than discarding functional infrastructure.

## Decision
We select **Branch A** and implement the minimal, isolated, authorized Plan-then-Execute agent loop.

### Key Architectural Choices:
1. **P-t-E Reasoning Pipeline**:
   The workflow follows the sequence:
   `estimate complexity` → `plan (Pydantic, frozen)` → `dispatch across Parallax boundary` → `authorize & execute` → `typed result`.
2. **Cognitive-Executive Separation**:
   Reasoning steps emit immutable, frozen `AgentPlan` trajectories composed of discrete `ReasoningStep` items. No code within the reasoning or cognitive layer executes OS, subprocess, or network calls directly.
3. **Deny-by-Default Authorization Allowlist**:
   `ExecutiveEngineProcess` maintains an explicit `authorized_tools: Set[str]` allowlist. Any tool invocation request not in the allowlist is rejected immediately with `ObservationPayload(success=False, error_message=...)` without handler dispatch.
4. **Isolate-or-Refuse Inheritance**:
   When `ActionType.SANDBOX_EXECUTE` is dispatched, it invokes `E2BSandboxEngine`. In environments without verified container/microVM isolation backends, it strictly refuses execution (`SandboxUnavailableError`) unless `allow_unsafe_local=True` is explicitly passed by an authorized operator.
5. **No Silent Success (Strict Error Propagation)**:
   Any step failure (validation error, authorization denial, sandbox failure, timeout, cost limit exceeded) immediately halts subsequent plan execution. Failures propagate as structured errors inside `AgentLoopResult`.
6. **Lane-1 Compatibility**:
   The core agent loop and planner are implemented using standard library and Pydantic v2 without importing `torch` or heavy ML dependencies, preserving fast CI testability.

## Consequences

### Positive
- Delivers a verifiable, end-to-end `AgentLoop` in `src/engine/agent_loop.py` executing real tools (`AthenaClient` and `E2BSandboxEngine`).
- Guarantees unauthorized tools cannot execute across the Parallax boundary.
- Prevents unisolated code escape or silent local execution under default configurations.
- All integration tests run cleanly offline in Lane 1 with zero paid cloud API calls.

### Negative / Trade-offs
- Dynamic LLM-based generation of reasoning steps at runtime (e.g. using `SWIReasoningEngine`) remains separated in Lane 2 due to PyTorch model weights and memory requirements.
- Multi-turn dynamic replanning based on intermediate observations is deferred to future milestones; the loop strictly executes deterministic, validated P-t-E plans.
