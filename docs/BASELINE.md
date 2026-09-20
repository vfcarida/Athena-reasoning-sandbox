# Athena Reasoning Sandbox — Baseline & Preflight Harness Record

- **Task ID**: ARS-T01
- **Repository**: [Athena-reasoning-sandbox](https://github.com/vfcarida/Athena-reasoning-sandbox)
- **Baseline SHA**: `e1ec990df319fec3cc18533788917257f58aa82c` (branch `main`)
- **Execution Date**: 2026-09-18
- **Operating System**: Windows 11 (NT 10.0.26200 amd64)
- **Host Python**: Python 3.12.10 (`C:\Users\vinicius\AppData\Local\Programs\Python\Python312\python.exe`)
- **Virtual Environment**: `.venv` (`C:\Users\vinicius\Documents\GeminiCodes\Athena-reasoning-sandbox\.venv`)
- **Companion**: `SHARED-CONVENTIONS.md`
- **Execution Boundary Compliance**: Local `.venv` only, zero AWS/E2B/paid network calls, zero modifications to `src/`, `training_state.pt` unpickled and untouched, no pushing.

---

## 1. Environment & Preflight

### Git Status Preflight
- Baseline commit SHA: `e1ec990df319fec3cc18533788917257f58aa82c`
- Initial working tree clean on branch `main`.

### Toolchain Inventory
- **Python**: 3.12.10
- **pytest**: 9.1.1
- **pytest-cov**: 7.1.0
- **pytest-asyncio**: 1.4.0
- **ruff**: 0.16.8
- **bandit**: 1.9.4
- **mypy**: 2.3.1 (compiled: yes)

### Environment Packages (`pip list`)
```text
Package                            Version
---------------------------------- ---------------
aiohappyeyeballs                   2.7.1
aiohttp                            3.14.3
aiosignal                          1.4.0
annotated-doc                      0.0.5
annotated-types                    0.8.0
anyio                              4.15.1
ast_serialize                      0.11.2
attrs                              26.1.0
backoff                            2.2.1
bandit                             1.9.4
boto3                              1.43.98
botocore                           1.43.98
certifi                            2026.7.22
charset-normalizer                 3.5.1
click                              8.3.3
colorama                           0.4.6
coverage                           7.16.1
deepeval                           4.2.3
distro                             1.9.0
execnet                            2.1.2
frozenlist                         1.8.0
grpcio                             1.84.0
h11                                0.16.0
httpcore2                          2.13.0
httpx2                             2.13.0
idna                               3.20
iniconfig                          2.3.0
Jinja2                             3.1.6
jiter                              0.17.0
jmespath                           1.1.0
librt                              0.15.0
markdown-it-py                     4.2.0
MarkupSafe                         3.0.3
mdurl                              0.1.2
multidict                          6.9.0
mypy                               2.3.1
mypy_extensions                    1.1.0
nest-asyncio                       1.6.0
numpy                              2.5.3
openai                             3.16.1
opentelemetry-api                  1.44.0
opentelemetry-sdk                  1.44.0
opentelemetry-semantic-conventions 0.65b0
packaging                          26.3
pathspec                           1.1.1
pip                                25.0.1
pluggy                             1.6.0
portalocker                        4.3.2
posthog                            7.58.0
prompt_toolkit                     3.0.53
propcache                          0.5.4
pydantic                           2.13.5
pydantic_core                      2.46.5
pydantic-settings                  2.15.0
pyfiglet                           1.0.4
Pygments                           2.21.0
pytest                             9.1.1
pytest-asyncio                     1.4.0
pytest-cov                         7.1.0
pytest-repeat                      0.9.4
pytest-rerunfailures               16.7
pytest-xdist                       3.8.0
python-dateutil                    2.9.0.post0
python-dotenv                      1.2.3
PyYAML                             6.0.3
questionary                        2.1.1
requests                           2.34.2
rich                               14.3.4
ruff                               0.16.8
s3transfer                         0.19.2
setuptools                         84.0.0
shellingham                        1.5.4
six                                1.17.0
sniffio                            1.3.1
stevedore                          5.9.1
tabulate                           0.9.0
tenacity                           9.1.4
tqdm                               4.70.1
truststore                         0.10.4
typer                              0.27.2
types-PyYAML                       6.0.12.20260906
typing_extensions                  4.16.0
typing-inspection                  0.4.4
urllib3                            2.8.0
wcwidth                            0.8.4
wheel                              0.48.0
yarl                               1.25.1
```

---

## 2. Dependency Budget & Torch Blocker

- **Budget & Boundary Rule**: Per execution boundary 8 ("Torch download requires approval else documented blocker"), full `torch>=2.0.0` wheel download (~2+ GB with CUDA/CPU wheels) was not approved.
- **Documented Blocker**:
  `pip install -r requirements.txt` blocked by disallowed ~2+ GB PyTorch stack download (`torch>=2.0.0`, `transformers>=4.35.0`, `accelerate>=0.25.0`, `peft>=0.7.0`, `trl>=0.7.0`).
- **Installed Lane-1 Subset**:
  Installed the complete offline testing, linting, typing, and security dependencies required for Lane 1: `pytest`, `pytest-cov`, `pytest-asyncio`, `ruff`, `bandit`, `mypy`, `pydantic`, `pydantic-settings`, `types-PyYAML`, `pyyaml`, `rich`, `tqdm`, `deepeval`, `boto3`, `numpy`.

---

## 3. Test Lanes Partitioning

Tests are partitioned into two lanes:
- **Lane 1 (Lightweight / Torch-free / Offline)**: 28 unit tests executing without PyTorch, AWS network calls, or paid API requirements.
- **Lane 2 (Heavy / Torch / Cloud / GPU)**: 13 tests tagged with `pytest.mark.heavy` requiring PyTorch or ML runtime stacks.

### Lane 1 Tests (28 passed)
1. `tests/test_athena_cost_guard.py` (8 tests):
   - `test_partition_filter_guard_success`
   - `test_partition_filter_guard_missing_where`
   - `test_partition_filter_guard_missing_key`
   - `test_select_star_guard_failure`
   - `test_missing_limit_guard_failure`
   - `test_ctas_pipeline_statement_generation`
   - `test_athena_client_dry_run_execution`
   - `test_app_config_validation`
2. `tests/test_phase4.py` (3 tests):
   - `test_sandbox_local_fallback_success`
   - `test_sandbox_timeout_quota`
   - `test_checkpoint_manager_save_and_restore`
3. `tests/test_phase3.py` (6 tests):
   - `test_plan_quality_metric`
   - `test_plan_adherence_metric`
   - `test_tool_correctness_metric`
   - `test_task_completion_metric`
   - `test_hybrid_search_rrf`
   - `test_telemetry_tracer_spans`
4. `tests/test_rag_evaluator.py` (7 tests):
   - `test_evaluate_rag_triad_success`
   - `test_evaluate_rag_triad_faithfulness_failure`
   - `test_evaluate_rag_triad_relevancy_failure`
   - `test_evaluate_rag_triad_precision_failure`
   - `test_evaluate_rag_triad_validation_errors`
   - `test_sagemaker_llm_payload_and_invocation`
   - `test_sagemaker_llm_async_generate`
5. `tests/test_phase1.py` (4 tests in Lane 1):
   - `test_pydantic_schema_validation`
   - `test_executive_engine_process_ping`
   - `test_executive_engine_unregistered_tool`
   - `test_parallax_tool_dispatcher`

### Lane 2 Tests (13 marked `heavy` / deselected in Lane 1)
1. `tests/test_bridge.py` (3 tests):
   - `test_bridge_initialization`
   - `test_agent_bench_metrics_list`
   - `test_export_for_evaluation_yaml`
2. `tests/test_finetuning.py` (3 tests):
   - `test_sft_format_alpaca`
   - `test_sft_format_chatml`
   - `test_lora_config`
3. `tests/test_merging_and_reasoning.py` (3 tests):
   - `test_slerp`
   - `test_dare_drop_and_rescale`
   - `test_swi_reasoning_metrics`
4. `tests/test_pretraining.py` (3 tests):
   - `test_transformer_from_scratch_build`
   - `test_tokenizer_and_dataset`
   - `test_continued_pretrainer_config`
5. `tests/test_phase1.py` (1 test):
   - `test_swi_reasoning_async_simulation`

---

## 4. Quality Gate Executions

### 4.1. Ruff Linter
- **Command**: `.\.venv\Scripts\python.exe -m ruff check .`
- **Exit Code**: `1`
- **Summary**: Found 287 errors (232 automatically fixable with `--fix`, 15 hidden fixes with `--unsafe-fixes`).
- **Primary Diagnostics**:
  - `F401`: Unused imports across test and source files (`pathlib.Path`, `CheckpointState`, `ContinuedPretrainer`, `unittest.mock.patch`).
  - `I001`: Unsorted/unformatted import blocks.
  - `UP045`: Deprecated typing syntax (use `X | None` instead of `Optional[X]`).
  - `F841`: Local variables assigned but never used (e.g. `schema_name` in `test_rag_evaluator.py`).
  - `SIM114`: Uncombined `if` branches using identical bodies.
  - `BLE001`: Blind `except Exception:` blocks.

---

### 4.2. Bandit Security Scan
- **Command**: `.\.venv\Scripts\python.exe -m bandit -r src`
- **Exit Code**: `1`
- **Metrics**:
  - Total Lines of Code: 4,377
  - Issues by Severity: Low: 6, Medium: 11, High: 0 (Total: 17)
  - Issues by Confidence: Medium: 4, High: 13
- **Detailed Findings**:
  1. **Subprocess Invocations & Shell Injection Sinks**:
     - `B404` (Low / High) `src\main.py:25`: Import of `subprocess` module.
     - `B603` (Low / High) `src\main.py:281`: `subprocess.run(cmd, ...)` call without shell check.
     - **Manual Audit Finding (`src/sandbox/e2b_sandbox.py:147-150`)**:
       `E2BSandboxEngine._execute_local_fallback` executes arbitrary commands via `["bash", "-c", code]` or `["python", "-c", code]` passed directly to `asyncio.create_subprocess_exec(*cmd)`. This represents an arbitrary shell-exec sink if untrusted code or strings reach local fallback execution.
  2. **Unsafe Model Deserialization**:
     - `B614` (Medium / High) `src\pretraining\from_scratch.py:558`:
       `checkpoint = torch.load(state_file, map_location=self.device, weights_only=False)`
       Explicit use of `weights_only=False` enables arbitrary code execution via Python pickle payloads (directly related to `training_state.pt`).
  3. **Unpinned Remote Model Downloads**:
     - `B615` (Medium / High): 10 occurrences in `src/finetuning/lora_trainer.py` (lines 122, 132, 181, 340, 344), `src/finetuning/sft_trainer.py` (lines 217, 229), and `src/pretraining/continued_pretraining.py` (lines 183, 190, 200).
       Hugging Face Hub `from_pretrained` downloads without revision pinning (supply chain tampering risk).
  4. **Hardcoded Credential Token False Positives**:
     - `B106` (Low / Medium) `src\pretraining\from_scratch.py:244, 262`: Vocabulary tokens `<unk>`, `<s>` flagged as potential passwords.
     - `B105` (Low / Medium) `src\reasoning\swi_reasoning.py:83, 84`: Special tokens `<think>`, `</think>` flagged as potential passwords.

---

### 4.3. MyPy Strict Type Checker
- **Command**: `.\.venv\Scripts\python.exe -m mypy --strict src evals`
- **Exit Code**: `1`
- **Summary**: Found 39 errors in 14 files.
- **Primary Diagnostics**:
  - `import-untyped`: Missing type stubs or `py.typed` marker for `boto3` (`src/athena/athena_client.py:104`, `src/utils/rag_evaluator.py:15`).
  - `import-not-found`: Missing stubs/packages for optional or heavy stacks in torch-free environment (`torch`, `torch.nn`, `torch.cuda.amp`, `torch.optim`, `transformers`, `peft`, `trl`, `datasets`, `tokenizers`, `e2b_code_interpreter`, `agent_bench`).
  - Duplicate module path resolution: `evals\metrics\trajectory_metrics.py` found twice under `"trajectory_metrics"` and `"evals.metrics.trajectory_metrics"` due to lack of explicit package base configuration.

---

### 4.4. Pytest Execution & Coverage (Lane 1)
- **Command**: `.\.venv\Scripts\python.exe -m pytest -m "not heavy" -q --cov=src --cov-report=term-missing`
- **Exit Code**: `0`
- **Execution Summary**: **28 passed, 13 deselected, 2 warnings in 3.33s**

#### Coverage Report Table
```text
Name                                       Stmts   Miss  Cover   Missing
------------------------------------------------------------------------
src\__init__.py                                2      0   100%
src\athena\__init__.py                         3      0   100%
src\athena\athena_client.py                   45     12    73%   103-113, 151-159, 191-200
src\athena\query_guard.py                     46      0   100%
src\bridge\__init__.py                         2      0   100%
src\bridge\agent_bench_bridge.py             110     83    25%   72-75, 86-106, 111, 146-206, 230-247, 269-303, 323-340, 355
src\engine\__init__.py                         3      0   100%
src\engine\executor.py                        42      8    81%   106-119
src\engine\grpc_boundary.py                   30     13    57%   70, 83-105
src\finetuning\__init__.py                     3      0   100%
src\finetuning\lora_trainer.py               106     72    32%   94-99, 117-151, 165-193, 210-244, 263-272, 298-319, 337-351, 362-381
src\finetuning\sft_trainer.py                144    103    28%   123-128, 146-175, 191-200, 215-240, 263-273, 283-320, 334-401, 420-428
src\main.py                                  184    184     0%   13-361
src\merging\__init__.py                        3      0   100%
src\merging\merge_operators.py                88     77    12%   77-125, 160-238, 278-321
src\merging\merge_pipeline.py                103     86    17%   49-54, 72-91, 120-134, 144-178, 188-234, 250-254
src\pretraining\__init__.py                    3      0   100%
src\pretraining\continued_pretraining.py     190    142    25%   105-120, 126, 129-130, 157-164, 178-225, 241-259, 271-277, 299-314, 330, 356-479, 489-504
src\pretraining\from_scratch.py              203    152    25%   109-127, 133, 136-137, 165-173, 188-220, 240-275, 304-449, 469-486, 513-535, 555-571, 578-590
src\rag\__init__.py                            2      0   100%
src\rag\hybrid_search.py                      85      6    93%   139, 171, 214-221
src\reasoning\__init__.py                      2      0   100%
src\reasoning\schemas.py                      37      0   100%
src\reasoning\swi_reasoning.py               168    125    26%   108-121, 143-156, 172-243, 267-268, 297-306, 323-332, 353-359, 363-374, 382-432, 445-446
src\sandbox\__init__.py                        2      0   100%
src\sandbox\e2b_sandbox.py                    60     18    70%   102, 114-133, 147, 176-178
src\state\__init__.py                          2      0   100%
src\state\checkpoint_manager.py               48     11    77%   48, 121-132
src\telemetry\__init__.py                      2      0   100%
src\telemetry\tracer.py                       49     20    59%   28-30, 54, 70, 77-79, 82-83, 86-87, 90, 106-112
src\utils\__init__.py                          2      0   100%
src\utils\config.py                           17      0   100%
src\utils\metrics.py                          53     45    15%   53-79, 122-156, 197-210
src\utils\rag_evaluator.py                   101     28    72%   64, 104, 107-112, 117-132, 135-137, 233-237
------------------------------------------------------------------------
TOTAL                                       1940   1185    39%
```

---

## 5. Discrepancy Analysis (ARS-F9/C8)

1. **CI Quality Gate Reality Gap**:
   `.github/workflows/ci.yml` claims a strict coverage threshold:
   `pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=90`
   In reality, actual baseline coverage under Lane 1 is **39%**. Even when including Lane-2 tests, extensive parts of `main.py` (0%), `sft_trainer.py` (28%), `from_scratch.py` (25%), `continued_pretraining.py` (25%), and `merge_operators.py` (12%) are uncovered. The CI gate threshold of 90% is unachievable without substantial test harness expansion.
2. **Heavy Stack Collection Dependency**:
   As discovered in dossier §12, eager imports of `torch` in `src/utils/metrics.py` and `src/reasoning/swi_reasoning.py` previously caused collection crashes for any test importing `src.utils.config` or `src.reasoning.schemas`. The new `tests/conftest.py` meta-path hook isolates Lane 1 so lightweight tests collect and run cleanly without PyTorch installed.
3. **Behavioral Preservation**:
   All files in `src/` remain completely unchanged. Checkpoint state file `checkpoints/pretrain/checkpoint-final/training_state.pt` was not unpickled or loaded, preserving safety against `B614` unpickling vulnerabilities.

---

## 6. Rollback & Handoff

- **Rollback Procedure**:
  Remove `.venv/`, delete `docs/BASELINE.md`, delete `tests/conftest.py`, and revert test marker additions (`git checkout -- pyproject.toml tests/`).
- **Handoff Status**:
  ARS-T01 is complete. Enabler for downstream tasks T02–T08.
