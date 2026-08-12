# Contributing to Athena Reasoning Sandbox

Thank you for your interest in contributing to **Athena Reasoning Sandbox**. We welcome contributions from AI researchers, data engineers, and cloud architects.

## 📜 Code of Conduct & Standards

To maintain high technical quality and architectural rigor across the repository, all contributions must adhere to the following standards:

1. **Strict Type Safety**: All Python code must be compatible with Python 3.11+ and pass strict `mypy` type checking without errors or `type: ignore` bypasses (unless strictly necessary and documented).
2. **Coding Style**: Code must strictly conform to PEP 8 standards enforced via `ruff check` and `ruff format`.
3. **Comprehensive Docstrings**: All public modules, classes, algorithms, and functions must include Google-style docstrings detailing purpose, parameters, return types, exceptions, and algorithmic complexity.
4. **Test-Driven Development (TDD)**: Every bug fix or new feature must include corresponding unit tests in `tests/`. Line and branch coverage must remain above 90%.

---

## 🔁 Pull Request Guidelines & Workflow

### 1. Branch Naming Convention
- Feature branches: `feat/short-description`
- Bug fixes: `fix/short-description`
- Documentation/Evals: `docs/short-description` or `eval/short-description`

### 2. Artifact Feedback Loop & AI Code Verification
If you use generative AI models or coding assistants during development:
- You must review all generated code for logic flaws, security vulnerabilities, or anti-patterns ("vibe coding").
- Ensure all generated schemas (Pydantic/Zod) enforce explicit validation and defensive bounds.
- Verify that SQL generation logic passes `AthenaQueryGuard` partition and projection checks.

### 3. Human-in-the-Loop Review & CI Verification
Before submitting a Pull Request:
1. Run local linting and formatting:
   ```bash
   ruff check src evals tests
   mypy src evals --strict
   ```
2. Run full unit test suite:
   ```bash
   pytest tests/ -v --cov=src --cov-fail-under=90
   ```
3. Run DeepEval agent trajectory suite:
   ```bash
   deepeval test run evals/test_trajectory_eval.py
   ```
4. PRs require passing CI build checks and approval from at least one core maintainer.

---

## 🏛️ Repository Owner & Maintainer

Maintained by **Vinicius Caridá** ([@vfcarida](https://github.com/vfcarida)).
