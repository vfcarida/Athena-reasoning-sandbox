# Contributing to Athena Reasoning Sandbox

Thank you for your interest in contributing to **Athena Reasoning Sandbox**! We welcome contributions from AI researchers, data engineers, cloud architects, and open-source practitioners.

## 📜 Development Standards & Code Quality

To maintain high technical quality, reproducibility, and architectural safety, all contributions must adhere to the following standards:

1. **Strict Type Safety**: All Python code must be compatible with Python 3.10+ and pass `mypy src/engine src/athena src/state src/sandbox` with zero type errors.
2. **Linting & Formatting**: Code must conform to PEP 8 standards enforced via `ruff check src evals tests`.
3. **Security Auditing**: Code must pass automated security audits with zero high-severity findings: `bandit -r src/ -x tests/ -s B105,B106,B615 -ll`.
4. **Google-Style Docstrings**: All public classes, functions, and modules must include complete Google-style docstrings with parameter descriptions, return types, exceptions, and algorithmic complexity notes.
5. **Architectural Separation**: Code must respect the **Parallax Principle** (Cognitive-Executive Separation). Reasoning loops must never directly execute system calls; all actions must route through the typed dispatcher and authorized executor.
6. **FinOps Governance**: All analytical SQL queries must undergo AST validation via `AthenaQueryGuard` to enforce partition key bounding, projection limiting, and row limits.

---

## 🛠️ Local Development Setup

### 1. Clone the Repository & Initialize Virtualenv

```bash
git clone https://github.com/vfcarida/Athena-reasoning-sandbox.git
cd Athena-reasoning-sandbox

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Activate environment (Linux / macOS)
source .venv/bin/activate

# Upgrade pip and install development dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# Install git pre-commit hooks
pre-commit install
```

---

## 🔬 Testing & Quality Verification

Athena uses a dual-lane testing architecture to allow fast, deterministic local development without requiring cloud credentials or GPUs:

### Lane 1: Fast Offline Testing (No Cloud, No GPU Required)
Run the full local suite (100+ unit, integration, and FinOps tests) using in-memory DuckDB:

```bash
pytest -m "not e2b and not network" -v --cov=src --cov-report=term-missing
```

### Static Analysis & Linting

```bash
# Run Ruff linting
ruff check src evals tests

# Run MyPy type checking
mypy src/engine src/athena src/state src/sandbox src/rag src/telemetry

# Run Bandit security scanning
bandit -r src/ -x tests/ -s B105,B106,B615 -ll

# Run pre-commit hooks manually on all files
pre-commit run --all-files
```

---

## 🔁 Pull Request Lifecycle

1. **Branch Naming**:
   - Features: `feat/<feature-name>`
   - Bug fixes: `fix/<bug-description>`
   - Documentation: `docs/<topic>`
   - Performance / Refactor: `refactor/<topic>`
2. **Commit Hygiene**: Write clear, descriptive commit messages in the imperative mood (`feat: add async duckdb query client`).
3. **Pull Request Checklist**: Fill out the PR template in `.github/pull_request_template.md` ensuring all tests and linters pass before requesting review.

---

## 📄 License & Code of Conduct

By contributing to Athena Reasoning Sandbox, you agree that your contributions will be licensed under the project's [MIT License](LICENSE) and that you will abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
