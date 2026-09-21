## Description

Please provide a concise description of the changes introduced by this pull request, along with the motivation and architectural context.

Fixes #(issue) / Related to #(issue)

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring / Performance improvement

## Architectural Alignment

- [ ] Adheres to Cognitive-Executive Separation (Parallax Principle)
- [ ] Preserves FinOps AST query governance (no unpartitioned or unbounded SQL)
- [ ] Preserves sandbox isolation guarantees (refuses unisolated local fallback by default)
- [ ] Compatible with offline Lane-1 testing (zero cloud/GPU dependencies)

## Quality Gate Checklist

Before requesting a review, verify all steps below pass locally:

- [ ] Code conforms to PEP 8 standards: `ruff check src evals tests`
- [ ] Static type checking passes: `mypy src/engine src/athena src/state src/sandbox`
- [ ] Security scan passes: `bandit -r src/ -x tests/ -s B105,B106,B615 -ll`
- [ ] Full offline unit & integration test suite passes: `pytest -m "not e2b and not network" -v`
- [ ] Code coverage meets or exceeds minimum threshold
- [ ] All new functions, classes, and modules include Google-style docstrings with algorithmic complexity notes
