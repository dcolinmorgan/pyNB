# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

- **Method Registry Pattern**: Use `@registry.register` decorator on `InferenceMethod` subclasses. Set `name` class attribute for lookup key. Import the methods module to trigger registration.
- **InputData handling**: All methods accept `Union[np.ndarray, pd.DataFrame]` via `self._to_array()` helper in base class.

---

## 2026-04-11 - US-001
- What was implemented: Standalone sparselink package with unified `fit(X) -> InferenceResult` interface, common types (`AdjacencyMatrix`, `EdgeList`, `InferenceResult`), method registry pattern, and two reference implementations (LassoMethod, PartialCorrelation).
- Files changed:
  - `sparselink/pyproject.toml` - Package config (hatchling, deps, mypy strict, ruff)
  - `sparselink/src/sparselink/__init__.py` - Public API exports
  - `sparselink/src/sparselink/types.py` - Common types: AdjacencyMatrix, EdgeList, InferenceResult, InputData
  - `sparselink/src/sparselink/base.py` - Abstract InferenceMethod base class with fit() interface
  - `sparselink/src/sparselink/registry.py` - Registry class with register/get/list + module-level helpers
  - `sparselink/src/sparselink/methods/__init__.py` - Methods subpackage
  - `sparselink/src/sparselink/methods/lasso.py` - LassoMethod implementation
  - `sparselink/src/sparselink/methods/correlation.py` - PartialCorrelation implementation
- **Learnings:**
  - numpy typing: use `npt.NDArray[np.floating]` not `np.ndarray` for typed aliases
  - mypy strict: avoid `# type: ignore` comments that become stale; use `np.asarray()` for safe conversions
  - Registry decorator pattern: class must have `name` attribute set before registration; decorator returns the class unchanged
---

