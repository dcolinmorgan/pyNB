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


## 2026-04-11 - US-002
- What was implemented: LSCO, CLR, Elastic Net, and Ridge methods ported to sparselink with unified `fit(X) -> InferenceResult` interface. All registered via `@registry.register` decorator.
- Files changed:
  - `sparselink/src/sparselink/methods/lsco.py` - LSCOMethod (least squares + hard threshold)
  - `sparselink/src/sparselink/methods/clr.py` - CLRMethod (mutual information + CLR z-score transform)
  - `sparselink/src/sparselink/methods/elastic_net.py` - ElasticNetMethod and RidgeMethod
  - `sparselink/src/sparselink/methods/__init__.py` - Updated imports for all new methods
  - `sparselink/src/sparselink/__init__.py` - Added `import sparselink.methods` to trigger registration
  - `sparselink/tests/test_methods.py` - 16 tests covering all methods
- **Learnings:**
  - Registration trigger: The package `__init__.py` must `import sparselink.methods` to ensure decorators run and methods are discoverable via `get_method()`/`list_methods()`
  - PYTHONPATH for tests: When running pytest from the sparselink subpackage, use `PYTHONPATH=src` since the parent directory name (`sparselink/`) conflicts with the package namespace
  - CLR output is symmetric and non-negative by construction (sqrt of sum of squared z-scores)
---


## 2026-04-11 - US-003
- What was implemented: PCMCI, Granger Causality, and Transfer Entropy methods in sparselink (Partial Correlation already existed from US-001). All use unified `fit(X) -> InferenceResult` interface and `@registry.register` decorator.
- Files changed:
  - `sparselink/src/sparselink/methods/pcmci.py` - PCMCIMethod (lagged partial correlations)
  - `sparselink/src/sparselink/methods/granger.py` - GrangerCausality (pairwise F-test on VAR)
  - `sparselink/src/sparselink/methods/transfer_entropy.py` - TransferEntropy (binning-based conditional entropy)
  - `sparselink/src/sparselink/methods/__init__.py` - Added imports for new methods
  - `sparselink/tests/test_causal_methods.py` - 18 tests covering all 4 methods
- **Learnings:**
  - Time-series methods need synthetic data with known causal structure for meaningful tests (e.g., x0 -> x1 via lagged coupling)
  - Transfer entropy via binning is simple but effective; `np.digitize` with clipping handles edge cases
  - Granger causality F-stat uses `np.linalg.lstsq` for OLS — no external dependency needed
---
