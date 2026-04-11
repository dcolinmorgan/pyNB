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


## 2026-04-11 - US-004
- What was implemented: Graphical LASSO, GLASSO+StARS, and Neighborhood Selection (Meinshausen-Bühlmann) methods in sparselink with unified `fit(X) -> InferenceResult` interface and `@registry.register` decorator.
- Files changed:
  - `sparselink/src/sparselink/methods/glasso.py` - GraphicalLassoMethod and GLASSOStARS implementations
  - `sparselink/src/sparselink/methods/neighborhood.py` - NeighborhoodSelection implementation
  - `sparselink/src/sparselink/methods/__init__.py` - Added imports for new methods
  - `sparselink/tests/test_graphical_methods.py` - 15 tests covering all 3 methods
- **Learnings:**
  - sklearn's `GraphicalLasso` returns precision matrix directly; zero diagonal and take abs for adjacency
  - StARS instability metric: `2 * freq * (1 - freq)` where freq is edge selection frequency across subsamples
  - Neighborhood Selection produces binary adjacency; symmetrize with "and" (intersection) or "or" (union) rule
---


## 2026-04-11 - US-005
- What was implemented: GENIE3-style Random Forest importance and TIGRESS-style stability selection + LARS methods in sparselink with unified `fit(X) -> InferenceResult` interface and `@registry.register` decorator.
- Files changed:
  - `sparselink/src/sparselink/methods/genie3.py` - GENIE3Method (per-gene RF importance)
  - `sparselink/src/sparselink/methods/tigress.py` - TIGRESSMethod (stability selection + LARS)
  - `sparselink/src/sparselink/methods/__init__.py` - Added imports for new methods
  - `sparselink/tests/test_tree_methods.py` - 14 tests covering both methods
- **Learnings:**
  - GENIE3 output is non-negative by construction (RF feature_importances_ are always >= 0)
  - TIGRESS stability scores are naturally bounded [0, 1] — proportion of bootstraps selecting each feature
  - sparselink convention: data is (samples x features), unlike the original pyGS code which uses (features x samples)
---


## 2026-04-11 - US-008
- What was implemented: `sparselink.bench` subpackage with synthetic data generation, evaluation metrics, NestBoot bootstrap aggregation + FDR control, pipeline runner, and CLI entry point.
- Files changed:
  - `sparselink/src/sparselink/bench/__init__.py` - Public API exports
  - `sparselink/src/sparselink/bench/synthetic.py` - generate_network (random/scalefree) + generate_expression (linear model with SNR)
  - `sparselink/src/sparselink/bench/metrics.py` - evaluate() returning AUROC, AUPR, precision, recall, FDR
  - `sparselink/src/sparselink/bench/nestboot.py` - NestBoot class with bootstrap aggregation + FDR threshold via null distribution
  - `sparselink/src/sparselink/bench/runner.py` - run_benchmark() running N methods on M synthetic datasets
  - `sparselink/src/sparselink/bench/cli.py` - CLI entry point (`sparselink-bench` command)
  - `sparselink/pyproject.toml` - Added `[project.scripts]` for CLI entry point
- **Learnings:**
  - Expression generation needs spectral radius stabilization (scale A so rho < 1) before inverting (I - A)
  - NestBoot FDR control: sweep threshold from 1.0 down, compare real vs null edge counts at each level
  - sklearn lacks py.typed marker so mypy --strict always flags it; acceptable to ignore
---


## 2026-04-11 - US-009
- What was implemented: `pyGS.bio` subpackage with four modules covering all acceptance criteria: preprocessing (expression matrices, TF/target gene lists, regulon formatting), wrappers (scenicplus, pyscenic), visualization (network plots with gene annotations), and evaluation (gold standard comparison with AUROC/AUPR/F1/MCC).
- Files changed:
  - `src/bio/__init__.py` - Subpackage init with public API exports
  - `src/bio/preprocessing.py` - load_expression_matrix, filter_tf_targets, format_regulons
  - `src/bio/wrappers.py` - scenicplus_infer, pyscenic_infer (wrapping existing scenicplus.py)
  - `src/bio/visualization.py` - plot_grn (networkx-based with TF highlighting), plot_evaluation_summary
  - `src/bio/evaluation.py` - compare_to_gold_standard, compare_multiple (array-based API over CompareModels)
- **Learnings:**
  - The bio subpackage wraps existing src/ modules (methods/scenicplus.py, analyze/CompareModels.py) with cleaner array-based APIs
  - Keeping wrappers thin avoids code duplication while providing the domain-specific interface
  - ruff catches unused imports immediately — keep visualization modules lean
---


## 2026-04-11 - US-010
- What was implemented: Replaced all inline inference methods (Lasso, LSCO, CLR, GENIE3, TIGRESS) with thin wrappers that delegate to sparselink. Added `sparselink>=1.0.0` dependency to pyproject.toml. Bumped sparselink to v1.0.0. Preserved all existing pyGS API signatures and return types (3D thresholded arrays). Fixed SCENICPLUS import guard in methods/__init__.py.
- Files changed:
  - `src/methods/lasso.py` - Replaced inline implementation with sparselink wrapper
  - `src/methods/lsco.py` - Replaced inline implementation with sparselink wrapper
  - `src/methods/clr.py` - Replaced inline implementation with sparselink wrapper (kept helper functions for backward compat)
  - `src/methods/genie3.py` - Replaced inline implementation with sparselink wrapper
  - `src/methods/tigress.py` - Replaced inline implementation with sparselink wrapper (kept tigress_single_gene for backward compat)
  - `src/methods/__init__.py` - Fixed SCENICPLUS None guard in method_map
  - `pyproject.toml` - Added `sparselink>=1.0.0` to dependencies
  - `sparselink/pyproject.toml` - Bumped version to 1.0.0
  - `sparselink/src/sparselink/__init__.py` - Bumped __version__ to 1.0.0
- **Learnings:**
  - pyGS methods use (genes × samples) convention; sparselink uses (samples × features) — transpose Y.T when calling sparselink
  - pyGS returns 3D arrays (n_genes × n_genes × n_thresholds); sparselink returns single adjacency matrix — thresholding logic stays in wrapper
  - Tests import internal helpers (mutual_information_matrix, clr_transform, tigress_single_gene) — must keep these exported for backward compat
  - Small-sample guards (n_samples < 3 → zeros) must be preserved in wrappers since sparselink doesn't enforce this
---


## 2026-04-11 - US-014
- What was implemented: Set up proper PEP 621 packaging with pyproject.toml for both sparselink and pyGS. Both use hatchling backend, src/ layout, and have all 5 optional dependency groups ([dev], [test], [docs], [causal], [deep]). pyGS depends on sparselink>=1.0.0. sparselink installs standalone with no biology deps.
- Files changed:
  - `sparselink/pyproject.toml` - Added [test], [docs], [deep] optional dependency groups
  - `pyproject.toml` - Rewrote with hatchling backend, src/ layout via [tool.hatch.build.targets.wheel], removed biology-heavy deps from core (scanpy, scenicplus, snakemake), added all 5 optional groups
- **Learnings:**
  - hatchling `packages = ["src"]` maps the src/ directory as the package root for wheel builds
  - Keeping biology deps out of core pyGS dependencies means `pip install pyGS` stays lightweight; users opt-in via extras
  - PEP 621 `[project.optional-dependencies]` replaces the older `[dependency-groups]` pattern used by uv
---


## 2026-04-11 - US-006
- What was implemented: PC algorithm, FCI, NOTEARS, and DAG-GNN causal discovery methods in sparselink with unified `fit(X) -> InferenceResult` interface and `@registry.register` decorator. PC and FCI require `[causal]` optional dep (causal-learn). DAG-GNN requires `[deep]` optional dep (torch). NOTEARS uses only scipy (no extra deps).
- Files changed:
  - `sparselink/src/sparselink/methods/pc.py` - PCMethod (constraint-based, wraps causal-learn PC)
  - `sparselink/src/sparselink/methods/fci.py` - FCIMethod (constraint-based with latent confounders, wraps causal-learn FCI)
  - `sparselink/src/sparselink/methods/notears.py` - NOTEARSMethod (continuous optimization with acyclicity constraint via matrix exponential)
  - `sparselink/src/sparselink/methods/dag_gnn.py` - DAGGNNMethod (GNN-based with augmented Lagrangian DAG constraint, requires torch)
  - `sparselink/src/sparselink/methods/__init__.py` - Added imports for all 4 new methods
  - `sparselink/tests/test_us006_causal_discovery.py` - 18 tests (14 pass, 4 skip when causal-learn absent)
- **Learnings:**
  - Optional deps pattern: import inside `fit()` with clear ImportError message pointing to the correct extras group
  - NOTEARS overflow: matrix exponential of W◦W can overflow for large W values during early iterations; the algorithm still converges because thresholding zeros out unstable entries
  - causal-learn's PC/FCI return graph objects with `.graph` attribute (numpy array with edge types encoded as integers); take abs and symmetrize for undirected adjacency
  - DAG-GNN augmented Lagrangian: update rho every N epochs, not every step, for stability
---
