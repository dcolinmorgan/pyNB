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


## 2026-04-11 - US-007
- What was implemented: Bayesian structure learning with BDeu scoring (for discrete/discretized data) and BGe scoring (for continuous Gaussian data). Both use greedy hill-climbing DAG search with acyclicity constraint. Registered as "bdeu" and "bge" in the method registry.
- Files changed:
  - `sparselink/src/sparselink/methods/bayesian.py` - BDeuMethod and BGeMethod implementations with local scoring functions and greedy hill-climbing
  - `sparselink/src/sparselink/methods/__init__.py` - Added imports for BDeuMethod, BGeMethod
  - `sparselink/tests/test_bayesian_methods.py` - 13 tests covering registration, DAG output, edge detection, sparsity on independent data
- **Learnings:**
  - BDeu requires discretization of continuous data; `np.digitize` with linspace bins works well
  - BGe prior scatter matrix T0 must be scaled by (alpha_w - p - 1) for proper prior; alpha_w must be > p + 1
  - Greedy hill-climbing with acyclicity check via DFS is simple and effective for small networks
  - `scipy.special.gammaln` is the only extra dependency needed beyond numpy
---


## 2026-04-11 - US-011
- What was implemented: Full type annotations on all public APIs in both sparselink and pyGS packages. mypy strict mode passes on both. Explicit `__all__` in every `__init__.py`. `py.typed` marker files added.
- Files changed:
  - `sparselink/pyproject.toml` - Added mypy overrides for third-party untyped libs (sklearn, scipy, pandas, causallearn, torch)
  - `sparselink/src/sparselink/py.typed` - PEP 561 marker file
  - `pyproject.toml` - Added mypy_path, packages config, overrides for third-party libs, exclude vendor
  - `src/py.typed` - PEP 561 marker file
  - `src/datastruct/__init__.py` - Added `__all__`
  - `src/analyze/__init__.py` - Added `__all__` and imports
  - `src/bootstrap/__init__.py` - Added `__all__` and imports
  - `src/datastruct/Experiment.py` - Full type annotations on all methods/properties
  - `src/datastruct/Dataset.py` - Fixed union-attr and no-any-return errors
  - `src/datastruct/Network.py` - Fixed type assignments, return types, override
  - `src/datastruct/random.py` - Added function type annotation
  - `src/datastruct/scalefree.py` - Added function type annotation
  - `src/datastruct/stabilize.py` - Added function type annotation
  - `src/analyze/DataModel.py` - Full type annotations
  - `src/analyze/Data.py` - Full rewrite with type annotations
  - `src/analyze/Model.py` - Fixed no-any-return
  - `src/methods/__init__.py` - Added type annotation to `run()` function
  - `src/methods/lasso.py` - Fixed kwargs type, union-attr
  - `src/methods/lsco.py` - Fixed union-attr
  - `src/methods/scenicplus.py` - Added type annotations to internal functions
  - `src/methods/nestboot.py` - Fixed no-redef, type-arg, assignment, var-annotated errors
  - `src/bootstrap/utils.py` - Rewritten with proper type annotations
  - `src/bootstrap/nb_fdr_analysis.py` - Declared snakemake global
  - `src/bootstrap/generate_plots.py` - Declared snakemake global
  - `src/bootstrap/compute_density.py` - Declared snakemake global
  - `src/bio/evaluation.py` - Fixed return type for compare_multiple
  - `src/bio/visualization.py` - Fixed tuple type parameters
  - `src/bio/wrappers.py` - Fixed return type
- **Learnings:**
  - mypy strict with third-party untyped libs: use `[[tool.mypy.overrides]]` with `ignore_missing_imports = true` per module pattern
  - mypy cache must be cleared (`rm -rf .mypy_cache`) after config changes for overrides to take effect
  - Vendored code should use `follow_imports = "skip"` in mypy overrides
  - Snakemake scripts need `snakemake: Any` declaration at module level for type checking
  - numpy operations like `np.sign()`, `np.sum()` return `Any` in strict mode — assign to typed variable first
  - `mypy_path` in pyproject.toml accepts list format: `["src", "sparselink/src"]`
---


## 2026-04-11 - US-012
- What was implemented: Configured ruff for linting (replaces flake8 + isort + pyupgrade) and formatting (black-compatible) in both pyGS and sparselink. Added pre-commit hooks config. Fixed all lint errors so both packages pass cleanly.
- Files changed:
  - `pyproject.toml` - Added `[tool.ruff.lint]`, `[tool.ruff.lint.per-file-ignores]`, `[tool.ruff.lint.isort]`, `[tool.ruff.format]` sections
  - `sparselink/pyproject.toml` - Added `[tool.ruff.lint]`, `[tool.ruff.lint.isort]`, `[tool.ruff.format]` sections
  - `.pre-commit-config.yaml` - Created with ruff lint (--fix) and ruff-format hooks
  - `src/__init__.py` - Fixed trailing whitespace in docstring
  - `src/datastruct/Network.py` - Added noqa for graphistry availability import
  - `src/datastruct/stabilize.py` - Renamed ambiguous variable `I` to `eye`
  - `src/methods/nestboot.py` - Removed unused variables (eps, current_fdr, found, fp, curr_orig_index, param_list)
  - `src/methods/scenicplus.py` - Removed unused snakemake_config_overrides variable
  - `sparselink/src/sparselink/types.py` - Replaced `Union` with `X | Y` syntax
  - `sparselink/src/sparselink/registry.py` - Replaced `Type` with `type` (UP006/UP035)
  - Multiple files reformatted by ruff format
- **Learnings:**
  - ruff `select = ["E", "F", "I", "UP", "W"]` covers flake8 + isort + pyupgrade in one tool
  - Snakemake scripts need per-file-ignores for F821 (undefined `snakemake` global injected at runtime)
  - `ruff format` doesn't touch content inside docstrings/`__doc__` assignments — those need manual whitespace fixes
  - `from __future__ import annotations` enables `X | Y` syntax for type aliases at runtime on Python 3.10+
---


## 2026-04-11 - US-013
- What was implemented: GitHub Actions CI/CD workflows for both pyGS and sparselink packages. CI workflow with Python 3.10/3.11/3.12 matrix, pytest+pytest-cov, and ruff linting step. Release workflow triggered by version tags that builds and publishes both packages to PyPI using trusted publishing. Removed old single-version pytest.yml.
- Files changed:
  - `.github/workflows/ci.yml` - New CI workflow (lint + test matrix)
  - `.github/workflows/release.yml` - New release workflow (tag → build → PyPI publish)
  - `.github/workflows/pytest.yml` - Removed (superseded by ci.yml)
- **Learnings:**
  - GitHub Actions `pypa/gh-action-pypi-publish@release/v1` supports trusted publishing via OIDC (id-token: write permission)
  - Matrix strategy with `include` allows mapping package names to directory paths
  - Branch protection must be configured in GitHub repo settings (Settings → Branches → Add rule for `main`), not in workflow files
---


## 2026-04-11 - US-016
- What was implemented: Achieved 94% test coverage on sparselink (above 80% target). Added unit tests for bench module (synthetic, metrics, nestboot, runner) and integration tests for the pipeline runner. Configured pytest-cov reporting with `--cov-fail-under=80` in pyproject.toml.
- Files changed:
  - `sparselink/tests/test_bench.py` - Unit tests for synthetic data generation, evaluation metrics, NestBoot, and benchmark runner (16 tests)
  - `sparselink/tests/test_bench_integration.py` - Integration tests for pipeline runner end-to-end flow (4 tests)
  - `sparselink/pyproject.toml` - Added `[tool.pytest.ini_options]`, `[tool.coverage.run]`, `[tool.coverage.report]` sections
- **Learnings:**
  - pytest-cov `--cov-fail-under=80` in pyproject.toml enforces coverage gate in CI without extra config
  - `[tool.coverage.run] omit` is useful for excluding CLI entry points that are hard to unit test
  - The `evaluate()` function uses median of nonzero values as default threshold — zero predictions don't mean zero recall
  - sparselink tests must run from `sparselink/` dir with `PYTHONPATH=src` due to namespace conflict with parent directory
---


## 2026-04-11 - US-015
- What was implemented: MkDocs documentation site with Material theme, mkdocstrings for auto-generated API reference, quickstart guide, method comparison table, contributing guide, and GitHub Pages deployment workflow.
- Files changed:
  - `mkdocs.yml` - MkDocs configuration with Material theme, mkdocstrings plugin, nav structure
  - `docs/site/index.md` - Homepage with feature overview and quick links
  - `docs/site/quickstart.md` - Installation, basic usage, benchmarking, CLI examples
  - `docs/site/methods.md` - Method comparison table (20 methods) with category details and selection guide
  - `docs/site/contributing.md` - Dev setup, testing, code quality, adding methods, project structure
  - `docs/site/api/sparselink.md` - Auto-generated API docs for core sparselink
  - `docs/site/api/methods.md` - Auto-generated API docs for all inference methods
  - `docs/site/api/bench.md` - Auto-generated API docs for benchmarking module
  - `docs/site/api/bio.md` - Auto-generated API docs for biology subpackage
  - `.github/workflows/docs.yml` - GitHub Actions workflow deploying to GitHub Pages
  - `pyproject.toml` - Updated docs optional deps to mkdocs-material + mkdocstrings
- **Learnings:**
  - mkdocstrings `paths` option in mkdocs.yml must point to source roots (src, sparselink/src) for module resolution
  - `docs_dir` in mkdocs.yml allows placing source markdown in a subdirectory while keeping mkdocs.yml at project root
  - GitHub Pages deployment uses actions/deploy-pages@v4 with `pages: write` and `id-token: write` permissions
---
