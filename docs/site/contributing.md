# Contributing

## Development Setup

```bash
git clone https://github.com/dcolinmorgan/pyGS.git
cd pyGS
python -m venv .venv && source .venv/bin/activate
pip install -e "./sparselink[test]"
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
# sparselink
cd sparselink && PYTHONPATH=src pytest tests/ -q && cd ..

# pyGS
pytest tests/ -q
```

## Code Quality

```bash
ruff check src/ sparselink/src/    # lint
ruff format src/ sparselink/src/   # format
mypy                                # type check
```

## Adding a New Inference Method

1. Create `sparselink/src/sparselink/methods/your_method.py`:

```python
from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData

@registry.register
class YourMethod(InferenceMethod):
    name = "your_method"

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        data = self._to_array(X)
        # ... compute adjacency matrix ...
        return InferenceResult(adjacency_matrix=adj)
```

2. Import it in `sparselink/src/sparselink/methods/__init__.py`
3. Add tests in `sparselink/tests/`

## Project Structure

```
pyGS/
├── src/                    # pyGS package (biology-focused wrappers)
│   ├── methods/            # Thin wrappers delegating to sparselink
│   ├── bio/                # Biology-specific preprocessing/evaluation
│   ├── datastruct/         # Data structures (Experiment, Dataset, Network)
│   ├── analyze/            # Analysis utilities
│   └── bootstrap/          # NestBoot Snakemake workflows
├── sparselink/             # Standalone inference engine
│   └── src/sparselink/
│       ├── methods/        # All inference algorithms
│       ├── bench/          # Benchmarking (synthetic, metrics, runner)
│       ├── base.py         # InferenceMethod ABC
│       ├── registry.py     # Method registry
│       └── types.py        # Common types
├── tests/                  # pyGS tests
├── docs/site/              # MkDocs documentation source
└── mkdocs.yml              # MkDocs configuration
```

## Conventions

- Data is `(samples × features)` — sparselink convention
- All methods return `InferenceResult` with `.adjacency_matrix` and `.edge_list`
- Use `@registry.register` decorator; set `name` class attribute
- Type annotations required on all public APIs (mypy strict)
- Format with ruff (`line-length = 88`)

## Submitting Changes

1. Fork and create a feature branch
2. Make changes following the conventions above
3. Ensure `ruff check`, `ruff format --check`, and `mypy` pass
4. Run tests with `pytest`
5. Open a PR against `main`
