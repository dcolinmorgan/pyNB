# Stable Import Paths & Deprecation Policy

## Stable Import Paths

These public APIs are guaranteed stable within a major version (1.x):

### sparselink

```python
from sparselink import InferenceMethod, get_method, list_methods, registry
from sparselink.types import AdjacencyMatrix, EdgeList, InferenceResult
from sparselink.bench import generate_network, generate_expression, evaluate, NestBoot
```

### pyGS

```python
from pyGS import NetworkBootstrap, create_network_bootstrap, Nestboot
from pyGS.bio import (
    load_expression_matrix,
    filter_tf_targets,
    scenicplus_infer,
    compare_to_gold_standard,
    plot_grn,
)
from pyGS.methods import run
```

## Deprecation Policy

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes to public API
- **MINOR** (1.X.0): New features, deprecations announced
- **PATCH** (1.0.X): Bug fixes only

### Deprecation Process

1. A `DeprecationWarning` is emitted for **1 minor version** before removal.
2. The CHANGELOG documents all deprecations with the planned removal version.
3. Deprecated features continue to work during the warning period.

**Example timeline:**
- v1.1.0: `old_function()` deprecated with warning → use `new_function()`
- v1.2.0: `old_function()` removed

### How to See Deprecation Warnings

```python
import warnings
warnings.filterwarnings("default", category=DeprecationWarning, module="pyGS|sparselink")
```

Or run Python with `-Wd` flag:
```bash
python -Wd your_script.py
```
