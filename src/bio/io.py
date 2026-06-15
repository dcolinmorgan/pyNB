"""Data loading utilities."""
import numpy as np
import pandas as pd
from pathlib import Path


def detect_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mapping = {".h5ad": "h5ad", ".csv": "csv", ".tsv": "tsv", ".npy": "npy", ".txt": "tsv"}
    if suffix not in mapping:
        raise ValueError(f"Unsupported format: {suffix}")
    return mapping[suffix]


def load_expression(path: str) -> tuple[np.ndarray, list[str]]:
    """Load expression matrix. Returns (samples x genes, gene_names)."""
    fmt = detect_format(path)
    if fmt == "h5ad":
        try:
            import scanpy as sc
        except ImportError:
            raise ImportError("scanpy required for .h5ad: pip install scanpy")
        adata = sc.read_h5ad(path)
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        return X, list(adata.var_names)
    if fmt in ("csv", "tsv"):
        sep = "," if fmt == "csv" else "\t"
        df = pd.read_csv(path, sep=sep, index_col=0)
        return df.values.astype(float), list(df.columns)
    arr = np.load(path)
    return arr, [f"V{i}" for i in range(arr.shape[1])]


def load_tf_list(path: str) -> list[str]:
    """Load TF names from text file (one per line)."""
    return Path(path).read_text().strip().splitlines()
