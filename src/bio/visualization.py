"""Domain-specific visualization for GRN analysis.

Network plots with gene annotations, TF highlighting, and edge weight display.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Dict, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:  # pragma: no cover
    plt = None  # type: ignore


def plot_grn(
    adjacency: np.ndarray,
    gene_names: List[str],
    tf_names: Optional[List[str]] = None,
    threshold: float = 0.0,
    title: str = "Gene Regulatory Network",
    figsize: tuple = (10, 10),
    save_path: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Plot a gene regulatory network with gene annotations.

    Parameters
    ----------
    adjacency : np.ndarray
        Weighted adjacency matrix (n_genes x n_genes).
    gene_names : list of str
        Gene names for node labels.
    tf_names : list of str, optional
        Transcription factors to highlight.
    threshold : float
        Minimum absolute edge weight to display.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure. If None, displays interactively.

    Returns
    -------
    fig : matplotlib Figure or None
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualization")

    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for network visualization")

    # Build graph
    G = nx.DiGraph()
    tf_set = set(tf_names) if tf_names else set()

    for i, name in enumerate(gene_names):
        G.add_node(name, is_tf=(name in tf_set))

    for i in range(len(gene_names)):
        for j in range(len(gene_names)):
            w = adjacency[i, j]
            if abs(w) > threshold and i != j:
                G.add_edge(gene_names[i], gene_names[j], weight=w)

    if G.number_of_edges() == 0:
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    pos = nx.spring_layout(G, seed=42)

    # Node colors
    node_colors = ["#e74c3c" if G.nodes[n].get("is_tf") else "#3498db" for n in G.nodes()]

    # Edge widths scaled by weight
    weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    edge_widths = [1.0 + 2.0 * (w / max_w) for w in weights]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=300)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.6, arrows=True)

    # Legend
    patches = [
        mpatches.Patch(color="#e74c3c", label="TF"),
        mpatches.Patch(color="#3498db", label="Target"),
    ]
    ax.legend(handles=patches, loc="upper left")
    ax.set_title(title)
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_evaluation_summary(
    metrics: Dict[str, float],
    title: str = "GRN Evaluation",
    save_path: Optional[str] = None,
) -> Any:
    """Bar plot of evaluation metrics (AUROC, AUPR, F1, MCC).

    Parameters
    ----------
    metrics : dict
        Metric name -> value mapping.
    title : str
        Plot title.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : matplotlib Figure or None
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualization")

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(metrics.keys())
    values = list(metrics.values())

    bars = ax.bar(names, values, color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random baseline")
    ax.legend()

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
