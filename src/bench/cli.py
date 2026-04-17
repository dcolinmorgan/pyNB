#!/usr/bin/env python3
"""pygs — CLI for gene regulatory network inference & benchmarking.

Wraps sparselink inference methods with GeneSpider data, NestBoot FDR,
biology-specific preprocessing, and gold-standard evaluation.

Usage::

    pygs status                                  # system info
    pygs methods                                 # list inference methods
    pygs infer data.csv -m lasso                 # infer a GRN
    pygs bench --tier fast                       # synthetic benchmark
    pygs bench-gs --tier fast --sizes N50        # GeneSpider benchmark
    pygs nestboot data.csv -m lasso              # NestBoot FDR analysis
    pygs evaluate pred.npy --gold gold.npy       # evaluate against gold standard
    pygs plot pred.npy --genes genes.txt         # plot a GRN
    pygs dashboard -i results.json               # HTML dashboard
    pygs show results.json                       # render result table
    pygs                                         # interactive mode
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

console = Console()

# ── Palette ───────────────────────────────────────────────────────────────
TEAL = "bold #14B8A6"
INDIGO = "bold #818CF8"
GREEN = "#22C55E"
ORANGE = "#FB923C"
ROSE = "#FB7185"
DIM = "#71717A"
BOLD = "bold #FAFAFA"

BANNER = [
    " ▄▄▄▄  ▄  ▄  ▄▄▄▄  ▄▄▄▄",
    " █  █  █  █  █     █    ",
    " █▀▀█  ▀▄▄█  █ ▀█▀ ▀▀▀█",
    " █     ▄  █  █  █  ▄  █",
    " ▀     ▀▀▀▀  ▀▀▀▀  ▀▀▀▀",
]

# Method categories for display
METHOD_CATEGORIES: dict[str, str] = {
    "lasso": "regression",
    "elastic_net": "regression",
    "ridge": "regression",
    "lsco": "regression",
    "tigress": "stability selection",
    "genie3": "tree-based",
    "clr": "information theory",
    "partial_correlation": "correlation",
    "glasso": "graphical model",
    "glasso_stars": "graphical model",
    "neighborhood_selection": "graphical model",
    "pcmci": "causal (time-series)",
    "granger_causality": "causal (time-series)",
    "transfer_entropy": "causal (time-series)",
    "pc": "constraint-based",
    "fci": "constraint-based",
    "notears": "continuous optimization",
    "dag_gnn": "deep learning",
    "bdeu": "bayesian",
    "bge": "bayesian",
}


# ── Helpers ───────────────────────────────────────────────────────────────


def _print_banner() -> None:
    for line in BANNER:
        console.print(f"[{TEAL}]{line}[/]")
    console.print(
        f"  [{BOLD}]pyGS[/]  [{DIM}]python genespider —"
        f" network inference & benchmarking[/]"
    )
    console.print()


def _color(val: float, lo: float = 0.4, hi: float = 0.7) -> str:
    if val >= hi:
        return GREEN
    if val >= lo:
        return ORANGE
    return ROSE


def _bar(value: float, width: int = 20) -> str:
    v = max(0.0, min(1.0, value))
    filled = int(v * width)
    return f"[{GREEN}]{'█' * filled}[/][dim]{'░' * (width - filled)}[/]"


def _pick_multi(
    label: str, options: dict[str, str], default: str = ""
) -> list[str]:
    for k, v in options.items():
        console.print(f"    [{INDIGO}]{k}[/] [{DIM}]{v}[/]")
    raw = Prompt.ask(f"  [{DIM}]{label} (comma-separated)[/]", default=default)
    keys = [k.strip() for k in raw.split(",")]
    return [options[k] for k in keys if k in options]


def _render_results(data: list[dict], title: str = "Results") -> None:
    """Render benchmark results as a rich table."""
    if not data:
        console.print("[dim]No results[/]")
        return

    ok = [r for r in data if not r.get("error")]
    errs = [r for r in data if r.get("error")]
    methods = sorted(set(r["method"] for r in ok))
    metrics = ["auroc", "aupr", "f1", "mcc"]

    t = Table(title=title, title_style=TEAL, border_style="dim")
    t.add_column("Method", style="bold")
    for m in metrics:
        t.add_column(m.upper(), justify="right")
    t.add_column("", min_width=22)
    t.add_column("Time", justify="right")
    t.add_column("Runs", justify="right")

    for method in methods:
        rows = [r for r in ok if r["method"] == method]
        avgs = {m: np.mean([r.get(m, 0) for r in rows]) for m in metrics}
        avg_time = np.mean([r.get("elapsed_sec", 0) for r in rows])
        t.add_row(
            method,
            *[f"[{_color(avgs[m])}]{avgs[m]:.3f}[/]" for m in metrics],
            _bar(avgs["auroc"]),
            f"{avg_time:.2f}s",
            str(len(rows)),
        )
    console.print(t)
    if errs:
        console.print(f"  [{DIM}]{len(errs)} errors skipped[/]")

    # SNR breakdown (for GeneSpider results)
    snrs = sorted(set(r.get("snr", 0) for r in ok))
    if len(snrs) > 1:
        console.print()
        for snr in snrs:
            sub = [r for r in ok if r.get("snr") == snr]
            st = Table(
                title=f"SNR={snr}",
                title_style=f"bold {ORANGE}",
                border_style="dim",
                show_edge=False,
            )
            st.add_column("Method", style="bold", width=25)
            st.add_column("AUROC", justify="right")
            st.add_column("F1", justify="right")
            st.add_column("MCC", justify="right")
            st.add_column("", min_width=22)
            for method in methods:
                rows = [r for r in sub if r["method"] == method]
                if not rows:
                    continue
                a = np.mean([r.get("auroc", 0) for r in rows])
                f = np.mean([r.get("f1", 0) for r in rows])
                m = np.mean([r.get("mcc", 0) for r in rows])
                st.add_row(
                    method,
                    f"[{_color(a)}]{a:.3f}[/]",
                    f"[{_color(f)}]{f:.3f}[/]",
                    f"[{_color(m, 0.2, 0.5)}]{m:.3f}[/]",
                    _bar(a),
                )
            console.print(st)


# ── Status ────────────────────────────────────────────────────────────────


def _cmd_status(args: argparse.Namespace) -> None:
    """Show sparselink + pyGS system status."""
    # sparselink status
    tree = Tree(f"[{TEAL}]sparselink[/]", guide_style="dim")
    try:
        from sparselink import list_methods

        import sparselink.methods  # noqa: F401

        methods = list_methods()
        mb = tree.add(f"Methods ({len(methods)})")
        for m in methods:
            mb.add(f"[{GREEN}]✓[/] {m}")
    except Exception as e:
        tree.add(f"[{ROSE}]✗ {e}[/]")

    accel = tree.add("Acceleration")
    try:
        import mlx.core  # noqa: F401

        accel.add(f"[{GREEN}]✓[/] MLX (Apple Silicon)")
    except ImportError:
        accel.add("[dim]○[/] MLX not available")

    deps = tree.add("Optional deps")
    for pkg, label in [("causallearn", "causal"), ("torch", "deep")]:
        try:
            __import__(pkg)
            deps.add(f"[{GREEN}]✓[/] {label}")
        except ImportError:
            deps.add(f"[dim]○[/] {label} — pip install sparselink[{label}]")

    console.print(tree)

    # pyGS status
    pygs_tree = Tree(f"[{TEAL}]pyGS[/]", guide_style="dim")

    cache = Path(".gs_cache")
    if cache.exists():
        n = len(list(cache.glob("*.json")))
        pygs_tree.add(f"[{GREEN}]✓[/] GeneSpider cache: {n} files")
    else:
        pygs_tree.add(
            f"[dim]○[/] No GeneSpider cache (auto-downloaded on first run)"
        )

    mods = pygs_tree.add("pyGS modules")
    for mod in [
        "analyze.Data",
        "datastruct.Network",
        "methods.nestboot",
        "bio.evaluation",
        "bio.preprocessing",
        "bench.genespider",
    ]:
        try:
            __import__(mod)
            mods.add(f"[{GREEN}]✓[/] {mod}")
        except ImportError:
            mods.add(f"[dim]○[/] {mod}")

    console.print(pygs_tree)


# ── Methods ───────────────────────────────────────────────────────────────


def _cmd_methods(args: argparse.Namespace) -> None:
    """List all registered inference methods."""
    from sparselink import list_methods

    import sparselink.methods  # noqa: F401

    t = Table(title="Available Methods", title_style=TEAL, border_style="dim")
    t.add_column("Name", style="bold")
    t.add_column("Category", style=DIM)

    for m in sorted(list_methods()):
        t.add_row(m, METHOD_CATEGORIES.get(m, ""))
    console.print(t)


# ── Infer ─────────────────────────────────────────────────────────────────


def _cmd_infer(args: argparse.Namespace) -> None:
    """Infer a network from a data file using a sparselink method."""
    warnings.simplefilter("ignore")
    from sparselink import get_method

    import sparselink.methods  # noqa: F401

    path = Path(args.file)
    if path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        X = df.select_dtypes(include=[np.number]).values
        feature_names = list(df.select_dtypes(include=[np.number]).columns)
    elif path.suffix in (".tsv", ".txt"):
        import pandas as pd

        df = pd.read_csv(path, sep="\t")
        X = df.select_dtypes(include=[np.number]).values
        feature_names = list(df.select_dtypes(include=[np.number]).columns)
    elif path.suffix == ".npy":
        X = np.load(path)
        feature_names = [f"V{i}" for i in range(X.shape[1])]
    else:
        console.print(
            f"[{ROSE}]Unsupported format: {path.suffix}."
            f" Use .csv, .tsv, .txt, or .npy[/]"
        )
        return

    console.print(
        f"  [{TEAL}]Data[/]    {path.name}:"
        f" {X.shape[0]} samples × {X.shape[1]} features"
    )
    console.print(f"  [{TEAL}]Method[/]  {args.method}")

    method = get_method(args.method)
    with console.status(f"[{TEAL}]Running {args.method}...[/]"):
        result = method().fit(X)

    adj = result.adjacency_matrix
    n_edges = int(np.count_nonzero(adj) - np.count_nonzero(np.diag(adj)))
    console.print(f"  [{GREEN}]✓[/] {n_edges} edges inferred")

    # Top edges
    edges = (
        result.edge_list
        if hasattr(result, "edge_list") and result.edge_list
        else []
    )
    if not edges:
        mask = ~np.eye(adj.shape[0], dtype=bool)
        idx = np.argsort(-np.abs(adj[mask]))
        rows_i, cols_j = np.where(mask)
        edges = [
            (rows_i[i], cols_j[i], adj[rows_i[i], cols_j[i]])
            for i in idx[:20]
        ]

    t = Table(title="Top Edges", title_style=TEAL, border_style="dim")
    t.add_column("Source", style="bold")
    t.add_column("Target", style="bold")
    t.add_column("Weight", justify="right")
    for src, tgt, w in edges[:15]:
        sn = feature_names[src] if src < len(feature_names) else str(src)
        tn = feature_names[tgt] if tgt < len(feature_names) else str(tgt)
        t.add_row(sn, tn, f"{w:.4f}")
    console.print(t)

    if args.output:
        np.save(args.output, adj)
        console.print(f"  [{DIM}]Adjacency saved to {args.output}[/]")


# ── Synthetic benchmark ───────────────────────────────────────────────────


def _configure_synthetic() -> argparse.Namespace:
    """Interactive config builder for synthetic benchmark."""
    console.print(f"\n  [{TEAL}]Configure Synthetic Benchmark[/]\n")

    console.print(f"  [{TEAL}]A) Method tier[/]")
    tiers = _pick_multi(
        "Tiers",
        {"1": "fast", "2": "medium", "3": "slow", "4": "very_slow"},
        default="1",
    )
    tier = ",".join(tiers) if tiers else "fast"

    n_nodes = int(Prompt.ask(f"\n  [{DIM}]Number of genes[/]", default="50"))
    n_datasets = int(Prompt.ask(f"  [{DIM}]Number of datasets[/]", default="5"))
    timeout = int(
        Prompt.ask(f"  [{DIM}]Timeout per method (seconds)[/]", default="60")
    )
    output = Prompt.ask(
        f"  [{DIM}]Output file[/]", default="benchmark_results.json"
    )

    console.print()
    return argparse.Namespace(
        tier=tier,
        n_nodes=n_nodes,
        n_samples=n_nodes * 4,
        n_datasets=n_datasets,
        seed=42,
        timeout=timeout,
        output=output,
    )


def _run_benchmark_live(args: argparse.Namespace) -> None:
    """Run synthetic benchmark using sparselink methods."""
    warnings.simplefilter("ignore")

    from sparselink import list_methods
    from sparselink.bench.run_benchmark import TIERS, run_single
    from sparselink.bench.synthetic import generate_data, generate_network

    import sparselink.methods  # noqa: F401

    selected_tiers = [t.strip() for t in args.tier.split(",")]
    methods: list[str] = []
    for t in selected_tiers:
        methods.extend(TIERS.get(t, []))
    registered = set(list_methods())
    methods = [m for m in methods if m in registered]

    sparsities = [0.02, 0.06, 0.1]
    noise_levels = [0.01, 0.1, 1.0]
    topologies = ["random", "scalefree", "smallworld"]
    total = (
        len(methods)
        * len(topologies)
        * len(sparsities)
        * len(noise_levels)
        * args.n_datasets
    )

    console.print(f"  [{TEAL}]Methods[/]       {', '.join(methods)}")
    console.print(f"  [{TEAL}]Topologies[/]    {topologies}")
    console.print(f"  [{TEAL}]Nodes[/]         {args.n_nodes}")
    console.print(f"  [{TEAL}]Datasets[/]      {args.n_datasets}")
    console.print(f"  [{TEAL}]Total runs[/]    {total}")
    console.print(f"  [{TEAL}]Timeout[/]       {args.timeout}s")
    console.print()

    progress = Progress(
        SpinnerColumn(style=TEAL),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style=TEAL, finished_style=GREEN),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    results: list[dict] = []
    rng = np.random.default_rng(args.seed)

    with progress:
        task = progress.add_task("Benchmarking", total=total)
        for topo in topologies:
            for sp in sparsities:
                for noise_std in noise_levels:
                    for ds_idx in range(args.n_datasets):
                        ds_seed = int(rng.integers(0, 2**31))
                        true_net = generate_network(
                            args.n_nodes,
                            topology=topo,
                            sparsity=sp,
                            seed=ds_seed,
                        )
                        X = generate_data(
                            true_net,
                            n_samples=args.n_samples,
                            noise_std=noise_std,
                            seed=ds_seed,
                        )
                        for method_name in methods:
                            progress.update(
                                task,
                                description=(
                                    f"{method_name:20s}"
                                    f" {topo}/sp={sp}/noise={noise_std}"
                                ),
                            )
                            r = run_single(
                                method_name,
                                X,
                                true_net,
                                ds_idx,
                                args.n_nodes,
                                args.n_samples,
                                topo,
                                sp,
                                noise_std,
                                args.timeout,
                            )
                            results.append(asdict(r))
                            progress.advance(task)

    _render_results(results, f"Synthetic Benchmark ({total} runs)")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n  [{DIM}]Results saved to {args.output}[/]")


# ── GeneSpider benchmark ─────────────────────────────────────────────────


def _configure_genespider() -> argparse.Namespace:
    """Interactive config builder for GeneSpider benchmark."""
    console.print(f"\n  [{TEAL}]Configure GeneSpider Benchmark[/]\n")

    console.print(f"  [{TEAL}]A) Method tier[/]")
    tiers = _pick_multi(
        "Tiers",
        {"1": "fast", "2": "medium", "3": "slow", "4": "nestboot"},
        default="1",
    )

    use_nestboot = "nestboot" in tiers
    tier_names = [t for t in tiers if t != "nestboot"]
    tier = ",".join(tier_names) if tier_names else "fast"

    # NestBoot params
    nest_runs = 0
    boot_runs = 0
    fdr = 0.05
    if use_nestboot:
        console.print(f"\n  [{TEAL}]NestBoot config[/]")
        nest_runs = int(
            Prompt.ask(f"  [{DIM}]Outer runs[/]", default="10")
        )
        boot_runs = int(
            Prompt.ask(f"  [{DIM}]Inner runs[/]", default="10")
        )
        fdr = float(
            Prompt.ask(f"  [{DIM}]FDR threshold[/]", default="0.05")
        )

    console.print(f"\n  [{TEAL}]B) Network sizes[/]")
    sizes = _pick_multi(
        "Sizes",
        {"1": "N10", "2": "N50", "3": "N100"},
        default="2",
    )
    sizes_str = ",".join(sizes) if sizes else "N50"

    console.print(f"\n  [{TEAL}]C) Max datasets per size[/]")
    max_ds = int(Prompt.ask(f"  [{DIM}]0 = all[/]", default="0"))

    timeout = int(
        Prompt.ask(f"\n  [{DIM}]Timeout per method (seconds)[/]", default="120")
    )
    output = Prompt.ask(
        f"  [{DIM}]Output file[/]", default="benchmark_genespider.json"
    )

    console.print()
    return argparse.Namespace(
        tier=tier,
        sizes=sizes_str,
        max_datasets=max_ds,
        timeout=timeout,
        output=output,
        nestboot=use_nestboot,
        nest_runs=nest_runs,
        boot_runs=boot_runs,
        fdr=fdr,
        seed=42,
    )


def _run_genespider_live(args: argparse.Namespace) -> None:
    """Run GeneSpider benchmark using sparselink methods on real data."""
    warnings.simplefilter("ignore")

    from bench.genespider import TIERS, _list_datasets, load_dataset, run_single
    from sparselink import list_methods

    import sparselink.methods  # noqa: F401

    # Parse tiers — "nestboot" in the tier list enables NestBoot wrapping
    selected_tiers = [t.strip() for t in args.tier.split(",")]
    use_nestboot = getattr(args, "nestboot", False) or "nestboot" in selected_tiers
    method_tiers = [t for t in selected_tiers if t != "nestboot"]
    if not method_tiers:
        method_tiers = ["fast"]

    methods: list[str] = []
    for t in method_tiers:
        methods.extend(TIERS.get(t, []))
    registered = set(list_methods())
    methods = [m for m in methods if m in registered]
    sizes = [s.strip() for s in args.sizes.split(",")]

    console.print(f"  [{TEAL}]Methods[/]  {', '.join(methods)}")
    console.print(f"  [{TEAL}]Sizes[/]    {sizes}")
    console.print(f"  [{TEAL}]Timeout[/]  {args.timeout}s")
    if use_nestboot:
        console.print(
            f"  [{TEAL}]NestBoot[/] {args.nest_runs}×{args.boot_runs},"
            f" FDR={args.fdr}"
        )
    console.print()

    progress = Progress(
        SpinnerColumn(style=TEAL),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style=TEAL, finished_style=GREEN),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    results: list[dict] = []

    for size in sizes:
        console.print(f"  [{INDIGO}]Fetching {size} datasets...[/]")
        datasets = _list_datasets(size)
        if args.max_datasets > 0:
            datasets = datasets[: args.max_datasets]

        total = len(datasets) * len(methods)
        label = "NestBoot" if use_nestboot else "Direct"
        console.print(
            f"  [{DIM}]{len(datasets)} datasets × {len(methods)}"
            f" methods = {total} runs ({label})[/]"
        )

        with progress:
            task = progress.add_task(f"{size}", total=total)
            for ds_meta in datasets:
                try:
                    X, P, A_true, topology, net_name = load_dataset(
                        ds_meta, size
                    )
                except Exception:
                    progress.advance(task, len(methods))
                    continue

                for method_name in methods:
                    snr = ds_meta["snr"]
                    tag = f"{'NB:' if use_nestboot else ''}{method_name}"
                    progress.update(
                        task,
                        description=f"{tag:20s} {topology}/SNR={snr}",
                    )

                    if use_nestboot:
                        r = _run_nestboot_on_gs(
                            method_name,
                            X,
                            A_true,
                            ds_meta,
                            topology,
                            net_name,
                            args,
                        )
                    else:
                        r = run_single(
                            method_name,
                            X,
                            A_true,
                            ds_meta,
                            topology,
                            net_name,
                            args.timeout,
                            P,
                        )
                        r = asdict(r)
                    results.append(r)
                    progress.advance(task)

    _render_results(results, f"GeneSpider Benchmark ({len(results)} runs)")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n  [{DIM}]Results saved to {args.output}[/]")


def _run_nestboot_on_gs(
    method_name: str,
    X: np.ndarray,
    A_true: np.ndarray,
    ds_meta: dict,
    topology: str,
    net_name: str,
    args: argparse.Namespace,
) -> dict:
    """Run a single method through NestBoot on a GeneSpider dataset."""
    from sparselink import get_method
    from sparselink.bench.metrics import evaluate

    from config import AnalysisConfig
    from datastruct.Dataset import Dataset
    from datastruct.Network import Network
    from analyze.Data import Data
    from methods.nestboot import Nestboot

    n_genes = X.shape[1]
    gene_names = [f"G{i}" for i in range(n_genes)]

    # Build Dataset (X is samples×genes, Dataset wants genes×samples)
    ds = Dataset()
    ds._Y = X.T
    ds._P = np.eye(n_genes)
    ds._network = Network(A_true)
    ds._names = gene_names
    data_obj = Data(ds)

    method_cls = get_method(method_name)

    def _infer(dataset: Data, **kwargs: object) -> np.ndarray:
        Y = dataset.data.Y  # type: ignore[union-attr]
        result = method_cls().fit(Y.T)
        return result.adjacency_matrix

    config = AnalysisConfig(
        total_runs=args.nest_runs * args.boot_runs,
        inner_group_size=args.boot_runs,
        fdr_threshold=args.fdr,
    )
    nb = Nestboot(config)
    nb.logger.setLevel(logging.WARNING)

    base = dict(
        method=f"nestboot:{method_name}",
        dataset_name=ds_meta["path"],
        network_name=net_name,
        topology=topology,
        n_genes=ds_meta["n_genes"],
        snr=ds_meta["snr"],
    )
    fail = dict(
        auroc=0, aupr=0, precision=0, recall=0,
        f1=0, fdr=1, mcc=0, r2=0,
    )

    try:
        t0 = time.perf_counter()
        nb_results = nb.run_nestboot(
            dataset=data_obj,
            inference_method=_infer,
            nest_runs=args.nest_runs,
            boot_runs=args.boot_runs,
            seed=getattr(args, "seed", 42),
        )
        elapsed = time.perf_counter() - t0

        metrics = evaluate(A_true, nb_results.xnet)
        return {
            **base,
            "auroc": metrics.auroc,
            "aupr": metrics.aupr,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "fdr": metrics.fdr,
            "mcc": metrics.mcc,
            "r2": metrics.r2,
            "elapsed_sec": round(elapsed, 4),
            "error": None,
        }
    except Exception as e:
        return {**base, **fail, "elapsed_sec": 0, "error": str(e)[:120]}


# ── NestBoot FDR ──────────────────────────────────────────────────────────


def _cmd_nestboot(args: argparse.Namespace) -> None:
    """Run NestBoot FDR analysis on expression data using a sparselink method."""
    warnings.simplefilter("ignore")

    from sparselink import get_method

    import sparselink.methods  # noqa: F401

    from bio.preprocessing import load_expression_matrix
    from config import AnalysisConfig
    from methods.nestboot import Nestboot

    # Load expression data
    console.print(f"  [{TEAL}]Loading[/]  {args.file}")
    matrix, gene_names = load_expression_matrix(args.file)
    n_genes, n_samples = matrix.shape
    console.print(f"  [{DIM}]{n_genes} genes × {n_samples} samples[/]")

    # Build a Dataset for Nestboot
    from analyze.Data import Data
    from datastruct.Dataset import Dataset
    from datastruct.Network import Network

    ds = Dataset()
    ds._Y = matrix
    ds._P = np.eye(n_genes)
    ds._network = Network(np.zeros((n_genes, n_genes)))
    ds._names = gene_names
    data_obj = Data(ds)

    # Build inference callable using sparselink method
    method_name = args.method
    console.print(f"  [{TEAL}]Method[/]   {method_name}")
    console.print(
        f"  [{TEAL}]Runs[/]     {args.nest_runs} outer × {args.boot_runs} inner"
    )
    console.print(f"  [{TEAL}]FDR[/]      {args.fdr}")

    method_cls = get_method(method_name)

    def _infer(dataset: Data, **kwargs: object) -> np.ndarray:
        """Run sparselink method on a pyGS Data object."""
        Y = dataset.data.Y  # type: ignore[union-attr]
        X = Y.T  # sparselink expects (samples × features)
        result = method_cls().fit(X)
        return result.adjacency_matrix

    # Run NestBoot
    config = AnalysisConfig(
        total_runs=args.nest_runs * args.boot_runs,
        inner_group_size=args.boot_runs,
        fdr_threshold=args.fdr,
    )
    nb = Nestboot(config)
    nb.logger.setLevel(logging.WARNING)

    with console.status(
        f"[{TEAL}]Running NestBoot"
        f" ({args.nest_runs}×{args.boot_runs})...[/]"
    ):
        results = nb.run_nestboot(
            dataset=data_obj,
            inference_method=_infer,
            nest_runs=args.nest_runs,
            boot_runs=args.boot_runs,
            seed=args.seed,
        )

    # Report
    n_edges = int(np.sum(results.xnet != 0))
    console.print(f"\n  [{GREEN}]✓[/] {n_edges} edges at FDR ≤ {args.fdr}")
    console.print(f"  [{DIM}]Support threshold: {results.support}[/]")

    # Save adjacency
    out = Path(args.output)
    np.save(out, results.xnet)
    console.print(f"  [{DIM}]Adjacency saved to {out}[/]")

    # Save signed network
    signed_out = out.with_stem(out.stem + "_signed")
    np.save(signed_out, results.sxnet)
    console.print(f"  [{DIM}]Signed network saved to {signed_out}[/]")

    # Export text summary
    txt_out = out.with_suffix(".txt")
    nb.export_results(results, txt_out)
    console.print(f"  [{DIM}]Summary saved to {txt_out}[/]")


# ── Evaluate ──────────────────────────────────────────────────────────────


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a predicted GRN against a gold standard."""
    from bio.evaluation import compare_to_gold_standard

    pred_path = Path(args.predicted)
    if pred_path.suffix == ".npy":
        predicted = np.load(pred_path)
    elif pred_path.suffix in (".csv", ".tsv"):
        import pandas as pd

        sep = "\t" if pred_path.suffix == ".tsv" else ","
        predicted = pd.read_csv(pred_path, sep=sep, index_col=0).values
    else:
        console.print(f"[{ROSE}]Unsupported format: {pred_path.suffix}[/]")
        return

    gold_path = Path(args.gold)
    if gold_path.suffix == ".npy":
        gold = np.load(gold_path)
    elif gold_path.suffix == ".json":
        with open(gold_path) as f:
            data = json.load(f)
        gold = np.array(data["obj_data"]["A"])
    elif gold_path.suffix in (".csv", ".tsv"):
        import pandas as pd

        sep = "\t" if gold_path.suffix == ".tsv" else ","
        gold = pd.read_csv(gold_path, sep=sep, index_col=0).values
    else:
        console.print(f"[{ROSE}]Unsupported format: {gold_path.suffix}[/]")
        return

    console.print(
        f"  [{TEAL}]Predicted[/]  {pred_path.name}  {predicted.shape}"
    )
    console.print(
        f"  [{TEAL}]Gold std[/]   {gold_path.name}  {gold.shape}"
    )

    metrics = compare_to_gold_standard(predicted, gold)

    t = Table(title="GRN Evaluation", title_style=TEAL, border_style="dim")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    for k, v in metrics.items():
        color = _color(v) if k in ("AUROC", "AUPR", "F1", "MCC") else ""
        t.add_row(k, f"[{color}]{v:.4f}[/]" if color else f"{v:.4f}")
    console.print(t)


# ── Plot ──────────────────────────────────────────────────────────────────


def _cmd_plot(args: argparse.Namespace) -> None:
    """Plot a gene regulatory network."""
    from bio.visualization import plot_grn

    adj_path = Path(args.adjacency)
    if adj_path.suffix == ".npy":
        adj = np.load(adj_path)
    elif adj_path.suffix in (".csv", ".tsv"):
        import pandas as pd

        sep = "\t" if adj_path.suffix == ".tsv" else ","
        adj = pd.read_csv(adj_path, sep=sep, index_col=0).values
    else:
        console.print(f"[{ROSE}]Unsupported format: {adj_path.suffix}[/]")
        return

    n = adj.shape[0]

    if args.genes:
        with open(args.genes) as f:
            gene_names = [line.strip() for line in f if line.strip()]
    else:
        gene_names = [f"G{i + 1}" for i in range(n)]

    tf_names = None
    if args.tfs:
        with open(args.tfs) as f:
            tf_names = [line.strip() for line in f if line.strip()]

    out = args.output or f"{adj_path.stem}_grn.png"
    n_edges = int(np.count_nonzero(adj) - np.count_nonzero(np.diag(adj)))
    console.print(
        f"  [{TEAL}]Network[/]  {adj_path.name}"
        f"  ({n} genes, {n_edges} edges)"
    )

    plot_grn(
        adj,
        gene_names,
        tf_names=tf_names,
        threshold=args.threshold,
        save_path=out,
    )
    console.print(f"  [{GREEN}]✓[/] Saved to {out}")


# ── Show / Dashboard ─────────────────────────────────────────────────────


def _cmd_show(args: argparse.Namespace) -> None:
    """Render a previous result JSON."""
    with open(args.file) as f:
        data = json.load(f)
    _render_results(data, f"Results from {args.file}")


def _cmd_dashboard(args: argparse.Namespace) -> None:
    """Generate an interactive HTML dashboard from benchmark results."""
    # Dashboard generation is a pyGS feature — build a self-contained HTML
    import webbrowser

    with open(args.input) as f:
        data = json.load(f)

    if not data:
        console.print(f"[{ROSE}]No data in {args.input}[/]")
        return

    # Build simple HTML dashboard
    methods = sorted(set(r["method"] for r in data if not r.get("error")))
    ok = [r for r in data if not r.get("error")]

    rows_html = ""
    for m in methods:
        m_rows = [r for r in ok if r["method"] == m]
        avgs = {
            k: np.mean([r.get(k, 0) for r in m_rows])
            for k in ("auroc", "aupr", "f1", "mcc", "elapsed_sec")
        }
        rows_html += (
            f"<tr><td>{m}</td>"
            f"<td>{avgs['auroc']:.3f}</td>"
            f"<td>{avgs['aupr']:.3f}</td>"
            f"<td>{avgs['f1']:.3f}</td>"
            f"<td>{avgs['mcc']:.3f}</td>"
            f"<td>{avgs['elapsed_sec']:.2f}s</td>"
            f"<td>{len(m_rows)}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html><head><title>pyGS Benchmark Dashboard</title>
<style>
body {{ font-family: system-ui; margin: 2rem; background: #0f172a; color: #f8fafc; }}
h1 {{ color: #14b8a6; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
th, td {{ padding: 0.5rem 1rem; text-align: left; border-bottom: 1px solid #334155; }}
th {{ color: #94a3b8; font-weight: 600; }}
tr:hover {{ background: #1e293b; }}
</style></head><body>
<h1>pyGS Benchmark Dashboard</h1>
<p>{len(ok)} successful runs, {len(data) - len(ok)} errors</p>
<table>
<tr><th>Method</th><th>AUROC</th><th>AUPR</th><th>F1</th><th>MCC</th><th>Time</th><th>Runs</th></tr>
{rows_html}
</table>
<script>
const data = {json.dumps(data)};
</script>
</body></html>"""

    with open(args.output, "w") as f:
        f.write(html)
    console.print(f"  [{GREEN}]✓[/] Dashboard saved to {args.output}")

    if not args.no_open:
        webbrowser.open(str(Path(args.output).resolve()))


# ── Interactive mode ──────────────────────────────────────────────────────

_MENU = {
    "1": ("status", "Show system status & available methods"),
    "2": ("methods", "List all inference methods"),
    "3": ("infer", "Infer a network from a data file"),
    "4": ("bench", "Run synthetic benchmark"),
    "5": ("bench-gs", "Run GeneSpider benchmark (+ NestBoot)"),
    "6": ("evaluate", "Evaluate predicted GRN vs gold standard"),
    "7": ("plot", "Plot a gene regulatory network"),
    "8": ("dashboard", "Generate interactive HTML dashboard"),
    "9": ("show", "Render a previous result JSON"),
}


def _interactive() -> None:
    _print_banner()

    console.print(f"[{INDIGO}]Interactive mode[/]  [{DIM}]Ctrl+C to exit[/]\n")
    for key, (_, desc) in _MENU.items():
        console.print(f"  [{INDIGO}]{key}[/]  [{DIM}]{desc}[/]")
    console.print()

    try:
        while True:
            try:
                choice = Prompt.ask(
                    f"[{GREEN}]pygs ❯[/]",
                    choices=[*_MENU, "q", "quit", "help"],
                    show_choices=False,
                    default="help",
                )
            except EOFError:
                break

            if choice in ("q", "quit"):
                break
            if choice == "help":
                for key, (_, desc) in _MENU.items():
                    console.print(f"  [{INDIGO}]{key}[/]  [{DIM}]{desc}[/]")
                console.print(f"  [{INDIGO}]q[/]  [{DIM}]Quit[/]")
                continue

            cmd, _ = _MENU[choice]

            if cmd == "status":
                _cmd_status(argparse.Namespace())

            elif cmd == "methods":
                _cmd_methods(argparse.Namespace())

            elif cmd == "infer":
                fpath = Prompt.ask(
                    f"  [{DIM}]Data file (.csv, .tsv, .npy)[/]"
                )
                method = Prompt.ask(f"  [{DIM}]Method[/]", default="lasso")
                out = Prompt.ask(f"  [{DIM}]Output (.npy)[/]", default="")
                _cmd_infer(
                    argparse.Namespace(
                        file=fpath, method=method, output=out or None
                    )
                )

            elif cmd == "bench":
                ns = _configure_synthetic()
                _run_benchmark_live(ns)

            elif cmd == "bench-gs":
                ns = _configure_genespider()
                _run_genespider_live(ns)

            elif cmd == "evaluate":
                pred = Prompt.ask(
                    f"  [{DIM}]Predicted adjacency (.npy, .csv)[/]"
                )
                gold = Prompt.ask(
                    f"  [{DIM}]Gold standard (.npy, .json, .csv)[/]"
                )
                _cmd_evaluate(
                    argparse.Namespace(predicted=pred, gold=gold)
                )

            elif cmd == "plot":
                adj = Prompt.ask(
                    f"  [{DIM}]Adjacency file (.npy, .csv)[/]"
                )
                genes = Prompt.ask(
                    f"  [{DIM}]Gene names file (blank=auto)[/]",
                    default="",
                )
                tfs = Prompt.ask(
                    f"  [{DIM}]TF list file (blank=none)[/]", default=""
                )
                out = Prompt.ask(
                    f"  [{DIM}]Output image[/]", default=""
                )
                _cmd_plot(
                    argparse.Namespace(
                        adjacency=adj,
                        genes=genes or None,
                        tfs=tfs or None,
                        threshold=0.0,
                        output=out or None,
                    )
                )

            elif cmd == "dashboard":
                inp = Prompt.ask(
                    f"  [{DIM}]Input JSON[/]",
                    default="benchmark_results.json",
                )
                out = Prompt.ask(
                    f"  [{DIM}]Output HTML[/]",
                    default="benchmark_dashboard.html",
                )
                _cmd_dashboard(
                    argparse.Namespace(
                        input=inp, output=out, no_open=False
                    )
                )

            elif cmd == "show":
                path = Prompt.ask(f"  [{DIM}]Path to result JSON[/]")
                if path:
                    _cmd_show(argparse.Namespace(file=path))

            console.print()

    except KeyboardInterrupt:
        console.print(f"\n[{DIM}]bye[/]")


# ── CLI entrypoint ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pygs",
        description="pyGS — gene regulatory network inference & benchmarking",
    )
    subs = parser.add_subparsers(dest="command")

    # status
    subs.add_parser("status", help="System status & available methods")

    # methods
    subs.add_parser("methods", help="List all inference methods")

    # infer
    ip = subs.add_parser("infer", help="Infer a network from expression data")
    ip.add_argument("file", help="Input data (.csv, .tsv, .npy)")
    ip.add_argument("-m", "--method", default="lasso", help="Inference method")
    ip.add_argument(
        "-o", "--output", default=None, help="Save adjacency as .npy"
    )

    # bench (synthetic)
    bp = subs.add_parser("bench", help="Synthetic benchmark")
    bp.add_argument("--tier", default="fast")
    bp.add_argument("--n-nodes", type=int, default=50)
    bp.add_argument("--n-samples", type=int, default=200)
    bp.add_argument("--n-datasets", type=int, default=5)
    bp.add_argument("--seed", type=int, default=42)
    bp.add_argument("--timeout", type=int, default=60)
    bp.add_argument("-o", "--output", default="benchmark_results.json")

    # bench-gs (GeneSpider)
    gp = subs.add_parser(
        "bench-gs", help="GeneSpider benchmark (real data)"
    )
    gp.add_argument(
        "--tier", default="fast",
        help="Comma-separated: fast,medium,slow,nestboot",
    )
    gp.add_argument(
        "--sizes", default="N50", help="Comma-separated: N10,N50,N100"
    )
    gp.add_argument("--max-datasets", type=int, default=0, help="0 = all")
    gp.add_argument("--timeout", type=int, default=120)
    gp.add_argument(
        "--nestboot", action="store_true",
        help="Wrap methods in NestBoot FDR",
    )
    gp.add_argument("--nest-runs", type=int, default=10)
    gp.add_argument("--boot-runs", type=int, default=10)
    gp.add_argument("--fdr", type=float, default=0.05)
    gp.add_argument("--seed", type=int, default=42)
    gp.add_argument("-o", "--output", default="benchmark_genespider.json")

    # nestboot
    nb = subs.add_parser(
        "nestboot", help="NestBoot FDR analysis on expression data"
    )
    nb.add_argument(
        "file", help="Expression data (.csv, .tsv, .h5ad, .npy)"
    )
    nb.add_argument(
        "-m", "--method", default="lasso", help="Inference method"
    )
    nb.add_argument(
        "--nest-runs", type=int, default=10, help="Outer bootstrap runs"
    )
    nb.add_argument(
        "--boot-runs", type=int, default=10, help="Inner bootstrap runs"
    )
    nb.add_argument("--fdr", type=float, default=0.05, help="FDR threshold")
    nb.add_argument("--seed", type=int, default=42)
    nb.add_argument("-o", "--output", default="nestboot.npy")

    # evaluate
    ep = subs.add_parser(
        "evaluate", help="Evaluate predicted GRN vs gold standard"
    )
    ep.add_argument("predicted", help="Predicted adjacency (.npy, .csv)")
    ep.add_argument(
        "--gold", required=True, help="Gold standard (.npy, .json, .csv)"
    )

    # plot
    pp = subs.add_parser("plot", help="Plot a gene regulatory network")
    pp.add_argument("adjacency", help="Adjacency matrix (.npy, .csv)")
    pp.add_argument(
        "--genes", default=None, help="Gene names file (one per line)"
    )
    pp.add_argument(
        "--tfs", default=None, help="TF list file (one per line)"
    )
    pp.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Min edge weight to display",
    )
    pp.add_argument(
        "-o", "--output", default=None, help="Output image path"
    )

    # show
    sp = subs.add_parser("show", help="Render a previous result JSON")
    sp.add_argument("file", help="Path to result JSON")

    # dashboard
    dp = subs.add_parser(
        "dashboard", help="Generate interactive HTML dashboard"
    )
    dp.add_argument("-i", "--input", default="benchmark_results.json")
    dp.add_argument("-o", "--output", default="benchmark_dashboard.html")
    dp.add_argument("--no-open", action="store_true")

    args = parser.parse_args()

    if not args.command:
        _interactive()
        return

    _print_banner()

    if args.command == "status":
        _cmd_status(args)
    elif args.command == "methods":
        _cmd_methods(args)
    elif args.command == "infer":
        _cmd_infer(args)
    elif args.command == "bench":
        _run_benchmark_live(args)
    elif args.command == "bench-gs":
        _run_genespider_live(args)
    elif args.command == "nestboot":
        _cmd_nestboot(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "plot":
        _cmd_plot(args)
    elif args.command == "show":
        _cmd_show(args)
    elif args.command == "dashboard":
        _cmd_dashboard(args)


if __name__ == "__main__":
    main()
