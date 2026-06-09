# Running SCENIC+ — Practical Guide

This document covers how to run SCENIC+ both directly through its own CLI/Snakemake interface and via the pyGS wrapper (`src/methods/scenicplus.py`).

---

## 1. Environment Setup

```bash
conda create --name scenicplus python=3.11
conda activate scenicplus

# Core dependencies
git clone https://github.com/aertslab/pycisTopic.git
cd pycisTopic && pip install -e . && cd ..

pip install scanpy

git clone https://github.com/aertslab/scenicplus
cd scenicplus && git checkout development && pip install . && cd ..
```

The pre-built environment on this system lives at:
```
/scratch/dmorgan/scenicplus/scenicplus_env/
```

Activate it with:
```bash
source /scratch/dmorgan/scenicplus/scenicplus_env/bin/activate
```

---

## 2. Required Input Files

| File | Description | Example Location |
|------|-------------|-----------------|
| **cisTopic object** (`.pkl`) | Pre-computed topic model from scATAC-seq | `/scratch/dmorgan/scenicplus/scenicplus/data/outs/cistopic_obj.pkl` |
| **Gene expression** (`.h5ad`) | AnnData with scRNA-seq counts | `/scratch/dmorgan/scenicplus/scenicplus/data/adata.h5ad` or `/scratch/dmorgan/scenicplus/data/adata.h5ad` |
| **Region sets folder** | Genomic regions from scATAC-seq | `/scratch/dmorgan/scenicplus/scenicplus/data/outs/region_sets/` |
| **Rankings database** (`.feather`) | cisTarget motif rankings | `/scratch/dmorgan/scenicplus/data/hg38_screen_v10_clust.regions_vs_motifs.rankings.feather` |
| **Scores database** (`.feather`) | DEM motif scores | `/scratch/dmorgan/scenicplus/data/hg38_screen_v10_clust.regions_vs_motifs.scores.feather` |
| **Motif annotations** (`.tbl`) | Motif-to-TF mapping | `/scratch/dmorgan/scenicplus/data/Motifsv10nrclustnr.tbl` |
| **TF list** (optional, `.txt`) | Transcription factor gene names | `/scratch/dmorgan/scenicplus/resources/allTFs_hg38.txt` |

### Downloading the 10X PBMC demo data

```bash
wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.tar.gz
tar -xzf pbmc_granulocyte_sorted_3k_filtered_feature_bc_matrix.tar.gz
```

---

## 3. Running via Snakemake (native SCENIC+ pipeline)

### Config file

The config lives at:
```
/scratch/dmorgan/scenicplus/src/scenicplus/snakemake/config/config.yaml
```

Key sections to edit:
- `input_data` — absolute paths to all input files
- `params_general.output_dir` — where results go (uses `{run_id}` template)
- `params_general.n_cpu` — parallelism (default 40)

### Running the full pipeline

```bash
cd /scratch/dmorgan/scenicplus/src/scenicplus/snakemake/

# Dry run to check DAG
snakemake --snakefile Snakefile --configfile config/config.yaml \
  --config run_id=1 --cores 40 -n

# Full run
snakemake --snakefile Snakefile --configfile config/config.yaml \
  --config run_id=1 --cores 40
```

### Running individual steps

```bash
# Step 1: Prepare multiome data
snakemake --forcerun prepare_GEX_ACC --cores 40

# Step 2: Motif enrichment
snakemake --forcerun motif_enrichment_cistarget --cores 40
snakemake --forcerun motif_enrichment_dem --cores 40

# Step 3: Prepare motif enrichment results
snakemake --forcerun prepare_menr --cores 40

# Step 4: TF-to-gene and region-to-gene inference
snakemake --forcerun tf_to_gene --cores 40
snakemake --forcerun region_to_gene --cores 40

# Step 5: Build eRegulons
snakemake --forcerun eGRN_direct --cores 40
snakemake --forcerun eGRN_extended --cores 40
```

### Running via the `scenicplus` CLI

```bash
# Prepare data
scenicplus prepare_data prepare_GEX_ACC \
  --cisTopic_obj_fname /path/to/cistopic_obj.pkl \
  --GEX_anndata_fname /path/to/GEX_anndata.h5ad

# Motif enrichment
scenicplus grn_inference motif_enrichment_cistarget \
  --region_set_folder /path/to/region_sets \
  --cistarget_db_fname /path/to/rankings.feather

scenicplus grn_inference motif_enrichment_dem \
  --region_set_folder /path/to/region_sets \
  --dem_db_fname /path/to/scores.feather

# TF-to-gene
scenicplus grn_inference TF_to_gene \
  --multiome_mudata_fname results/run_1/ACC_GEX.h5mu

# Region-to-gene
scenicplus grn_inference region_to_gene \
  --multiome_mudata_fname results/run_1/ACC_GEX.h5mu \
  --search_space_fname results/run_1/search_space.tsv

# eGRN construction
scenicplus grn_inference eGRN \
  --TF_to_gene_adj_fname results/run_1/tf_to_gene_adj.tsv \
  --region_to_gene_adj_fname results/run_1/region_to_gene_adj.tsv \
  --cistromes_fname results/run_1/cistromes_direct.h5ad \
  --ranking_db_fname /path/to/rankings.feather \
  --eRegulon_out_fname results/run_1/eRegulon_direct.tsv
```

---

## 4. Running via pyGS Wrapper

The pyGS wrapper at `src/methods/scenicplus.py` provides two modes:

### Direct mode (recommended, no dask/arboreto needed)

```python
from methods.scenicplus import SCENICPLUS
from datastruct.Dataset import Dataset

dataset = Dataset(...)  # your expression data

# Lightweight correlation+GBM approach
adj_matrix, _ = SCENICPLUS(
    dataset=dataset,
    cisTopic_obj_fname="/scratch/dmorgan/scenicplus/scenicplus/data/outs/cistopic_obj.pkl",
    n_cpu=20,
    use_snakemake=False,   # direct Python API
    use_arboreto=False,    # lightweight mode
)
```

### Arboreto/GRNBoost2 mode (more accurate, requires dask)

```python
adj_matrix, network_df = SCENICPLUS(
    dataset=dataset,
    cisTopic_obj_fname="/path/to/cistopic_obj.pkl",
    n_cpu=20,
    use_snakemake=False,
    use_arboreto=True,     # uses vendored arboreto
)
```

### Snakemake mode (full SCENIC+ pipeline via pyGS)

```python
adj_matrix, _ = SCENICPLUS(
    dataset=dataset,
    scenic_workflow_dir="/scratch/dmorgan/pyNB/src/methods/scenic_workflow/",
    cisTopic_obj_fname="/path/to/cistopic_obj.pkl",
    n_cpu=20,
    use_snakemake=True,
)
```

### With NestBoot FDR control

```python
robust_adj = SCENICPLUS(
    dataset=dataset,
    cisTopic_obj_fname="/path/to/cistopic_obj.pkl",
    nested_boot=True,
    nest_runs=50,
    boot_runs=50,
    fdr=0.05,
    n_cpu=20,
)
```

### Via pyGS CLI

```bash
pygs infer data.h5ad -m scenicplus -o adj.npy
pygs nestboot data.h5ad -m scenicplus --fdr 0.05
```

---

## 5. Key File Locations on This System

```
/scratch/dmorgan/scenicplus/
├── scenicplus_env/              # Pre-built Python 3.11 venv
├── src/scenicplus/
│   └── snakemake/
│       ├── Snakefile            # Main Snakemake workflow
│       └── config/config.yaml   # Pipeline configuration
├── data/
│   ├── adata.h5ad                         # Gene expression
│   ├── hg38_screen_v10_clust...rankings.feather  # cisTarget DB
│   ├── hg38_screen_v10_clust...scores.feather    # DEM DB
│   └── Motifsv10nrclustnr.tbl            # Motif annotations
├── resources/
│   ├── allTFs_hg38.txt          # Human TF list
│   ├── allTFs_mm.txt            # Mouse TF list
│   └── allTFs_dmel.txt          # Drosophila TF list
├── scplus_pipeline/Snakemake/workflow/Snakefile  # Alternative pipeline entry
└── scenicplus/data/outs/
    ├── cistopic_obj.pkl         # Pre-computed cisTopic object
    └── region_sets/             # Region sets from ATAC

/scratch/dmorgan/pyNB/
├── src/methods/
│   ├── scenicplus.py            # pyGS wrapper (direct/snakemake)
│   └── scenic_workflow/
│       ├── Snakefile            # (if present, linked workflow)
│       ├── config/config.yaml
│       └── results/run_1/       # Previous run outputs
└── docs/
    └── SCENIC+ chapter.md      # Extended documentation
```

---

## 6. Output Files

After a successful run, the output directory contains:

| File | Description |
|------|-------------|
| `ACC_GEX.h5mu` | Combined multiome MuData |
| `dem_results.hdf5` / `.html` | DEM motif enrichment |
| `ctx_results.hdf5` / `.html` | cisTarget motif enrichment |
| `cistromes_direct.h5ad` | Direct cistromes |
| `cistromes_extended.h5ad` | Extended cistromes |
| `tf_names.txt` | Detected TF names |
| `search_space.tsv` | Genomic search space |
| `tf_to_gene_adj.tsv` | TF→Gene importance |
| `region_to_gene_adj.tsv` | Region→Gene correlations |
| `eRegulon_direct.tsv` | Direct eRegulons |
| `eRegulons_extended.tsv` | Extended eRegulons |
| `AUCell_direct.h5mu` | AUCell scores (direct) |
| `AUCell_extended.h5mu` | AUCell scores (extended) |
| `scplusmdata.h5mu` | Final SCENIC+ MuData |

---

## 7. Troubleshooting

| Issue | Solution |
|-------|----------|
| **Path not found** errors | Use **absolute paths** in config. Relative paths fail in Snakemake subprocesses. |
| **Memory errors** during motif enrichment or GBM | Reduce `n_cpu` to lower per-core memory, or run on a high-memory node. |
| **Database version mismatch** | Ensure genome annotation (hg38) matches the motif database version (v10nr_clust). |
| **Empty eRegulons** | Lower `rho_threshold` (default 0.05) or widen `search_space` settings. |
| **arboreto/dask conflicts** | Use `use_arboreto=False` in the pyGS wrapper for dask-free inference. |
| **cisTopic object not found** | Verify the `.pkl` path; pycisTopic must be run first to generate it. |
| **Snakemake not installed** | `pip install snakemake` within the scenicplus environment. |
| **`scenicplus` CLI not found** | Ensure the scenicplus package is installed: `pip install -e /scratch/dmorgan/scenicplus/src/` |

---

## 8. Quick-Start Recipe (10X PBMC Demo)

```bash
# 1. Activate environment
source /scratch/dmorgan/scenicplus/scenicplus_env/bin/activate

# 2. Run full pipeline
cd /scratch/dmorgan/scenicplus/src/scenicplus/snakemake/
snakemake --snakefile Snakefile \
  --configfile config/config.yaml \
  --config run_id=1 \
  --cores 40

# 3. Check results
ls results/run_1/eRegulon_direct.tsv
ls results/run_1/scplusmdata.h5mu
```

---

## References

- [SCENIC+ GitHub](https://github.com/aertslab/scenicplus)
- [pycisTopic GitHub](https://github.com/aertslab/pycisTopic)
- [pySCENIC GitHub](https://github.com/aertslab/pySCENIC)
- Extended documentation: `docs/SCENIC+ chapter.md`
