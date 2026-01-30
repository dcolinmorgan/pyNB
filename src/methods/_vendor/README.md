# Vendored arboreto with modern dask compatibility

This directory contains a vendored copy of [arboreto](https://github.com/aertslab/arboreto) 
with fixes for modern dask (2024+) compatibility.

## Why vendored?

The original arboreto package hasn't been updated for modern dask versions. Rather than 
forcing users to downgrade their dependencies, we vendor and patch it locally.

## Changes from upstream

- Updated dask imports for compatibility with dask 2024.3+
- Graceful fallback for missing dask.dataframe.utils functions

## License

Original arboreto license applies (BSD 3-Clause).
