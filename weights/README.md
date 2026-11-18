# Included SwaT inference weights

This directory contains optimizer-free checkpoints tracked with Git LFS:

- `original/M1..M3/renderer.pt`: the three completed original-recipe renderers;
- `am-query-transfer/M1..M2/query_transfer.pt`: currently completed AM query
  transfer modules (M3 was not complete when this snapshot was made);
- `swat-code/M1..M3/renderer.pt`: the three completed code-adapted renderers.

The exporter removes Adam optimizer/scheduler/RNG history but preserves every
inference tensor and its original dtype exactly. Each exported file remains
directly loadable by `load_renderer_checkpoint` or `load_query_transfer`.
Run `git lfs pull` after cloning; see `manifest.json` for source paths, sizes,
parameter counts, dropped training-only keys, and SHA-256 checksums.
