# Changelog

All notable changes are documented here. The project follows Semantic Versioning after the first
stable release.

## 0.1.0 - 2026-08-16

- Deterministic plane/cylinder region growing and parameter fitting.
- Orientation-independent rectangular-block recognition with one or more box-axis through-holes.
- Explicit CSG graph evaluated as a validated OpenCascade BRep and exported to STEP.
- Reconstruction error, coverage, and topology metrics.
- Bidirectional surface fidelity, P95/P99/max deviation, area/volume/normal checks, STEP preview,
  and a deviation heatmap.
- Synthetic cube-with-through-hole fixture and tracked demonstration output.
- Local-only drag-and-drop web workbench with WebGL STL preview and STEP download.
- Async local operations dashboard for loss, dataset progress, GPU/VRAM, storage, and active phase.
- Automatic BRep face labels, uncapped lossless graph partitioning, CUDA AMP, rotation
  augmentation, atomic recovery checkpoints, decoded-sample caching, per-epoch candidates, and
  direct neural surface proposals.
- Auditable conservative repair plus a confidence-gated neural analytic repairer with explicit
  abstention and paired clean/corrupted evaluation.
- AutoMate acquisition/preparation support and checkpoint-bound staged provenance, exact path-free
  split export, inference-only model export, model-card generation, and explicit release gates.
- Reproducible deterministic and model-bundled neural Windows x64 onedir builds, exact dependency
  license collection, packaged API/CLI/repair smoke tests, personal-data scanning, and CI.
