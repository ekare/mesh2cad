# Validated results

These are engineering smoke tests, not claims about arbitrary real-world reconstruction quality.

## Deterministic demo

Input: generated 10 x 8 x 6 block, radius-2 through-hole, 768 STL triangles.

| Check | Result |
|---|---:|
| Planes detected | 6 |
| Cylinders detected | 1 |
| Assigned triangle/area coverage | 100% |
| Analytic surface RMSE | `1.414e-8` model units |
| BRep vertex RMSE | `2.349e-8` model units |
| OpenCascade shape | valid, 1 solid, 7 faces |

Rigid-placement regression tests rotate and translate both a minimally triangulated box and the
block-with-hole fixture around non-cardinal axes. Both remain analytic CSG solids rather than
falling back to mesh-derived BRep. A separate rotated two-hole fixture reduces 1,284 source
triangles to one valid eight-face solid with two explicit boolean cuts and 100% sampled geometric
fidelity to its OpenCascade-generated STL.

## AutoMate v0.1 neural model

The v0.1 model was trained on a deterministic 16,000-model staged subset of the CC0-licensed
AutoMate corpus: 12,800 training, 1,600 validation, and 1,600 held-out test models. Split ownership
is assigned before augmentation. The release includes the exact path-free split manifest, dataset
provenance, model card, qualification report, and SHA-256 manifest. The public inference checkpoint
SHA-256 is `bf22602fcb93db30e5bf8cbe8e1c45744d8484756b9e05cba4a602172c54c57d`.

The selected epoch was 12 of 12. Selection used validation plane/cylinder macro IoU plus boundary
F1; the held-out test set was not used for selection.

| Held-out test check | Result |
|---|---:|
| Plane/cylinder macro IoU | `0.7926` |
| Plane/cylinder accuracy | `0.8900` |
| Boundary F1 | `0.9733` |

The release qualification suite then exercised 24 seeded block-with-hole reconstruction cases in
deterministic, direct-neural, and automatic modes. Direct neural reconstruction succeeded in all
24 cases, produced analytic CSG in all 24 cases, and achieved mean sampled geometric fidelity
`0.9701`. In 23 of 24 cases (`0.9583`), its fidelity was no worse than deterministic reconstruction
within an absolute `0.001` tolerance. Automatic mode also succeeded and produced analytic CSG in
all 24 cases.

A separate 24-case repair suite covered coarse tessellation, duplicated faces, identity inputs,
missing faces, vertex noise, and quantization. All 24 outputs were watertight and none worsened the
measured truth P95 distance. The learned repairer accepted 20 cases and safely abstained on all four
noise cases. Mean truth P95 distance decreased from `5.7034e-5` to `3.8836e-5` model units.

These are bounded v0.1 qualification results on the published test split and seeded synthetic
fixtures. They are not a guarantee for arbitrary meshes, manufacturing use, or safety-critical
work; see the model card and disclaimer before use.

## ABC/CUDA plumbing smoke test

The official ABC STEP shard 0000 was MD5-verified and extracted outside the repository. Twenty
models were tessellated with shape-relative deflection `0.001`, angular deflection `0.1`, four
workers, and no triangle limit. All 20 completed: 1,341,926 total triangles, 500 minimum, and
374,930 maximum per model.

A three-epoch CUDA run on an NVIDIA GTX 1660 Ti processed 1,108 overlapping core-and-halo graph
partitions. The held-out two-model test produced surface macro IoU `0.1433` and boundary F1
`0.7102`. This tiny, class-imbalanced corpus is intentionally insufficient for a useful model; the
figures only verify tessellation, labeling, loss balancing, partitioning, CUDA AMP, checkpointing,
and held-out metric reporting. That early smoke-test checkpoint is not included in the project or
release.

## Multi-source corpus validation

The official CadQuarry 100K STEP Parquet artifact was SHA-256 verified before materialization.
It contains 99,981 valid STEP payloads; 19 upstream rows have no STEP payload and are recorded as
rejections. A strict, uncapped 100-row tessellation smoke test accepted 99 solids and rejected one
non-watertight result. The accepted samples contained 222,612 triangles in total, from 92 to 7,772
triangles per model.

The official Zero-to-CAD 100K Parquet snapshot contains 100,516 rows and 100,000 unique STEP UUIDs.
The 516 repeated UUIDs are entirely test-to-validation overlap; the materializer keeps their first
test ownership and writes a leakage-free split manifest with 81,015 training, 9,218 validation,
and 9,767 test models. A strict, uncapped 100-row tessellation smoke test accepted 95 solids and
rejected three non-watertight models plus two files with no solid. Accepted samples contained
468,188 triangles in total, from 20 to 53,254 triangles per model.

These counts describe upstream and geometric validation, not model-quality claims. Failures remain
in machine-readable manifests rather than being silently repaired or relabeled.

## Mesh-only robustness holdout

The AIAx CAD archive was verified against its published MD5 checksum
`3196e85d1685fd6dead9075eed5f1783`. It contains 3,128 mechanical STL models across 12 labeled
classes. Because AIAx does not provide source BRep faces, it is reserved for inference robustness
and failure reporting and is not used as analytic-surface ground truth.
