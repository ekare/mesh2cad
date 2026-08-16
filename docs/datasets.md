# Dataset strategy and provenance

mesh2cad ships the reviewed inference checkpoint, but not training data. Dataset downloads live
outside the repository, and every public checkpoint requires its own model card, provenance
manifest, evaluation report, and license review.

## Corpus roles

| Dataset | Scale and useful payload | License gate | Intended role |
|---|---|---|---|
| ABC v00 | 1,000,000 CAD models in 100 STEP shards | Per-model copyright; follow the ABC and Onshape terms | Broad BRep pre-training after validation and deduplication |
| AutoMate | 447,442 STEP files in the published archive | CC0 on Dryad; retain citation and provenance | Real mechanical-part diversity and assembly-derived geometry |
| Zero-to-CAD 100K | 100,516 CadQuery programs with STEP, STL, renders, and fixed splits | Apache-2.0 | Permissive synthetic training and cross-dataset evaluation |
| CadQuarry 100K | Execution-validated parametric programs; STEP/STL are reproducible from seeds | Generated data CC0-1.0; generator Apache-2.0 | Permissive analytic/mechanical coverage and controlled augmentation |
| AIAx CAD | 3,128 labeled mechanical STL files in 12 classes; 1.1 GB archive | CC BY 4.0 | Mesh-only robustness holdout; never treated as analytic ground truth |
| Fusion 360 Gallery | Reconstruction and face-segmentation subsets | Non-commercial research only | Isolated research benchmark; exclude from public unrestricted weights |

DeepCAD, Text2CAD, and many later CAD-sequence datasets are derived from ABC. They can provide
extra annotations, but they do not count as independent geometry and must share an ABC provenance
group during splitting and reporting.

## Recommended public-training mix

1. Start with CadQuarry and Zero-to-CAD because their geometry, programs, and licenses are suitable
   for a reproducible public baseline.
2. Add validated AutoMate parts to reduce the synthetic-to-real domain gap.
3. Add ABC only through a separate provenance group and publish the exact accepted/rejected file
   manifest. Do not imply that the mesh2cad license covers ABC or a derived checkpoint.
4. Keep AIAx and other mesh-only collections outside the supervised analytic-surface targets. Use
   them for corruption, scale, topology, and inference-failure tests.
5. Keep research-only datasets in a physically separate directory and never merge them into a
   checkpoint intended for unrestricted release.

Dataset size is not a quality metric. Sampling should balance source, surface type, BRep face
count, triangle count, operation family, and geometric complexity. Report per-source metrics as
well as the aggregate score.

## Quality gate

Every STEP model must pass the same deterministic gate before tessellation:

- OpenCascade can read the file and finds at least one finite, non-empty shape;
- the shape passes BRep validity checks, or an explicitly recorded repair produces a valid copy;
- units and bounding-box scale are finite and plausible;
- all triangulated vertices, normals, areas, and labels are finite;
- triangle-to-source-face mapping covers the complete retained mesh;
- degenerate triangles and zero-area faces are rejected or recorded, never silently relabeled;
- a canonical geometry fingerprint prevents duplicate leakage across sources and splits;
- train, validation, and test assignment occurs at provenance-group level before augmentation.

There is no triangle-count cutoff. Large graphs use lossless core-and-halo partitions so every
triangle and adjacency target contributes exactly once. Disk and batch limits may stage work, but
they must not alter the geometry of an accepted model.

## Storage planning

Storage depends far more on tessellation density than on model count. Measurements from the
high-resolution, uncapped pilot configuration (`relative_deflection=0.001`,
`angular_deflection=0.1`) project roughly 4.6 GiB for 100,000 CadQuarry samples, 14.8 GiB for
100,000 Zero-to-CAD samples, and 292 GiB for 100,000 ABC samples. Treat these as planning
estimates, not quotas: the geometry distribution has a long tail.

One terabyte of verified free space is therefore a comfortable target for the current baseline
when ABC is processed in source batches and completed batches are compacted or archived. Multiple
terabytes are only relevant to an optional full-million archive that retains source CAD, every
high-resolution graph, multiple checkpoints, and scratch copies simultaneously; mesh2cad does not
require that layout.

Do not use a WSL or VM virtual disk unless its physical backing store and growth limits have been
verified. Point dataset roots at an explicit host drive and preserve a free-space safety reserve.

## Citations and upstream sources

- [ABC Dataset](https://deep-geometry.github.io/abc-dataset/)
- [AutoMate on Dryad](https://doi.org/10.5061/dryad.2547d7wvw)
- [Zero-to-CAD 100K](https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-100k)
- [CadQuarry](https://github.com/jacobjennings/CadQuarry)
- [AIAx CAD Dataset](https://doi.org/10.35097/420)
- [Fusion 360 Gallery Dataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset)

Record the upstream revision, download URL, byte size, published checksum when available, local
checksum, acceptance result, and processing configuration for every acquired artifact.

For AutoMate, `mesh2cad-automate` implements this acquisition gate and resumable extraction. The
official archive contains 447,442 STEP files (74,088,040,746 uncompressed bytes); source files and
generated graph samples remain outside the repository.
