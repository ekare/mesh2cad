# Learned segmentation track

The deterministic pipeline remains the default. The optional graph network proposes surface
classes and boundaries; analytic fitting, the CSG graph, OpenCascade booleans, validation, and STEP
export remain deterministic.

## Predictions

For every triangle the network predicts one of `plane`, `cylinder`, `cone`, `sphere`, `torus`,
`bspline`, or `other`. For every triangle adjacency it predicts a face-boundary probability.
Removing predicted boundary edges and taking connected components produces face instances.

## ABC data

ABC provides original CAD models and explicitly parameterized surfaces suitable for automatic
labels. The official dataset consists of one million models in 10,000-model chunks. Download and
verify the first STEP chunk with:

```bash
mesh2cad-abc /datasets/abc --format step --chunk 0 --extract
```

The downloader resumes partial transfers, retries interruptions, checks the official MD5 manifest,
and extracts only after verification. Always use a physical host path with sufficient free space;
do not assume a WSL/VM virtual disk has the same capacity as the visible host drive.

Prepare high-resolution labeled graphs directly from the STEP faces:

```bash
mesh2cad-dataset step /datasets/abc/v00/step /datasets/abc/mesh2cad-npz \
  --relative-deflection 0.001 \
  --angular-deflection 0.1 \
  --workers 4
```

The linear tessellation tolerance is the shape bounding-box diagonal times the relative deflection,
so quality is scale-independent. There is no triangle-count limit by default. An optional
`--max-triangles` switch exists only for explicitly constrained environments. The manifest records
the complete configuration, success/failure details, duration, and triangle-count distribution.

Each compressed NPZ includes vertices, triangles, adjacency, twelve normalized triangle features,
semantic labels, BRep face-instance labels, and adjacency-boundary labels.

ABC data is not covered by the mesh2cad MIT license. Copyright in each CAD model remains with its
creator. Downloading accepts the terms on the official ABC page, and research use should cite the
ABC paper. Do not commit source models, generated NPZ files, or derived checkpoints.

## Additional public corpora

The recommended permissive companion sources are Zero-to-CAD 100K (Apache-2.0), CadQuarry
(CC0-1.0 generated data), and AutoMate (CC0 on Dryad). See [datasets.md](datasets.md) for their
roles, independence caveats, and release gates.

Zero-to-CAD and CadQuarry publish STEP payloads in Parquet files. Materialize them in bounded-memory
batches before using the normal STEP preparation command:

```bash
hf download jacobjennings/cadquarry 100k/corpus-step.parquet \
  --repo-type dataset --local-dir /datasets/cadquarry
hf download ADSKAILab/Zero-To-CAD-100k \
  --repo-type dataset --include "data/**/*.parquet" README.md \
  --local-dir /datasets/zero-to-cad-100k

mesh2cad-parquet-step corpus-step.parquet -o /datasets/cadquarry/step
mesh2cad-parquet-step /datasets/zero-to-cad-100k/data -o /datasets/zero-to-cad-100k/step
mesh2cad-dataset step /datasets/cadquarry/step /datasets/cadquarry/mesh2cad-npz \
  --relative-deflection 0.001 --angular-deflection 0.1 --workers 4
```

AutoMate publishes its STEP payload as one ZIP archive. Verify it against the acquisition checksum
and extract each member atomically so an interruption can be resumed safely:

```bash
mesh2cad-automate /downloads/automate/step.zip /datasets/automate \
  --metadata-root /downloads/automate
mesh2cad-dataset step /datasets/automate/step /datasets/automate/mesh2cad-npz \
  --relative-deflection 0.001 --angular-deflection 0.1 --workers 4
```

The extractor rejects path traversal and unexpected payloads, verifies the official MD5 and
SHA-256 values in one pass, relies on ZIP CRC checks for every STEP member, and keeps progress and
provenance manifests. Existing complete members are skipped; incomplete writes retain a `.part`
suffix and are replaced on the next run.

Before committing storage to the complete corpus, run a deterministic random pilot. This limits
the number of parts, not the geometry: every selected part is tessellated without a triangle cap.

```bash
mesh2cad-dataset step /datasets/automate/step /datasets/automate/pilot-npz \
  --sample-count 512 --seed 17 \
  --relative-deflection 0.001 --angular-deflection 0.1 --workers 4
```

For a long full-corpus preparation, shuffle deterministically so a storage-limited partial run is
still representative, recycle native CAD worker pools, bound pathological files by wall-clock
time, and preserve a disk reserve. None of these options caps a model's triangle count:

```bash
mesh2cad-dataset step /datasets/automate/step /datasets/automate/full-npz \
  --shuffle --seed 17 \
  --relative-deflection 0.001 --angular-deflection 0.1 \
  --workers 2 --pool-batch-size 128 \
  --job-timeout-seconds 900 --min-free-gib 20 \
  --min-available-memory-gib 4
```

The memory reserve is especially useful inside WSL: if a pathological CAD model drives available
memory below the reserve, the active worker window is terminated and recorded in the persistent
failure ledger before the operating system's OOM killer can stop the complete preparation run.

Every graph is written atomically. `manifest.partial.json` is refreshed after each worker-pool
batch, and repeating the command skips completed NPZ files. When the free-space reserve is reached,
the final manifest reports `storage_limit` instead of treating the controlled stop as corruption.
Rejected and hard-timeout paths are appended durably to `failures.jsonl` and skipped on resume;
`--retry-failures` is the explicit opt-in for trying them again after a repair or importer change.

There is no default row or triangle limit. Existing non-empty outputs are skipped, so interrupted
materialization resumes without rewriting completed models. Keep source-specific output roots and
manifests; merge only through a provenance-aware training sampler.

Materialization writes `materialize-partitions.json` and a row-level
`materialize-index.jsonl`. Duplicate identifiers remain assigned to their first partition and are
reported rather than copied across splits. After preparing NPZ files, preserve the audited split:

```bash
mesh2cad-train /datasets/zero-to-cad-100k/mesh2cad-npz \
  --split-manifest /datasets/zero-to-cad-100k/step/materialize-partitions.json \
  -o checkpoints/zero-to-cad.pt --device cuda
```

## CUDA training

Install the CUDA wheel that matches the machine, followed by the project extras. For example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[ml]"
```

Start with a synthetic plumbing test:

```bash
mesh2cad-dataset synthetic /datasets/mesh2cad-smoke --count 32
mesh2cad-train /datasets/mesh2cad-smoke -o checkpoints/smoke.pt \
  --epochs 20 --device cuda
```

Train on ABC without truncating large meshes:

```bash
mesh2cad-train /datasets/abc/mesh2cad-npz -o checkpoints/abc.pt \
  --epochs 80 \
  --device cuda \
  --max-core-faces 50000 \
  --halo-hops 2
```

Large triangle graphs are divided into overlapping core-and-halo partitions. Core nodes are
disjoint and cover the entire mesh; halo nodes retain message-passing context; every node and edge
target contributes to loss and metrics exactly once. Partitioning limits peak GPU memory without
discarding any geometry.

Training uses class-balanced surface loss, boundary positive weighting, CUDA mixed precision,
reproducible train/validation/test splits, and random rigid-rotation augmentation of triangle
centroid and normal features. The augmentation makes surface semantics less dependent on the CAD
authoring orientation and is recorded in every checkpoint. By default, best-checkpoint selection
uses plane/cylinder macro IoU plus boundary F1 because those are the analytic surface classes the
v0.1 BRep builder can consume. `--selection-metric all-surfaces` is available for broader semantic
experiments. Every run also reports per-class IoU and recall, all-surface macro IoU, boundary
precision/recall/F1, and dataset statistics. Use `--no-rotation-augmentation` only for controlled
ablation experiments.

A bounded CPU prefetch queue prepares upcoming graph partitions concurrently with the active CUDA
step so compressed-sample I/O does not unnecessarily idle the GPU. `--prefetch-batches` controls
the RAM/throughput tradeoff and can be set to zero for memory-constrained machines.
`--sample-cache-gib` enables a bounded decoded/partitioned-sample LRU across epochs; its live size,
hits, and misses appear on the dashboard. It never changes split membership or drops a mesh.

Recovery checkpoints are written atomically every 25 optimizer batches and after every epoch. The
previous known-good file is retained beside the output as `<checkpoint>.bak`. After an interruption,
repeat the same command with `--resume`; use `--checkpoint-every-batches N` to change the interval
or `0` to disable intra-epoch snapshots.

Use `--epoch-snapshots DIRECTORY` to retain compact path-free candidate weights and validation
metrics after every epoch. This makes model selection auditable and prevents a later epoch from
erasing an earlier candidate during experiments.

Raw recovery checkpoints contain machine-local paths and optimizer state. Do not publish them.
Follow [Public model release](model-release.md) and use `mesh2cad-export-model` to create the
path-free inference artifact, provenance record, and external SHA-256 report. A staged public model
must also use `mesh2cad-export-split` so its exact AutoMate sample membership remains reproducible
without publishing any machine-local paths or upstream geometry payloads.

## Inference

```bash
mesh2cad-predict input.stl --checkpoint checkpoints/abc.pt -o prediction.npz \
  --max-core-faces 50000 --halo-hops 2
```

Run complete reconstruction with the learned proposer forced on every request:

```bash
mesh2cad input.stl -o output.step --checkpoint checkpoints/abc.pt \
  --segmentation-engine neural --device cuda
```

The default `auto` mode retains a verified deterministic proposal for high-confidence primitive
meshes and otherwise compares the refitted neural and deterministic candidates. Explicit
`neural` mode bypasses that selector while retaining geometric refitting, BRep validation, and the
faithful fallback guard.

The baseline is a compact pure-PyTorch message-passing network. It avoids extension-specific CUDA
packages so the data model and partitioner remain easy to reproduce. A future architecture can
replace the network without changing the NPZ schema or deterministic CAD backend.
