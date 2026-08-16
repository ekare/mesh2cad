# mesh2cad

Local-first STL-to-STEP reconstruction for mechanical parts.

mesh2cad detects analytic regions in a triangle mesh, fits their geometric parameters,
recognizes a constructive-solid-geometry graph, and asks OpenCascade to build and validate a
real BRep solid. It provides both an auditable deterministic baseline and a reviewed neural-first
surface proposer, with deterministic geometry verification and measured fallbacks around both.

> [!WARNING]
> This is experimental alpha software. It is not a general STL-to-STEP converter and its output
> must be independently inspected. Read [DISCLAIMER.md](DISCLAIMER.md) before using an output in
> design, manufacturing, regulated, safety-critical, or production work.

## Local web app

Install the application and launch the private workbench:

```bash
python -m pip install -r requirements.txt
mesh2cad-web
```

The launcher binds only to `127.0.0.1`, opens a browser, previews the STL locally, converts it,
reports bidirectional fidelity and CAD-quality metrics, provides a deviation heatmap, and offers
the STEP file for download. Files are processed in a temporary local directory, expire after one
hour, and are not transmitted or used for telemetry.

### Live operations dashboard

The web process also serves a read-only training dashboard at
`http://127.0.0.1:7860/dashboard`. It receives preparation and training snapshots over an
asynchronous server-sent event stream and shows the active pipeline phase, recent loss, epoch and
batch progress, validation metrics, dataset throughput, storage, and NVIDIA GPU/VRAM pressure.

Point the web process and the long-running jobs at the same local telemetry directory:

```bash
export MESH2CAD_DASHBOARD_STATE=/data/mesh2cad/telemetry
export MESH2CAD_GPU_UUID=GPU-your-training-device-uuid

mesh2cad-web --no-browser
mesh2cad-dataset step /datasets/step /datasets/mesh2cad-npz \
  --telemetry-path "$MESH2CAD_DASHBOARD_STATE/preparation.json"
mesh2cad-train /datasets/mesh2cad-npz -o checkpoints/model.pt \
  --device cuda --telemetry-path "$MESH2CAD_DASHBOARD_STATE/training.json"
```

Telemetry files are atomic local JSON snapshots. Dashboard failure cannot stop a preparation or
training run, and the application sends no operational data to a cloud service.

Pass `--checkpoint model.pt --segmentation-engine neural` to the web launcher when a reviewed
checkpoint should drive every request directly.

The repository includes the latest reviewed checkpoint under
`src/mesh2cad/models/mesh2cad-automate-v0.1.pt`. Install `requirements-inference.txt` to enable
PyTorch inference; the CLI and web app discover the bundled checkpoint automatically.

## Command line

```bash
mesh2cad input.stl -o output.step
```

With a trained checkpoint, `auto` compares verified deterministic and neural proposals. To send
the mesh directly through the learned surface proposer without the deterministic selector:

```bash
mesh2cad input.stl -o output.step --checkpoint model.pt --segmentation-engine neural
```

The command also creates `output.metrics.json`. To choose another report path:

```bash
mesh2cad input.stl -o output.step --report report.json
```

## Current reconstruction scope

Milestone 1 analytically reconstructs clean, watertight rectangular blocks with one or more
cylindrical through-holes aligned to a box axis. The part may be arbitrarily rotated and
translated in the STL coordinate system. It performs:

```text
STL -> normals + triangle adjacency -> region growing -> plane/cylinder fits
    -> explicit CSG graph -> exact OpenCascade booleans -> validated BRep -> STEP + metrics
```

For other meshes, the current fallback exports a faithful mesh-derived BRep (a solid when the input
is watertight, otherwise an open STEP shell) instead of inventing an incorrect analytic solid.
That fallback preserves geometry but is not a clean, feature-editable CAD reconstruction.
Arbitrary trim recovery, blends, pockets, cones, spheres, freeform patches, and general analytic
topology remain future milestones.

Large neural graphs are divided into overlapping core-and-halo partitions. Every triangle and
every adjacency target is counted exactly once, while halo triangles preserve local context. The
future general reconstructor can reduce recognized pieces into the same explicit CSG graph and
evaluate exact fuse/cut/common operations at the end. There is no default triangle-count cap in
dataset preparation or inference.

## Synthetic demo

```bash
mesh2cad-demo demo-output
```

This generates a faceted `cube_with_hole.stl`, reconstructs an analytic
`cube_with_hole.step`, and writes `cube_with_hole.metrics.json`. The tracked example records six
planes, one true cylinder, 100% assigned area, and a valid one-solid BRep. See
[docs/validated-results.md](docs/validated-results.md).

## Metrics

The JSON report contains errors in input model units:

- `analytic_surface_rmse`: area-weighted triangle-vertex distance to the fitted surfaces;
- `geometric_fidelity`: the mean of STL-to-STEP and STEP-to-STL tolerance coverage;
- bidirectional mean, RMS, P95, P99, and maximum surface deviation;
- surface-area and volume relative error plus normal-angle median and P95;
- fitted surface coverage, parameters, BRep validity, face count, and solid count;
- a separate `cad_quality` record identifying analytic CSG versus mesh-derived BRep and reporting
  face reduction and analytic coverage.

The default fidelity tolerance is `0.01%` of the source bounding-box diagonal and is always written
to the report. `brep_vertex_rmse` remains a diagnostic only: it can be zero for a mesh-derived BRep
and must not be interpreted as editability or analytic reconstruction quality.

## Mesh quality and repair plugins

Every analysis now reports structural mesh health, boundary and non-manifold edge counts,
component count, triangle quality, and scale-normalized sampling density. A coarse-sampling hint is
reported separately from structural damage; it is not treated as proof that geometry is wrong.

The optional repair command runs an auditable repairer and never overwrites the source:

```bash
mesh2cad-repair damaged.stl -o repaired.stl
```

The built-in `conservative` repairer removes duplicate/degenerate data and fixes winding without
inventing missing surfaces. Learned repairers use the `mesh2cad.repairers` entry-point contract.
The paired-data and evaluation plan is documented in [docs/repair.md](docs/repair.md).

The first learned repair path reuses the reviewed triangle-segmentation checkpoint to rebuild the
supported mechanical part as analytic CSG and retessellate it without a triangle cap. It only
publishes the learned result when structural and bidirectional fidelity gates pass; otherwise it
returns the conservative result with an explicit abstention in the JSON report:

```bash
mesh2cad-repair coarse-part.stl -o repaired.stl \
  --repairer neural-analytic --checkpoint model.pt --device cuda
```

## Python API

```python
from mesh2cad import analyze, convert

analysis = analyze("input.stl")
print(analysis.metrics.analytic_surface_rmse)

result = convert("input.stl", "output.step", report_path="metrics.json")
print(result.topology.valid)
```

## Install and develop

Python 3.10-3.13 is supported. OpenCascade wheel availability is the limiting factor.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[web,dev]"
ruff check .
pytest
```

Detailed setup instructions are in [docs/installation.md](docs/installation.md). The optional
CUDA learning track is documented in [docs/learning.md](docs/learning.md), including the
[multi-source dataset and provenance policy](docs/datasets.md). Release reproduction is documented
in [docs/releasing.md](docs/releasing.md).

`requirements.txt` installs deterministic CLI/web runtime dependencies;
`requirements-inference.txt` adds PyTorch for a reviewed neural checkpoint; and
`requirements-ml.txt` adds the dataset/training toolchain.

## Repository policy

The source repository contains no upstream dataset payload, user STL/STEP files, or generated
training corpus. The reviewed inference checkpoint is bundled as a small model asset with its
release metadata; private recovery checkpoints, temporary uploads, and common CAD output paths
remain ignored by Git. The included mesh example is generated entirely by this project.

## License and data

mesh2cad source code is MIT licensed. Dependencies remain under their own licenses; binary
releases include the applicable license texts. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Optional datasets are not part of mesh2cad and are not relicensed. Copyright in each ABC CAD model
remains with its creator, and users must accept the terms published on the
[official ABC dataset page](https://deep-geometry.github.io/abc-dataset/). Other sources retain
their own terms; see the [dataset strategy](docs/datasets.md). Upstream datasets are never
redistributed by mesh2cad. Any derived model release must pass the documented checkpoint-specific
license, provenance, evaluation, and model-card gates.

## Author

Prepared together with ChatGPT.
