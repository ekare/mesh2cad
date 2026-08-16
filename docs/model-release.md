# Public model release

Training checkpoints are operational recovery artifacts. They contain optimizer/scaler state,
random-number-generator state, exact local sample paths, and the complete train/validation/test
file lists. Never attach a raw training checkpoint to a public release.

After training and held-out evaluation are complete, create a dataset provenance JSON containing
only public facts such as dataset name, upstream revision, source URL, license, published checksum,
accepted/rejected counts, preparation settings, and split policy. Then export an inference-only
checkpoint:

```bash
mesh2cad-automate-provenance datasets/automate/ingest-manifest.json \
  prepared/automate/manifest.json \
  -o release/models/automate-provenance.json
```

The provenance command refuses incomplete preparation by default, aggregates rejection categories,
and strips machine paths and rejected filenames. Its `--allow-partial` switch is for internal
progress audits only; do not publish that record as final provenance.

A deliberately staged model may be released before every upstream STEP file has been prepared,
provided its training checkpoint is complete and the preparation snapshot already contains every
sample used by that model. First export the exact public sample membership, then bind it into the
`staged_training_subset` provenance record:

```bash
mesh2cad-export-split checkpoints/automate-v0.1-16k.pt \
  -o release/models/automate-v0.1-16k-split.json

mesh2cad-automate-provenance datasets/automate/ingest-manifest.json \
  prepared/automate/manifest.json \
  --training-checkpoint checkpoints/automate-v0.1-16k.pt \
  --split-manifest release/models/automate-v0.1-16k-split.json \
  -o release/models/automate-v0.1-16k-provenance.json
```

This mode records the honest partial-preparation snapshot, deterministic selection policy, split
counts, the exact path-free public sample identifiers, and SHA-256 identities for the
train/validation/test sample sets. It refuses an incomplete checkpoint, an undersized preparation
snapshot, overlapping splits, malformed identifiers, or a split file from another checkpoint.

Then export the inference-only checkpoint:

```bash
mesh2cad-export-model checkpoints/automate-full.pt \
  -o release/models/mesh2cad-automate-v0.1.pt \
  --provenance release/models/automate-provenance.json \
  --report release/models/mesh2cad-automate-v0.1.json
```

The exporter refuses incomplete training checkpoints by default and atomically writes only model
weights, architecture, feature/surface schemas, AMP and partition settings, augmentation metadata,
anonymous sample counts, aggregate validation/test metrics, dataset statistics, license metadata,
and public provenance. It strips optimizer state, recovery RNG state, and every local file or
dataset-root path. The external report records SHA-256 hashes for both the private source checkpoint
and the public artifact; publish only the public artifact hash.

Before publishing, run the personal-data scanner on the exact release tree, independently load the
exported checkpoint on CPU and CUDA, and reproduce the held-out evaluation and at least one
end-to-end STL-to-STEP conversion. A model card must state the supported analytic surface classes,
known topology limitations, per-source metrics, failure cases, training-data licenses, and the
project disclaimer.

The path-free reconstruction benchmark compares deterministic, forced-neural, and automatic
selection on the same arbitrarily rotated/coarsened synthetic cases:

```bash
mesh2cad-benchmark --checkpoint release/models/mesh2cad-automate-v0.1.pt \
  --synthetic-cases 24 --device cuda \
  -o release/models/mesh2cad-automate-v0.1-benchmark.json
```

The report stores source/checkpoint hashes and synthetic parameters, never local input paths. Treat
failed neural conversions and neural fidelity regressions as release evidence, not values to omit.

After the reconstruction and repair reports are final, generate a checkpoint-bound model card:

```bash
mesh2cad-model-card release/models/mesh2cad-automate-v0.1.pt \
  --export-report release/models/mesh2cad-automate-v0.1.json \
  --reconstruction-report release/models/mesh2cad-automate-v0.1-benchmark.json \
  --repair-report release/models/mesh2cad-automate-v0.1-repair-benchmark.json \
  -o release/models/MODEL_CARD.md
```

The generator refuses an export report or benchmark produced for a different checkpoint. The card
contains only the public artifact name and hash, aggregate metrics, public dataset facts, stated
limitations, authorship, and the project disclaimer.

Apply the explicit v0.1 release gates before configuring an application to use the checkpoint:

```bash
mesh2cad-qualify-model release/models/mesh2cad-automate-v0.1.pt \
  --reconstruction-report release/models/mesh2cad-automate-v0.1-benchmark.json \
  --repair-report release/models/mesh2cad-automate-v0.1-repair-benchmark.json \
  -o release/models/mesh2cad-automate-v0.1-qualification.json
```

The command verifies checkpoint hashes and requires completed training, at least 50% held-out
plane/cylinder macro IoU, at least 94% boundary F1, at least 24 cases in each end-to-end benchmark,
at least 95% direct-neural success/analytic-CSG/non-regression rates, at least 95% watertight repair
outputs, and no repair made worse beyond the benchmark tolerance. It exits nonzero and records every
failed gate when the model is not ready.
