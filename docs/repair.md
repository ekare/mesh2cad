# Mesh quality and learned repair

Repair and reconstruction are separate measurable stages. A smoother render is not evidence of a
more faithful mesh, and a watertight mesh can still be a coarse approximation of the intended
surface.

## Current contract

`mesh2cad.quality.assess_mesh` reports:

- watertightness, winding consistency, volume status, and connected components;
- boundary, non-manifold, and degenerate counts;
- fifth-percentile triangle quality;
- median and 95th-percentile edge length normalized by bounding-box diagonal;
- a structural-health score and a separate coarse-sampling hint.

`mesh2cad-repair` loads a repairer through the `mesh2cad.repairers` Python entry-point group. The
built-in `conservative` implementation merges duplicate vertices, removes duplicate and
degenerate faces, removes unreferenced vertices, and reorients winding. It deliberately does not
fill holes, smooth, remesh, or hallucinate geometry. It writes a new STL plus a before/after JSON
report.

The built-in `neural-analytic` repairer is the first learned v0.1 path. It requires an accepted
triangle-segmentation checkpoint and sends the mesh directly through that network. The predicted
surface regions are refitted analytically, reconstructed as supported box/cylinder CSG, and
retessellated from the exact BRep without an output triangle cap. The candidate is accepted only
when it is watertight, consistently wound, does not reduce structural health, and passes sampled
bidirectional P95/maximum deviation plus area/volume gates. Otherwise the plugin preserves the
conservative result and records `abstained: true`, the reason, metrics, and checkpoint SHA-256.

```bash
mesh2cad-repair coarse-part.stl -o repaired.stl \
  --repairer neural-analytic --checkpoint model.pt --device cuda
```

This v0.1 learned path is deliberately scoped to the analytic mechanical family supported by the
topology recognizer. It does not claim universal free-form completion.

Run the paired synthetic gate before accepting a checkpoint for repair. It cycles identity,
coarse tessellation, missing-face, duplicate-face, quantization, and vertex-noise corruptions;
measures every result against an independently tessellated clean reference; and reports abstention,
watertightness, acceptance, and made-worse fractions by corruption family:

```bash
mesh2cad-repair-benchmark --checkpoint model.pt --device cuda --cases 24 \
  -o repair-benchmark.json
```

## Learned repair dataset

A public learned repairer requires paired clean/corrupted geometry. Generate pairs only from CAD
models already assigned to a provenance-level train/validation/test split:

1. Tessellate the source BRep at a high-resolution reference setting and retain exact surface
   samples and normals.
2. Produce independently seeded corruption families: controlled decimation, quantization,
   vertex noise, normal flips, duplicate/degenerate faces, small holes, non-manifold insertions,
   disconnected fragments, and mixed corruptions.
3. Store corruption parameters, the clean geometry fingerprint, units, source license, and split
   ownership beside every pair.
4. Never allow clean variants or derived corruptions of one CAD model to cross split boundaries.
5. Keep mesh-only datasets without clean BRep truth as robustness holdouts, not supervised repair
   targets.

AutoMate, Zero-to-CAD, and CadQuarry can supply clean BRep supervision under their respective
license/provenance gates. AIAx is useful only as a mesh-only robustness holdout.

## Acceptance metrics

A learned repair checkpoint is not accepted on training loss alone. Report per corruption family:

- symmetric surface RMS, P95, P99, and Hausdorff-style maximum deviation to the clean reference;
- normal-angle median/P95 and relative area/volume error;
- watertight, manifold, winding-consistent, connected-component, and self-intersection rates;
- triangle-quality distribution and output-size growth;
- downstream analytic-surface IoU, boundary F1, and final STL-to-STEP fidelity;
- the fraction of inputs made worse, including clean-input identity holdouts.

The plugin must preserve a recoverable input, write its own provenance report, and expose confidence
or abstain when its output fails the structural or fidelity gates.
