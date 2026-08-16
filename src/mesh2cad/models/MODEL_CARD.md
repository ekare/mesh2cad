# mesh2cad-automate-v0.1

This is the reviewed inference checkpoint bundled with mesh2cad v0.1.

- Intended use: neural-assisted analytic surface proposal for mechanical STL meshes.
- Runtime: install `requirements-inference.txt`; CPU and CUDA inference are supported by the CLI.
- The deterministic verifier remains enabled and may reject or fall back from neural proposals.
- This model is experimental and must not be used without independent inspection.

The checkpoint is distributed under the mesh2cad project license. Training data remains separate
and is not included in this repository; see `docs/datasets.md` and `THIRD_PARTY_NOTICES.md` for
upstream dataset terms and provenance requirements.
