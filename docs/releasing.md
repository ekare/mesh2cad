# Releasing

Only release from a clean checkout on Windows x64. Do not place datasets, checkpoints, customer
meshes, personal files, or credentials inside the checkout.

## Local build

```powershell
pwsh -File scripts\build-release.ps1
```

PowerShell 7 is optional. On a stock Windows installation, use Windows PowerShell instead:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build-release.ps1
```

The script verifies the embedded dependency-free web assets, creates an isolated x64 Python
environment, installs only runtime/release dependencies, runs all Python tests, builds an onedir PyInstaller
application, copies project and third-party notices, performs an executable health/conversion
smoke test, verifies the packaged dashboard, exercises the repair executable, scans the package
for obvious personal absolute paths, and creates:

```text
release/mesh2cad-0.1.0-windows-x64.zip
release/mesh2cad-0.1.0-windows-x64.zip.sha256
```

After a reviewed public inference checkpoint and its model card have passed the release gates,
build the separately labelled neural package:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\build-release.ps1 `
  -ModelCheckpoint release\models\mesh2cad-automate-v0.1.pt `
  -ModelCard release\models\MODEL_CARD.md
```

This creates `mesh2cad-0.1.0-windows-x64-neural.zip`. It bundles CPU PyTorch and the reviewed
inference-only model, defaults the CLI, repair command, and web app to direct neural proposals, and
verifies `engine=neural` plus `checkpoint=loaded` through the packaged API before archiving. The
repair path retains its measured abstention gate. The model card, exact public split, provenance,
reconstruction/repair reports, qualification report, and public hashes are included under the
package's `model/` directory. CUDA training dependencies and the private recovery checkpoint are
not bundled.

Use the onedir format because the native OpenCascade and optional PyTorch libraries are large and must remain
replaceable as required by their applicable licenses. Do not convert this release to one-file
without a fresh license and startup review.

## Tag release

1. Review `THIRD_PARTY_NOTICES.md` and the generated `licenses/` inventory.
2. Verify that `git status` contains no generated data or personal artifacts.
3. Run the local release script on a clean Windows x64 machine.
4. Verify the SHA-256 digest on a second machine and exercise STL upload, STEP download, and CLI.
5. Create an annotated `v0.1.0` tag. The GitHub Actions workflow rebuilds and uploads an x64
   artifact for the tag; compare its test log and package inventory before publishing it.

The project does not automatically upload an ABC dataset or neural checkpoint. A future public
model release requires a separate dataset-license, provenance, evaluation, and model-card review.
