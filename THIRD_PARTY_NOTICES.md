# Third-party notices

mesh2cad source code is distributed under the MIT license in `LICENSE`. It uses independent
third-party packages that retain their own copyrights and licenses. Installing mesh2cad or using a
binary release does not relicense those components.

The exact license texts for components bundled in a Windows release are placed in the release
package under `licenses/`. The build script collects the license files from the exact installed
wheel versions rather than substituting a summary for their terms.

## Primary runtime components

| Component | Role | Upstream license |
|---|---|---|
| Python | Runtime in the Windows package | Python Software Foundation License |
| NumPy | Arrays and numerical kernels | BSD-3-Clause plus wheel notices |
| SciPy | Least-squares and numerical routines | BSD-3-Clause plus wheel notices |
| trimesh | STL loading and mesh processing | MIT |
| cadquery-ocp / OCP bindings | Python binding distribution | Apache-2.0 |
| Open CASCADE Technology | CAD/BRep kernel inside OCP | LGPL-2.1 with OCCT special exception |
| FastAPI | Local HTTP application | MIT |
| Starlette | ASGI application layer | BSD-3-Clause |
| Uvicorn | Local ASGI server | BSD-3-Clause |
| python-multipart | STL form upload parser | Apache-2.0 |
| Pydantic and supporting packages | API validation | MIT and applicable package licenses |

PyTorch is not bundled in the deterministic Windows application. It is bundled only in the
separately labelled neural Windows package, where its exact license and third-party notices are
collected from the installed wheel:

- PyTorch: BSD-3-Clause and included third-party notices;

Other optional tools are not bundled in either Windows application:

- Apache Arrow / PyArrow: Apache-2.0;
- Hugging Face Hub client: Apache-2.0;
- py7zr: LGPL-2.1-or-later and included component notices;
- tqdm: MPL-2.0 AND MIT;
- PyInstaller: GPL-2.0-or-later with its bootloader exception. PyInstaller is a build tool; its
  exception permits distribution of generated executables under the application's license.

## Optional datasets

ABC is optional training input, not project code and not included in the repository or releases.
The [official ABC page](https://deep-geometry.github.io/abc-dataset/) states that copyright in the
CAD models is owned by their creators, refers users to the Onshape terms for licensing details,
requests citation of the ABC paper, and provides the dataset without warranty. Downloading or
using ABC is the user's separate decision and responsibility. mesh2cad does not grant rights to
ABC data or to derived model checkpoints.

Other documented datasets are also optional inputs and remain under their upstream terms:

- Zero-to-CAD 100K is published under Apache-2.0. Preserve its license, revision, and official
  train/validation/test provenance when creating a derived corpus.
- CadQuarry's generated dataset is published under CC0-1.0; its generator is Apache-2.0. Record
  both the data snapshot and generator revision.
- AutoMate is published through Dryad under CC0. Preserve its DOI and citation even though
  attribution is not a CC0 condition.
- AIAx CAD is CC BY 4.0. Attribution is required in evaluation reports and redistributed derived
  artifacts. It contains mesh labels, not analytic BRep-face labels.
- Fusion 360 Gallery is limited to non-commercial research use by its upstream terms. Keep it out
  of unrestricted public checkpoints and release packages.

These summaries are routing guidance, not substitutes for the upstream license texts. A public
checkpoint needs a separate model card and license review based on the exact training manifest.

Project maintainers should run the release license collector after every dependency change and
review the resulting inventory before publishing a binary.
