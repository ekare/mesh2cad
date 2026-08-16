# Installation

## Supported environments

- Python 3.10 through 3.13
- Windows x64 and common x86-64 Linux distributions
- A browser with WebGL 2 for the interactive preview
- CUDA is optional and used only by the learned segmentation track

## Windows x64 application

Choose one release, verify its SHA-256 file, and extract it to a normal user-writable directory:

- `mesh2cad-0.1.0-windows-x64.zip` is the smaller deterministic baseline.
- `mesh2cad-0.1.0-windows-x64-neural.zip` includes CPU PyTorch, the qualified v0.1 inference
  checkpoint, its exact split/provenance/model card/evaluation records, and defaults to direct
  neural proposals.

Run the executable from the corresponding extracted folder, for example:

```powershell
.\mesh2cad-0.1.0-windows-x64\mesh2cad-web.exe
```

The app opens `http://127.0.0.1:7860`. If that port is occupied, it chooses a free local port. No
administrator permission or Python installation is required. Keep the complete extracted folder;
the executable depends on the adjacent libraries and `licenses` directory.

The same folder contains `mesh2cad.exe` for direct STL-to-STEP conversion and
`mesh2cad-repair.exe` for measured conservative mesh cleanup. The operations dashboard is served
at `/dashboard`; loss/GPU values appear when the packaged web process is pointed at active local
telemetry files.

## Python install

```bash
git clone YOUR_REPOSITORY_URL/mesh2cad.git
cd mesh2cad
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Launch the browser app with `mesh2cad-web`, or convert directly with
`mesh2cad input.stl -o output.step`.

For a separately downloaded reviewed neural checkpoint, install the inference dependency set:

```bash
python -m pip install -r requirements-inference.txt
mesh2cad-web --checkpoint mesh2cad-automate-v0.1.pt --segmentation-engine neural
```

Install `requirements-ml.txt` only for dataset preparation and training. Choose the appropriate
PyTorch wheel/index for CUDA training rather than assuming the generic requirement selected one.

## Development install

```bash
python -m pip install -e ".[web,dev]"
ruff check .
pytest
```

The browser workbench is dependency-free HTML, CSS, and JavaScript embedded directly in the Python
package. It needs no Node.js toolchain or frontend build step. The application is local-only
because CAD conversion requires the native OpenCascade backend; it is not published as a
non-functional cloud converter.

## Troubleshooting

- If `OCP` cannot be imported, confirm that Python is 3.10-3.13 and reinstall `cadquery-ocp`.
- If the preview is blank, enable hardware acceleration/WebGL in the browser. Conversion still
  happens in the Python backend.
- If reconstruction returns HTTP 422, the mesh is outside the current milestone or is not a clean
  supported STL. Run the CLI to see the same detailed error in a terminal.
- The web app imposes no upload cap by default. Set `MESH2CAD_MAX_UPLOAD_BYTES` to a positive byte
  count when operating in a managed environment.
