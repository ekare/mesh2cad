# Installation

## Supported environments

- Python 3.10 through 3.13
- Common x86-64 Linux distributions and WSL2
- A browser with WebGL 2 for the interactive preview
- CUDA is optional and used only by the learned segmentation track

## Linux / WSL source installation

The supported distribution path is the Python source package. WSL2 Ubuntu 22.04/24.04 is a tested
development environment; native Linux distributions work with Python 3.10–3.13 as well.

```bash
unzip mesh2cad-0.1.0-source.zip
cd mesh2cad-0.1.0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Launch the local web application with `mesh2cad-web --port 7861 --no-browser`, then open
`http://127.0.0.1:7861`. The browser UI exposes separate STL repair/simplification tools and
post-reconstruction STEP simplification. No Node.js toolchain is required.

## Python install

```bash
git clone YOUR_REPOSITORY_URL/mesh2cad.git
cd mesh2cad
python -m venv .venv
```

```bash
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

The reviewed neural checkpoint is bundled in the source tree. Install the inference dependency set
to enable it:

```bash
python -m pip install -r requirements-inference.txt
mesh2cad-web --segmentation-engine neural
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
