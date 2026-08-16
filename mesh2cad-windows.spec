# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_all

ocp_data, ocp_binaries, ocp_hidden = collect_all("OCP")
cq_data, cq_binaries, cq_hidden = collect_all("cadquery_ocp")
model_checkpoint = os.environ.get("MESH2CAD_BUNDLE_CHECKPOINT")
if model_checkpoint:
    model_checkpoint = os.path.abspath(model_checkpoint)
    if not os.path.isfile(model_checkpoint):
        raise ValueError(f"Bundled checkpoint does not exist: {model_checkpoint}")
    torch_data, torch_binaries, torch_hidden = collect_all("torch")
    model_data = [(model_checkpoint, "mesh2cad/models")]
    optional_excludes = []
    package_name = "mesh2cad-0.1.0-windows-x64-neural"
else:
    torch_data, torch_binaries, torch_hidden = [], [], []
    model_data = []
    optional_excludes = ["torch"]
    package_name = "mesh2cad-0.1.0-windows-x64"

common = dict(
    pathex=["src"],
    binaries=ocp_binaries + cq_binaries + torch_binaries,
    datas=ocp_data
    + cq_data
    + torch_data
    + model_data
    + [("src/mesh2cad/web_static", "mesh2cad/web_static")],
    hiddenimports=ocp_hidden + cq_hidden + torch_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=optional_excludes + [
        "vtk",
        "matplotlib",
        "tkinter",
        "PyQt5",
        "PyQt6",
        "IPython",
        "notebook",
        "pandas",
    ],
    noarchive=False,
    optimize=1,
)

web_analysis = Analysis(["src/mesh2cad/webapp.py"], **common)
web_pyz = PYZ(web_analysis.pure)
web_exe = EXE(
    web_pyz,
    web_analysis.scripts,
    [],
    exclude_binaries=True,
    name="mesh2cad-web",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

cli_analysis = Analysis(["src/mesh2cad/cli.py"], **common)
cli_pyz = PYZ(cli_analysis.pure)
cli_exe = EXE(
    cli_pyz,
    cli_analysis.scripts,
    [],
    exclude_binaries=True,
    name="mesh2cad",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

repair_analysis = Analysis(["src/mesh2cad/repair.py"], **common)
repair_pyz = PYZ(repair_analysis.pure)
repair_exe = EXE(
    repair_pyz,
    repair_analysis.scripts,
    [],
    exclude_binaries=True,
    name="mesh2cad-repair",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    web_exe,
    cli_exe,
    repair_exe,
    web_analysis.binaries,
    web_analysis.datas,
    cli_analysis.binaries,
    cli_analysis.datas,
    repair_analysis.binaries,
    repair_analysis.datas,
    strip=False,
    upx=True,
    name=package_name,
)
