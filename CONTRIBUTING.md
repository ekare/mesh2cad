# Contributing

Thanks for helping turn faceted meshes into editable CAD.

1. Use Python 3.10-3.13 and create a virtual environment.
2. Install with `python -m pip install -e ".[dev]"`.
3. Run `ruff check .` and `pytest` before opening a pull request.
4. Add a focused synthetic fixture for every new topology or surface type.

The first milestone intentionally supports only axis-aligned blocks with through-holes. New
recognizers should fail with a useful error when their geometric assumptions are not met.

