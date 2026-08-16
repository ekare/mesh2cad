from mesh2cad.cli import main
from mesh2cad.synthetic import write_cube_with_hole


def test_cli_runs_conversion(tmp_path, capsys):
    source = write_cube_with_hole(tmp_path / "part.stl", sections_per_side=12)
    output = tmp_path / "part.step"

    exit_code = main([str(source), "-o", str(output)])

    assert exit_code == 0
    assert output.is_file()
    assert output.with_suffix(".metrics.json").is_file()
    captured = capsys.readouterr()
    assert "6 planes, 1 cylinders" in captured.out
