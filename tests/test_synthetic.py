import numpy as np

from mesh2cad.synthetic import cube_with_hole


def test_cube_with_hole_is_watertight_and_has_expected_bounds():
    mesh = cube_with_hole(size=(10.0, 8.0, 6.0), radius=2.0, sections_per_side=8)

    assert mesh.is_watertight
    assert mesh.is_volume
    np.testing.assert_allclose(mesh.bounds, [[-5.0, -4.0, -3.0], [5.0, 4.0, 3.0]])
