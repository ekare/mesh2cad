import numpy as np
import pytest

from mesh2cad.mesh import prepare_mesh
from mesh2cad.metrics import analytic_surface_metrics
from mesh2cad.segmentation import segment_surfaces
from mesh2cad.synthetic import cube_with_hole
from mesh2cad.types import CylinderFit


def test_segments_six_planes_and_one_cylinder():
    data = prepare_mesh(cube_with_hole(sections_per_side=12))
    result = segment_surfaces(data)

    assert len(result.plane_regions) == 6
    assert len(result.cylinder_regions) == 1
    assert len(result.unassigned_faces) == 0

    cylinder = result.cylinder_regions[0].fit
    assert isinstance(cylinder, CylinderFit)
    assert cylinder.radius == pytest.approx(2.0, abs=1e-8)
    assert abs(float(np.dot(cylinder.axis, [0.0, 0.0, 1.0]))) == pytest.approx(1.0)

    metrics = analytic_surface_metrics(data, result)
    assert metrics.assigned_area_fraction == pytest.approx(1.0)
    assert metrics.analytic_surface_rmse < 1e-10
