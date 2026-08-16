import numpy as np
import pytest
import trimesh

from mesh2cad.metrics import geometric_fidelity, preview_deviations


def test_identical_surface_has_complete_bidirectional_fidelity() -> None:
    source = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    positions = np.asarray(source.triangles, dtype=np.float64)

    result = geometric_fidelity(source, positions, tolerance=1e-6, sample_count=1_000)

    assert result.geometric_fidelity == pytest.approx(1.0)
    assert result.rms_error < 1e-12
    assert result.p99_error < 1e-12
    assert result.surface_area_relative_error < 1e-12
    assert result.volume_relative_error < 1e-12
    assert result.normal_p95_error_degrees < 1e-6
    assert np.max(preview_deviations(source, positions)) < 1e-12


def test_bidirectional_fidelity_detects_a_separated_reconstruction() -> None:
    source = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    positions = np.asarray(source.triangles, dtype=np.float64) + [2.0, 0.0, 0.0]

    result = geometric_fidelity(source, positions, tolerance=0.01, sample_count=1_000)

    assert result.geometric_fidelity == pytest.approx(0.0)
    assert result.mean_error >= 1.0
    assert result.source_to_step_coverage == 0.0
    assert result.step_to_source_coverage == 0.0
