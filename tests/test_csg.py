import numpy as np
import trimesh

from mesh2cad.csg import Boolean, Box, Cylinder, count_operations, evaluate
from mesh2cad.mesh import load_mesh
from mesh2cad.pipeline import convert
from mesh2cad.preview import tessellate_shape
from mesh2cad.segmentation import segment_surfaces
from mesh2cad.synthetic import cube_with_hole, write_cube_with_hole
from mesh2cad.topology import _load_ocp, recognize_csg


def test_recognizer_emits_explicit_boolean_cut(tmp_path):
    source = write_cube_with_hole(tmp_path / "source.stl", sections_per_side=8)
    data = load_mesh(source)
    graph = recognize_csg(data, segment_surfaces(data))

    assert isinstance(graph, Boolean)
    assert isinstance(graph.left, Box)
    assert isinstance(graph.right, Cylinder)
    assert count_operations(graph, "cut") == 1


def test_rotated_translated_part_remains_analytic_csg(tmp_path):
    mesh = cube_with_hole(size=(11.0, 8.0, 5.0), radius=1.7, sections_per_side=14)
    transform = trimesh.transformations.rotation_matrix(
        np.deg2rad(37.0), np.array([1.0, 2.0, -0.5])
    )
    transform[:3, 3] = np.array([13.0, -4.0, 7.5])
    mesh.apply_transform(transform)
    source = tmp_path / "rotated.stl"
    mesh.export(source)

    result = convert(source, tmp_path / "rotated.step")

    assert result.topology is not None
    assert result.topology.recognizer == "verified_oriented_box_with_through_cylinders"
    assert result.topology.valid
    assert result.topology.solid_count == 1
    assert result.topology.cylinders_cut == 1
    assert result.cad_quality()["representation"] == "analytic-csg"
    assert result.metrics.brep_vertex_rmse is not None
    assert result.metrics.brep_vertex_rmse < 1e-5
    assert result.metrics.geometric_fidelity is not None
    assert result.metrics.geometric_fidelity > 0.8


def test_minimally_triangulated_rotated_box_is_analytic(tmp_path):
    mesh = trimesh.creation.box(extents=(9.0, 5.0, 3.0))
    transform = trimesh.transformations.rotation_matrix(
        np.deg2rad(29.0), np.array([-0.4, 1.0, 2.0])
    )
    transform[:3, 3] = np.array([-8.0, 3.5, 2.0])
    mesh.apply_transform(transform)
    source = tmp_path / "minimal-box.stl"
    mesh.export(source)

    result = convert(source, tmp_path / "minimal-box.step")

    assert len(result.segmentation.plane_regions) == 6
    assert len(result.segmentation.unassigned_faces) == 0
    assert result.topology is not None
    assert result.topology.recognizer == "verified_oriented_box_with_through_cylinders"
    assert result.topology.face_count == 6
    assert result.topology.cylinders_cut == 0
    assert result.metrics.geometric_fidelity == 1.0


def test_rotated_box_with_multiple_through_holes_reduces_to_boolean_csg(tmp_path):
    graph = Box(np.array([-6.0, -4.0, -3.0]), np.array([12.0, 8.0, 6.0]))
    for x_position, radius in ((-2.2, 1.0), (2.4, 0.8)):
        graph = Boolean(
            "cut",
            graph,
            Cylinder(
                axis_point=np.array([x_position, 0.0, -10.0]),
                axis=np.array([0.0, 0.0, 1.0]),
                radius=radius,
                height=20.0,
            ),
        )
    preview = tessellate_shape(
        evaluate(graph, _load_ocp()),
        relative_deflection=0.002,
        angular_deflection=0.08,
        max_triangles=200_000,
    )
    triangles = np.asarray(preview.positions, dtype=np.float64).reshape(-1, 3, 3)
    mesh = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(triangles.size // 3).reshape(-1, 3),
        process=True,
    )
    mesh.fix_normals(multibody=True)
    transform = trimesh.transformations.rotation_matrix(
        np.deg2rad(23.0), np.array([1.0, -0.4, 2.0])
    )
    transform[:3, 3] = np.array([3.0, 8.0, -5.0])
    mesh.apply_transform(transform)
    source = tmp_path / "multiple-holes.stl"
    mesh.export(source)

    result = convert(source, tmp_path / "multiple-holes.step")

    assert len(result.segmentation.cylinder_regions) == 2
    assert result.topology is not None
    assert result.topology.recognizer == "verified_oriented_box_with_through_cylinders"
    assert result.topology.face_count == 8
    assert result.topology.cylinders_cut == 2
    assert result.cad_quality()["representation"] == "analytic-csg"
    assert result.metrics.geometric_fidelity == 1.0


def test_uncapped_shape_tessellation_emits_every_triangle():
    shape = evaluate(
        Cylinder(
            axis_point=np.zeros(3),
            axis=np.array([0.0, 0.0, 1.0]),
            radius=2.0,
            height=5.0,
        ),
        _load_ocp(),
    )

    preview = tessellate_shape(
        shape,
        relative_deflection=5e-4,
        angular_deflection=0.05,
        max_triangles=None,
    )

    assert preview.shown_triangles == preview.total_triangles
