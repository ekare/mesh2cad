"""Least-squares fitting of analytic surfaces to triangle regions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from trimesh import Trimesh

from mesh2cad.types import CylinderFit, FloatArray, IntArray, PlaneFit


class FitError(ValueError):
    """Raised when an analytic fit is underconstrained or degenerate."""


def _region_points(mesh: Trimesh, face_indices: IntArray) -> FloatArray:
    vertex_ids = np.unique(np.asarray(mesh.faces[face_indices]).reshape(-1))
    return np.asarray(mesh.vertices[vertex_ids], dtype=np.float64)


def _canonical_direction(vector: FloatArray) -> FloatArray:
    result = np.asarray(vector, dtype=np.float64)
    result /= np.linalg.norm(result)
    dominant = int(np.argmax(np.abs(result)))
    if result[dominant] < 0.0:
        result = -result
    return result


def fit_plane(mesh: Trimesh, face_indices: IntArray) -> PlaneFit:
    points = _region_points(mesh, face_indices)
    if len(points) < 3:
        raise FitError("A plane needs at least three points")

    origin = points.mean(axis=0)
    covariance = np.cov((points - origin).T, bias=True)
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]

    areas = np.asarray(mesh.area_faces[face_indices], dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float64)
    mean_normal = np.average(face_normals, axis=0, weights=areas)
    if np.dot(normal, mean_normal) < 0.0:
        normal = -normal
    normal = _canonical_direction(normal) if np.linalg.norm(mean_normal) < 1e-12 else normal
    normal /= np.linalg.norm(normal)

    offset = float(np.dot(normal, origin))
    residuals = np.abs(points @ normal - offset)
    return PlaneFit(
        normal=normal,
        offset=offset,
        origin=origin,
        rms_error=float(np.sqrt(np.mean(np.square(residuals)))),
        max_error=float(residuals.max(initial=0.0)),
    )


def _orthonormal_basis(axis: FloatArray) -> tuple[FloatArray, FloatArray]:
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, axis))) > 0.8:
        helper = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    second /= np.linalg.norm(second)
    return first, second


def fit_cylinder(mesh: Trimesh, face_indices: IntArray) -> CylinderFit:
    """Fit a cylinder using the normal null-space, then refine its 2D circle."""
    points = _region_points(mesh, face_indices)
    if len(points) < 6:
        raise FitError("A cylinder needs at least six points")

    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float64)
    areas = np.asarray(mesh.area_faces[face_indices], dtype=np.float64)
    normal_covariance = np.einsum("i,ij,ik->jk", areas, normals, normals) / areas.sum()
    _, eigenvectors = np.linalg.eigh(normal_covariance)
    axis = _canonical_direction(eigenvectors[:, 0])
    basis_u, basis_v = _orthonormal_basis(axis)

    reference = points.mean(axis=0)
    relative = points - reference
    x = relative @ basis_u
    y = relative @ basis_v

    design = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    target = np.square(x) + np.square(y)
    center_u, center_v, constant = np.linalg.lstsq(design, target, rcond=None)[0]
    radius_squared = constant + center_u**2 + center_v**2
    if radius_squared <= 0.0:
        raise FitError("Cylinder circle fit is degenerate")
    initial_radius = float(np.sqrt(radius_squared))

    def residual(parameters: FloatArray) -> FloatArray:
        cx, cy, radius = parameters
        return np.sqrt(np.square(x - cx) + np.square(y - cy)) - radius

    optimized = least_squares(
        residual,
        x0=np.array([center_u, center_v, initial_radius]),
        bounds=(np.array([-np.inf, -np.inf, 0.0]), np.full(3, np.inf)),
    )
    center_u, center_v, radius = optimized.x
    axis_point = reference + center_u * basis_u + center_v * basis_v
    axial_coordinates = (points - axis_point) @ axis
    errors = np.abs(residual(optimized.x))
    return CylinderFit(
        axis=axis,
        axis_point=axis_point,
        radius=float(radius),
        axial_min=float(axial_coordinates.min()),
        axial_max=float(axial_coordinates.max()),
        rms_error=float(np.sqrt(np.mean(np.square(errors)))),
        max_error=float(errors.max(initial=0.0)),
    )
