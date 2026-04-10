from __future__ import annotations

import numpy as np


def default_thruster_mount_position_body(
    *,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    outward_normal_body: np.ndarray,
) -> np.ndarray:
    dims = np.array([lx_m, ly_m, lz_m], dtype=float)
    axis = np.array(outward_normal_body, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, -0.5 * lz_m], dtype=float)
    axis /= norm
    idx = int(np.argmax(np.abs(axis)))
    pos = np.zeros(3, dtype=float)
    pos[idx] = 0.5 * dims[idx] * float(np.sign(axis[idx]) if abs(axis[idx]) > 1e-12 else 1.0)
    return pos


def orthonormal_basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.array(axis, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n <= 1e-12:
        a = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        a /= n
    basis = np.eye(3)
    ref = basis[:, int(np.argmin(np.abs(a)))]
    s = ref - np.dot(ref, a) * a
    s_norm = float(np.linalg.norm(s))
    if s_norm <= 1e-12:
        ref = basis[:, (int(np.argmin(np.abs(a))) + 1) % 3]
        s = ref - np.dot(ref, a) * a
        s_norm = float(np.linalg.norm(s))
    s /= max(s_norm, 1e-12)
    u = np.cross(a, s)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    return a, s, u


def thruster_face_mount_body(
    *,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    thruster_position_body_m: np.ndarray | None,
    outward_normal_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    half_dims = 0.5 * np.array([lx_m, ly_m, lz_m], dtype=float)
    axis = np.array(outward_normal_body, dtype=float).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        axis = np.array([0.0, 0.0, -1.0], dtype=float)
    else:
        axis /= axis_norm

    if thruster_position_body_m is None:
        mount = default_thruster_mount_position_body(
            lx_m=lx_m,
            ly_m=ly_m,
            lz_m=lz_m,
            outward_normal_body=axis,
        )
        return mount, axis

    raw_mount = np.array(thruster_position_body_m, dtype=float).reshape(3)
    raw_mount = np.clip(raw_mount, -half_dims, half_dims)
    idx = int(np.argmax(np.abs(axis)))
    sign = float(np.sign(axis[idx]))
    if abs(sign) <= 1e-12 and float(np.linalg.norm(raw_mount)) > 1e-12:
        scaled = np.abs(raw_mount) / np.maximum(half_dims, 1e-12)
        idx = int(np.argmax(scaled))
        sign = float(np.sign(raw_mount[idx]))
    if abs(sign) <= 1e-12:
        sign = 1.0
    face_axis = np.zeros(3, dtype=float)
    face_axis[idx] = sign
    mount = raw_mount.copy()
    mount[idx] = sign * half_dims[idx]
    return mount, face_axis


def thruster_marker_geometry_body(
    *,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    thruster_position_body_m: np.ndarray | None = None,
    thruster_direction_body: np.ndarray | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    dims = np.array([lx_m, ly_m, lz_m], dtype=float)
    marker_scale = float(max(np.min(dims), 1e-6))
    axis_raw = (
        np.array(thruster_direction_body, dtype=float).reshape(3)
        if thruster_direction_body is not None
        else np.array([0.0, 0.0, 1.0], dtype=float)
    )
    # The stored mount axis is the nozzle / plume direction, so render the marker on that exterior face.
    mount, outward_axis = thruster_face_mount_body(
        lx_m=lx_m,
        ly_m=ly_m,
        lz_m=lz_m,
        thruster_position_body_m=thruster_position_body_m,
        outward_normal_body=axis_raw,
    )
    axis, side, up = orthonormal_basis_from_axis(outward_axis)

    inner_radius = 0.065 * marker_scale
    outer_radius = 0.13 * marker_scale
    collar_radius = 0.17 * marker_scale
    nozzle_length = 0.26 * marker_scale
    base_offset = 0.02 * marker_scale
    exit_offset = base_offset + nozzle_length
    collar_offset = 0.008 * marker_scale
    segments = 24

    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False, dtype=float)
    unit_ring = np.vstack([np.cos(theta), np.sin(theta)]).T
    base_center = mount + axis * base_offset
    exit_center = mount + axis * exit_offset
    collar_center = mount + axis * collar_offset
    base_ring = np.vstack([base_center + inner_radius * (c * side + s * up) for c, s in unit_ring])
    exit_ring = np.vstack([exit_center + outer_radius * (c * side + s * up) for c, s in unit_ring])
    collar_ring = np.vstack([collar_center + collar_radius * (c * side + s * up) for c, s in unit_ring])
    points = np.vstack([base_ring, exit_ring, collar_ring])

    faces: list[list[int]] = []
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append([idx, nxt, segments + nxt, segments + idx])
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append([2 * segments + idx, 2 * segments + nxt, nxt, idx])
    faces.append(list(range(segments)))
    faces.append(list(range(2 * segments - 1, segments - 1, -1)))
    faces.append(list(range(3 * segments - 1, 2 * segments - 1, -1)))
    return points, faces
