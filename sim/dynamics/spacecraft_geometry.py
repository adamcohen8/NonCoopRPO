from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RectangularPrismGeometry:
    lx_m: float
    ly_m: float
    lz_m: float

    def __post_init__(self) -> None:
        if self.lx_m <= 0.0 or self.ly_m <= 0.0 or self.lz_m <= 0.0:
            raise ValueError("Rectangular prism dimensions must be positive.")

    def face_centers_body_m(self) -> np.ndarray:
        return np.array(
            [
                [0.5 * self.lx_m, 0.0, 0.0],
                [-0.5 * self.lx_m, 0.0, 0.0],
                [0.0, 0.5 * self.ly_m, 0.0],
                [0.0, -0.5 * self.ly_m, 0.0],
                [0.0, 0.0, 0.5 * self.lz_m],
                [0.0, 0.0, -0.5 * self.lz_m],
            ],
            dtype=float,
        )

    def face_normals_body(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )

    def face_areas_m2(self) -> np.ndarray:
        a_x = self.ly_m * self.lz_m
        a_y = self.lx_m * self.lz_m
        a_z = self.lx_m * self.ly_m
        return np.array([a_x, a_x, a_y, a_y, a_z, a_z], dtype=float)

    def projected_area_m2(self, incident_dir_body: np.ndarray) -> float:
        u = np.array(incident_dir_body, dtype=float)
        n = float(np.linalg.norm(u))
        if n <= 0.0:
            return 0.0
        u = u / n
        normals = self.face_normals_body()
        areas = self.face_areas_m2()
        illum = np.maximum(0.0, -(normals @ u))
        return float(np.sum(areas * illum))

    def face_forces_body_n(self, incident_dir_body: np.ndarray, pressure_n_m2: float) -> np.ndarray:
        u = np.array(incident_dir_body, dtype=float)
        n = float(np.linalg.norm(u))
        if n <= 0.0 or pressure_n_m2 <= 0.0:
            return np.zeros((6, 3))
        u = u / n
        normals = self.face_normals_body()
        areas = self.face_areas_m2()
        illum = np.maximum(0.0, -(normals @ u))
        mags = pressure_n_m2 * areas * illum
        # Lumped body force direction opposite incident direction.
        return mags[:, None] * (-u[None, :])

    def face_torque_sum_body_nm(self, incident_dir_body: np.ndarray, pressure_n_m2: float) -> np.ndarray:
        r_faces = self.face_centers_body_m()
        f_faces = self.face_forces_body_n(incident_dir_body, pressure_n_m2)
        tau_faces = np.cross(r_faces, f_faces)
        return np.sum(tau_faces, axis=0)
