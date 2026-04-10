from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.config import scenario_config_from_dict
from sim.master_outputs import animate_outputs, plot_outputs
from sim.utils.plot_windows import attitude_axis_limits, axis_window_from_values, fuel_fraction_from_remaining_series, windows_from_points
from sim.utils.thruster_plot_geometry import thruster_marker_geometry_body


def _truth_hist(positions_km: list[list[float]], velocities_km_s: list[list[float]] | None = None) -> np.ndarray:
    pos = np.array(positions_km, dtype=float)
    vel = np.zeros_like(pos) if velocities_km_s is None else np.array(velocities_km_s, dtype=float)
    hist = np.zeros((pos.shape[0], 14), dtype=float)
    hist[:, 0:3] = pos
    hist[:, 3:6] = vel
    hist[:, 6] = 1.0
    return hist


def test_plot_outputs_uses_target_reference_orbit_for_ric_multi(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _capture_multi(
        t_s: np.ndarray,
        truth_hist_by_object: dict[str, np.ndarray],
        *,
        frame: str,
        jd_utc_start: float | None = None,
        reference_truth_hist: np.ndarray | None = None,
        mode: str = "interactive",
        out_path: str | None = None,
    ) -> None:
        captured["frame"] = frame
        captured["keys"] = sorted(truth_hist_by_object.keys())
        captured["reference_truth_hist"] = None if reference_truth_hist is None else np.array(reference_truth_hist, dtype=float)
        captured["out_path"] = out_path

    def _noop(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(
        "sim.master_outputs._load_plotting_functions",
        lambda: {
            "plot_orbit_eci": _noop,
            "plot_attitude_tumble": _noop,
            "animate_battlespace_dashboard": _noop,
            "animate_rectangular_prism_attitude": _noop,
            "animate_multi_ric_2d_projections": _noop,
            "plot_body_rates": _noop,
            "plot_control_commands": _noop,
            "animate_multi_trajectory_frame": _noop,
            "plot_multi_control_commands": _noop,
            "plot_multi_ric_2d_projections": _noop,
            "plot_multi_trajectory_frame": _capture_multi,
            "plot_quaternion_components": _noop,
            "plot_ric_2d_projections": _noop,
            "plot_trajectory_frame": _noop,
            "animate_ground_track": _noop,
            "animate_multi_ground_track": _noop,
            "animate_multi_rectangular_prism_ric_curv": _noop,
            "animate_side_by_side_rectangular_prism_ric_attitude": _noop,
        },
    )

    cfg = scenario_config_from_dict(
        {
            "scenario_name": "target_reference_plot",
            "target": {"enabled": True, "reference_orbit": {"enabled": True}},
            "chaser": {"enabled": True},
            "simulator": {"duration_s": 2.0, "dt_s": 1.0},
            "outputs": {
                "output_dir": str(tmp_path),
                "mode": "save",
                "plots": {"enabled": True, "figure_ids": ["trajectory_ric_rect_multi"]},
            },
            "monte_carlo": {"enabled": False},
        }
    )
    t_s = np.array([0.0, 1.0, 2.0], dtype=float)
    truth_hist = {
        "target": _truth_hist([[7000.0, 0.0, 0.0], [7000.2, 0.0, 0.0], [7000.6, 0.0, 0.0]]),
        "chaser": _truth_hist([[7001.0, 0.0, 0.0], [7001.0, 0.1, 0.0], [7001.0, 0.2, 0.0]]),
    }
    target_reference_orbit_truth = np.array(
        [
            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    out = plot_outputs(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        target_reference_orbit_truth=target_reference_orbit_truth,
        thrust_hist={oid: np.zeros((3, 3), dtype=float) for oid in truth_hist},
        desired_attitude_hist=None,
        knowledge_hist={},
        rocket_metrics=None,
        outdir=tmp_path,
        resolve_rocket_stack=lambda specs: None,
        resolve_satellite_isp_s=lambda specs: 0.0,
    )

    assert out["trajectory_ric_rect_multi"].endswith("trajectory_ric_rect_multi.png")
    assert captured["frame"] == "ric_rect"
    assert captured["keys"] == ["chaser", "target"]
    assert np.allclose(captured["reference_truth_hist"], target_reference_orbit_truth)


def test_animate_outputs_uses_target_reference_orbit_for_curv_animations(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {"2d_calls": []}

    def _capture_3d(
        t_s: np.ndarray,
        truth_hist_by_object: dict[str, np.ndarray],
        *,
        frame: str,
        jd_utc_start: float | None = None,
        reference_truth_hist: np.ndarray | None = None,
        mode: str = "interactive",
        out_path: str | None = None,
        fps: float = 30.0,
        speed_multiple: float = 10.0,
        frame_stride: int = 1,
        show_trajectory: bool = True,
    ) -> None:
        captured["3d_keys"] = sorted(truth_hist_by_object.keys())
        captured["3d_frame"] = frame
        captured["3d_show_trajectory"] = show_trajectory
        captured["3d_reference_truth_hist"] = None if reference_truth_hist is None else np.array(reference_truth_hist, dtype=float)
        captured["3d_out_path"] = out_path

    def _capture_2d(
        t_s: np.ndarray,
        truth_hist_by_object: dict[str, np.ndarray],
        *,
        frame: str,
        reference_truth_hist: np.ndarray,
        planes: list[str] | None = None,
        mode: str = "interactive",
        out_path: str | None = None,
        fps: float = 30.0,
        speed_multiple: float = 10.0,
        frame_stride: int = 1,
        show_trajectory: bool = True,
    ) -> None:
        captured["2d_calls"].append(
            {
                "keys": sorted(truth_hist_by_object.keys()),
                "frame": frame,
                "planes": list(planes or []),
                "show_trajectory": show_trajectory,
                "reference_truth_hist": np.array(reference_truth_hist, dtype=float),
                "out_path": out_path,
            }
        )

    def _noop(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(
        "sim.master_outputs._load_plotting_functions",
        lambda: {
            "plot_orbit_eci": _noop,
            "plot_attitude_tumble": _noop,
            "animate_battlespace_dashboard": _noop,
            "animate_rectangular_prism_attitude": _noop,
            "animate_multi_ric_2d_projections": _capture_2d,
            "plot_body_rates": _noop,
            "plot_control_commands": _noop,
            "animate_multi_trajectory_frame": _capture_3d,
            "plot_multi_control_commands": _noop,
            "plot_multi_ric_2d_projections": _noop,
            "plot_multi_trajectory_frame": _noop,
            "plot_quaternion_components": _noop,
            "plot_ric_2d_projections": _noop,
            "plot_trajectory_frame": _noop,
            "animate_ground_track": _noop,
            "animate_multi_ground_track": _noop,
            "animate_multi_rectangular_prism_ric_curv": _noop,
            "animate_side_by_side_rectangular_prism_ric_attitude": _noop,
        },
    )

    cfg = scenario_config_from_dict(
        {
            "scenario_name": "target_reference_animation",
            "target": {"enabled": True, "reference_orbit": {"enabled": True}},
            "chaser": {"enabled": True},
            "simulator": {"duration_s": 2.0, "dt_s": 1.0},
            "outputs": {
                "output_dir": str(tmp_path),
                "mode": "save",
                "animations": {
                    "enabled": True,
                    "types": [
                        "target_reference_ric_curv_3d",
                        "target_reference_ric_curv_2d",
                        "target_reference_ric_curv_2d_ri",
                        "target_reference_ric_curv_2d_rc",
                    ],
                    "target_reference_ric_curv_3d_show_trajectory": False,
                    "target_reference_ric_curv_2d_show_trajectory": True,
                    "target_reference_ric_curv_2d_ri_show_trajectory": False,
                    "target_reference_ric_curv_2d_rc_show_trajectory": True,
                    "target_reference_ric_curv_2d_planes": ["ri", "rc"],
                },
            },
            "monte_carlo": {"enabled": False},
        }
    )
    t_s = np.array([0.0, 1.0, 2.0], dtype=float)
    truth_hist = {
        "target": _truth_hist([[7000.0, 0.0, 0.0], [7000.2, 0.0, 0.0], [7000.6, 0.0, 0.0]]),
        "chaser": _truth_hist([[7001.0, 0.0, 0.0], [7001.0, 0.1, 0.0], [7001.0, 0.2, 0.0]]),
    }
    target_reference_orbit_truth = np.array(
        [
            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    out = animate_outputs(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        thrust_hist={oid: np.zeros((3, 3), dtype=float) for oid in truth_hist},
        target_reference_orbit_truth=target_reference_orbit_truth,
        outdir=tmp_path,
        resolve_satellite_isp_s=lambda specs: 0.0,
    )

    assert out["target_reference_ric_curv_3d"].endswith("target_reference_ric_curv_3d.mp4")
    assert out["target_reference_ric_curv_2d"].endswith("target_reference_ric_curv_2d.mp4")
    assert out["target_reference_ric_curv_2d_ri"].endswith("target_reference_ric_curv_2d_ri.mp4")
    assert out["target_reference_ric_curv_2d_rc"].endswith("target_reference_ric_curv_2d_rc.mp4")
    assert captured["3d_frame"] == "ric_curv"
    assert captured["3d_keys"] == ["chaser", "target"]
    assert captured["3d_show_trajectory"] is False
    assert np.allclose(captured["3d_reference_truth_hist"], target_reference_orbit_truth)
    assert len(captured["2d_calls"]) == 3

    calls_by_path = {Path(str(call["out_path"])).name: call for call in captured["2d_calls"]}
    combo = calls_by_path["target_reference_ric_curv_2d.mp4"]
    ri = calls_by_path["target_reference_ric_curv_2d_ri.mp4"]
    rc = calls_by_path["target_reference_ric_curv_2d_rc.mp4"]

    assert combo["frame"] == "ric_curv"
    assert combo["keys"] == ["chaser", "target"]
    assert combo["planes"] == ["ri", "rc"]
    assert combo["show_trajectory"] is True
    assert np.allclose(combo["reference_truth_hist"], target_reference_orbit_truth)

    assert ri["planes"] == ["ri"]
    assert ri["show_trajectory"] is False
    assert np.allclose(ri["reference_truth_hist"], target_reference_orbit_truth)

    assert rc["planes"] == ["rc"]
    assert rc["show_trajectory"] is True
    assert np.allclose(rc["reference_truth_hist"], target_reference_orbit_truth)


def test_animate_outputs_marks_thruster_active_in_ric_attitude_animation(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _capture_attitude(
        t_s: np.ndarray,
        truth_hist: np.ndarray,
        *,
        lx_m: float,
        ly_m: float,
        lz_m: float,
        frame: str = "eci",
        thruster_active_mask: np.ndarray | None = None,
        thruster_position_body_m: np.ndarray | None = None,
        thruster_direction_body: np.ndarray | None = None,
        body_facecolor: str = "#1F77B4",
        thruster_inactive_facecolor: str = "#808080",
        thruster_active_facecolor: str = "#D95F02",
        mode: str = "interactive",
        out_path: str | None = None,
        fps: float = 30.0,
        speed_multiple: float = 10.0,
    ) -> None:
        captured["frame"] = frame
        captured["dims"] = (lx_m, ly_m, lz_m)
        captured["mask"] = None if thruster_active_mask is None else np.array(thruster_active_mask, dtype=bool)
        captured["thruster_position_body_m"] = (
            None if thruster_position_body_m is None else np.array(thruster_position_body_m, dtype=float)
        )
        captured["thruster_direction_body"] = (
            None if thruster_direction_body is None else np.array(thruster_direction_body, dtype=float)
        )
        captured["body_facecolor"] = body_facecolor
        captured["thruster_inactive_facecolor"] = thruster_inactive_facecolor
        captured["thruster_active_facecolor"] = thruster_active_facecolor
        captured["out_path"] = out_path

    def _noop(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(
        "sim.master_outputs._load_plotting_functions",
        lambda: {
            "plot_orbit_eci": _noop,
            "plot_attitude_tumble": _noop,
            "animate_battlespace_dashboard": _noop,
            "animate_rectangular_prism_attitude": _capture_attitude,
            "animate_multi_ric_2d_projections": _noop,
            "plot_body_rates": _noop,
            "plot_control_commands": _noop,
            "animate_multi_trajectory_frame": _noop,
            "plot_multi_control_commands": _noop,
            "plot_multi_ric_2d_projections": _noop,
            "plot_multi_trajectory_frame": _noop,
            "plot_quaternion_components": _noop,
            "plot_ric_2d_projections": _noop,
            "plot_trajectory_frame": _noop,
            "animate_ground_track": _noop,
            "animate_multi_ground_track": _noop,
            "animate_multi_rectangular_prism_ric_curv": _noop,
            "animate_side_by_side_rectangular_prism_ric_attitude": _noop,
        },
    )

    cfg = scenario_config_from_dict(
        {
            "scenario_name": "attitude_ric_thruster_animation",
            "target": {"enabled": True, "specs": {"thruster": "BASIC_CHEMICAL_Z_BOTTOM"}},
            "simulator": {"duration_s": 2.0, "dt_s": 1.0},
            "outputs": {
                "output_dir": str(tmp_path),
                "mode": "save",
                "animations": {
                    "enabled": True,
                    "types": ["attitude_ric_thruster"],
                    "attitude_ric_thruster_object_ids": ["target"],
                    "attitude_ric_thruster_dims_m": {"target": [1.0, 2.0, 3.0]},
                },
            },
            "monte_carlo": {"enabled": False},
        }
    )
    t_s = np.array([0.0, 1.0, 2.0], dtype=float)
    truth_hist = {"target": _truth_hist([[7000.0, 0.0, 0.0], [7000.0, 0.0, 0.0], [7000.0, 0.0, 0.0]])}
    thrust_hist = {"target": np.array([[0.0, 0.0, 0.0], [1.0e-6, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)}

    out = animate_outputs(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        thrust_hist=thrust_hist,
        target_reference_orbit_truth=None,
        outdir=tmp_path,
        resolve_satellite_isp_s=lambda specs: 0.0,
    )

    assert out["target_attitude_ric_thruster"].endswith("target_attitude_ric_thruster.mp4")
    assert captured["frame"] == "ric"
    assert captured["dims"] == (1.0, 2.0, 3.0)
    assert np.array_equal(captured["mask"], np.array([False, True, False]))
    assert np.allclose(captured["thruster_position_body_m"], np.array([0.0, 0.0, -0.5]))
    assert np.allclose(captured["thruster_direction_body"], np.array([0.0, 0.0, 1.0]))
    assert captured["thruster_inactive_facecolor"] == "#808080"
    assert captured["thruster_active_facecolor"] == "#D95F02"


def test_animate_outputs_routes_battlespace_dashboard(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _capture_dashboard(
        t_s: np.ndarray,
        truth_hist_by_object: dict[str, np.ndarray],
        *,
        reference_truth_hist: np.ndarray,
        target_object_id: str = "target",
        chaser_object_id: str = "chaser",
        thrust_hist_by_object: dict[str, np.ndarray] | None = None,
        delta_v_remaining_m_s_by_object: dict[str, np.ndarray] | None = None,
        prism_dims_m_by_object: dict[str, list[float] | np.ndarray] | None = None,
        thruster_mounts_by_object: dict[str, dict[str, np.ndarray] | None] | None = None,
        thruster_active_threshold_km_s2: float = 1e-15,
        show_trajectory: bool = True,
        mode: str = "interactive",
        out_path: str | None = None,
        fps: float = 30.0,
        speed_multiple: float = 10.0,
        frame_stride: int = 1,
    ) -> None:
        captured["keys"] = sorted(truth_hist_by_object.keys())
        captured["reference_truth_hist"] = np.array(reference_truth_hist, dtype=float)
        captured["target_object_id"] = target_object_id
        captured["chaser_object_id"] = chaser_object_id
        captured["thrust_hist_keys"] = sorted((thrust_hist_by_object or {}).keys())
        captured["dv_remaining"] = {
            oid: np.array(hist, dtype=float) for oid, hist in (delta_v_remaining_m_s_by_object or {}).items()
        }
        captured["dims"] = dict(prism_dims_m_by_object or {})
        captured["thruster_mounts"] = {
            oid: (
                None
                if mount is None
                else {
                    "position_body_m": np.array(mount["position_body_m"], dtype=float),
                    "direction_body": np.array(mount["direction_body"], dtype=float),
                }
            )
            for oid, mount in (thruster_mounts_by_object or {}).items()
        }
        captured["threshold"] = thruster_active_threshold_km_s2
        captured["show_trajectory"] = show_trajectory
        captured["out_path"] = out_path

    def _noop(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(
        "sim.master_outputs._load_plotting_functions",
        lambda: {
            "plot_orbit_eci": _noop,
            "plot_attitude_tumble": _noop,
            "animate_battlespace_dashboard": _capture_dashboard,
            "animate_rectangular_prism_attitude": _noop,
            "animate_multi_ric_2d_projections": _noop,
            "plot_body_rates": _noop,
            "plot_control_commands": _noop,
            "animate_multi_trajectory_frame": _noop,
            "plot_multi_control_commands": _noop,
            "plot_multi_ric_2d_projections": _noop,
            "plot_multi_trajectory_frame": _noop,
            "plot_quaternion_components": _noop,
            "plot_ric_2d_projections": _noop,
            "plot_trajectory_frame": _noop,
            "animate_ground_track": _noop,
            "animate_multi_ground_track": _noop,
            "animate_multi_rectangular_prism_ric_curv": _noop,
            "animate_side_by_side_rectangular_prism_ric_attitude": _noop,
        },
    )

    cfg = scenario_config_from_dict(
        {
            "scenario_name": "battlespace_dashboard",
            "target": {
                "enabled": True,
                "reference_orbit": {"enabled": True},
                "specs": {"dry_mass_kg": 80.0, "fuel_mass_kg": 20.0, "thruster": "BASIC_CHEMICAL_Z_BOTTOM"},
            },
            "chaser": {
                "enabled": True,
                "specs": {"dry_mass_kg": 60.0, "fuel_mass_kg": 30.0, "thruster": "BASIC_CHEMICAL_Z_BOTTOM"},
            },
            "simulator": {"duration_s": 2.0, "dt_s": 1.0},
            "outputs": {
                "output_dir": str(tmp_path),
                "mode": "save",
                "animations": {
                    "enabled": True,
                    "types": ["battlespace_dashboard"],
                    "battlespace_dashboard_show_trajectory": False,
                    "battlespace_dashboard_attitude_dims_m": {
                        "target": [1.0, 1.5, 2.0],
                        "chaser": [2.0, 2.5, 3.0],
                    },
                    "battlespace_dashboard_thruster_active_threshold_km_s2": 2.0e-15,
                },
            },
            "monte_carlo": {"enabled": False},
        }
    )
    t_s = np.array([0.0, 1.0, 2.0], dtype=float)
    target_hist = _truth_hist([[7000.0, 0.0, 0.0], [7000.1, 0.0, 0.0], [7000.2, 0.0, 0.0]])
    chaser_hist = _truth_hist([[7001.0, 0.0, 0.0], [7001.0, 0.1, 0.0], [7001.0, 0.2, 0.0]])
    target_hist[:, 13] = np.array([100.0, 95.0, 90.0], dtype=float)
    chaser_hist[:, 13] = np.array([90.0, 80.0, 70.0], dtype=float)
    truth_hist = {
        "target": target_hist,
        "chaser": chaser_hist,
    }
    thrust_hist = {oid: np.zeros((3, 3), dtype=float) for oid in truth_hist}
    target_reference_orbit_truth = np.array(
        [
            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    out = animate_outputs(
        cfg=cfg,
        t_s=t_s,
        truth_hist=truth_hist,
        thrust_hist=thrust_hist,
        target_reference_orbit_truth=target_reference_orbit_truth,
        outdir=tmp_path,
        resolve_satellite_isp_s=lambda specs: 200.0,
    )

    assert out["battlespace_dashboard"].endswith("battlespace_dashboard.mp4")
    assert captured["keys"] == ["chaser", "target"]
    assert captured["target_object_id"] == "target"
    assert captured["chaser_object_id"] == "chaser"
    assert captured["thrust_hist_keys"] == ["chaser", "target"]
    assert np.allclose(captured["reference_truth_hist"], target_reference_orbit_truth)
    assert captured["dims"] == {
        "target": [1.0, 1.5, 2.0],
        "chaser": [2.0, 2.5, 3.0],
    }
    assert captured["threshold"] == 2.0e-15
    assert captured["show_trajectory"] is False
    assert "target" in captured["dv_remaining"]
    assert "chaser" in captured["dv_remaining"]
    assert np.allclose(captured["thruster_mounts"]["target"]["position_body_m"], np.array([0.0, 0.0, -0.5]))
    assert np.allclose(captured["thruster_mounts"]["target"]["direction_body"], np.array([0.0, 0.0, 1.0]))
    assert np.allclose(captured["thruster_mounts"]["chaser"]["position_body_m"], np.array([0.0, 0.0, -0.5]))
    assert np.allclose(captured["thruster_mounts"]["chaser"]["direction_body"], np.array([0.0, 0.0, 1.0]))
    assert float(captured["dv_remaining"]["target"][1]) < float(captured["dv_remaining"]["target"][0])


def test_thruster_marker_geometry_uses_plume_direction_face() -> None:
    points, _ = thruster_marker_geometry_body(
        lx_m=1.0,
        ly_m=2.0,
        lz_m=3.0,
        thruster_position_body_m=None,
        thruster_direction_body=np.array([0.0, 1.0, 0.0], dtype=float),
    )

    assert float(np.min(points[:, 1])) > 1.0


def test_thruster_marker_geometry_prioritizes_plume_face_over_conflicting_mount_face() -> None:
    points, _ = thruster_marker_geometry_body(
        lx_m=1.0,
        ly_m=2.0,
        lz_m=3.0,
        thruster_position_body_m=np.array([0.0, 0.0, -1.5], dtype=float),
        thruster_direction_body=np.array([0.0, 0.0, 1.0], dtype=float),
    )

    assert float(np.min(points[:, 2])) > 1.5


def test_axis_window_from_values_expands_independently() -> None:
    xlim = axis_window_from_values([np.array([0.0, 20.0], dtype=float)], min_span=1.0, margin=1.15)
    ylim = axis_window_from_values([np.array([0.0, 1.0], dtype=float)], min_span=1.0, margin=1.15)

    assert np.isclose(np.mean(xlim), 10.0, atol=1e-6)
    assert np.isclose(np.mean(ylim), 0.5, atol=1e-6)
    assert (xlim[1] - xlim[0]) > (ylim[1] - ylim[0])


def test_windows_from_points_center_on_midpoint_of_current_objects() -> None:
    xlim, ylim, zlim = windows_from_points(
        [
            np.array([0.0, 0.0, 0.0], dtype=float),
            np.array([1.0, 20.0, 5.0], dtype=float),
        ],
        axis_indices=(1, 0, 2),
        min_span=1.0,
        margin=1.15,
    )

    assert np.isclose(np.mean(xlim), 10.0, atol=1e-6)
    assert np.isclose(np.mean(ylim), 0.5, atol=1e-6)
    assert np.isclose(np.mean(zlim), 2.5, atol=1e-6)
    assert (xlim[1] - xlim[0]) > (zlim[1] - zlim[0]) > (ylim[1] - ylim[0])


def test_attitude_axis_limits_flip_intrack_only_for_ric() -> None:
    xlim, ylim, zlim = attitude_axis_limits("ric", 2.0)

    assert xlim == (2.0, -2.0)
    assert ylim == (-2.0, 2.0)
    assert zlim == (-2.0, 2.0)


def test_fuel_fraction_from_remaining_series_normalizes_to_initial_budget() -> None:
    frac = fuel_fraction_from_remaining_series(np.array([200.0, 150.0, 50.0, 0.0], dtype=float))

    assert np.allclose(frac, np.array([1.0, 0.75, 0.25, 0.0], dtype=float))
