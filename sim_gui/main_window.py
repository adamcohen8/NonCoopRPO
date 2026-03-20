from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import tempfile
import yaml

from PySide6.QtCore import QEvent, QProcess, QProcessEnvironment, Qt
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QCheckBox,
    QScrollArea,
    QStackedWidget,
    QSplitter,
    QSpinBox,
    QDoubleSpinBox,
    QStatusBar,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from sim.app.services import (
    build_cli_run_command,
    dump_config_text,
    get_default_config_path,
    get_output_files,
    get_repo_root,
    list_available_configs,
    load_config,
    parse_config_text,
    save_config,
    validate_config,
)


OUTPUT_MODES = ["interactive", "save", "both"]
ZERO_POINTER = {"kind": "python", "module": "sim.control.orbit.zero_controller", "class_name": "ZeroController", "params": {}}
ZERO_TORQUE_POINTER = {"kind": "python", "module": "sim.control.attitude.zero_torque", "class_name": "ZeroTorqueController", "params": {}}

BASE_GUIDANCE_OPTIONS = {
    "rocket": [
        ("Open Loop Pitch Program", {"kind": "python", "module": "sim.rocket.guidance", "class_name": "OpenLoopPitchProgramGuidance", "params": {}}),
        ("Closed Loop Insertion", {"kind": "python", "module": "sim.rocket.guidance", "class_name": "ClosedLoopInsertionGuidance", "params": {}}),
        ("Hold Attitude", {"kind": "python", "module": "sim.rocket.guidance", "class_name": "HoldAttitudeGuidance", "params": {}}),
    ],
}
GUIDANCE_MODIFIER_OPTIONS = [
    ("TVC Steering", {"kind": "python", "module": "sim.rocket.guidance", "class_name": "TVCSteeringGuidance", "params": {}}),
    ("Max Q Throttle Limiter", {"kind": "python", "module": "sim.rocket.guidance", "class_name": "MaxQThrottleLimiterGuidance", "params": {}}),
    ("Orbit Insertion Cutoff", {"kind": "python", "module": "sim.rocket.guidance", "class_name": "OrbitInsertionCutoffGuidance", "params": {}}),
]

ORBIT_CONTROL_OPTIONS = {
    "rocket": [("Zero Controller", ZERO_POINTER)],
    "chaser": [
        ("Zero Controller", ZERO_POINTER),
        ("HCW LQR", {"kind": "python", "module": "sim.control.orbit.lqr", "class_name": "HCWLQRController", "params": {}}),
        ("Relative Orbit MPC", {"kind": "python", "module": "sim.control.orbit.relative_mpc", "class_name": "RelativeOrbitMPCController", "params": {}}),
        ("HCW Relative MPC", {"kind": "python", "module": "sim.control.orbit.hcw_mpc", "class_name": "HCWRelativeOrbitMPCController", "params": {}}),
        ("Stationkeeping", {"kind": "python", "module": "sim.control.orbit.baseline", "class_name": "StationkeepingController", "params": {}}),
    ],
    "target": [
        ("Zero Controller", ZERO_POINTER),
        ("Stationkeeping", {"kind": "python", "module": "sim.control.orbit.baseline", "class_name": "StationkeepingController", "params": {}}),
    ],
}

ATTITUDE_CONTROL_OPTIONS = {
    "rocket": [("Zero Torque", ZERO_TORQUE_POINTER)],
    "chaser": [
        ("Zero Torque", ZERO_TORQUE_POINTER),
        ("Surrogate Snap ECI", {"kind": "python", "module": "sim.control.attitude.surrogate_snap", "class_name": "SurrogateSnapECIController", "params": {}}),
        ("Surrogate Snap RIC", {"kind": "python", "module": "sim.control.attitude.surrogate_snap", "class_name": "SurrogateSnapRICController", "params": {}}),
        ("RIC Detumble PD", {"kind": "python", "module": "sim.control.attitude.detumble_pd", "class_name": "RICDetumblePDController", "params": {}}),
        ("Quaternion PD", {"kind": "python", "module": "sim.control.attitude.baseline", "class_name": "QuaternionPDController", "params": {}}),
    ],
    "target": [
        ("Zero Torque", ZERO_TORQUE_POINTER),
        ("Surrogate Snap ECI", {"kind": "python", "module": "sim.control.attitude.surrogate_snap", "class_name": "SurrogateSnapECIController", "params": {}}),
        ("Quaternion PD", {"kind": "python", "module": "sim.control.attitude.baseline", "class_name": "QuaternionPDController", "params": {}}),
    ],
}

MISSION_OPTIONS = {
    "rocket": [
        ("None", None),
        ("Rocket Mission", {"kind": "python", "module": "sim.mission.modules", "class_name": "RocketMissionModule", "params": {}}),
    ],
    "chaser": [
        ("None", None),
        ("Satellite Mission", {"kind": "python", "module": "sim.mission.modules", "class_name": "SatelliteMissionModule", "params": {}}),
        ("End State Maneuver", {"kind": "python", "module": "sim.mission.modules", "class_name": "EndStateManeuverMissionModule", "params": {}}),
        ("Integrated Command", {"kind": "python", "module": "sim.mission.modules", "class_name": "IntegratedCommandMissionModule", "params": {}}),
        ("Predictive Integrated Command", {"kind": "python", "module": "sim.mission.modules", "class_name": "PredictiveIntegratedCommandMissionModule", "params": {}}),
        ("Attitude Detumble Gate", {"kind": "python", "module": "sim.mission.modules", "class_name": "AttitudeDetumbleGateMissionModule", "params": {}}),
    ],
    "target": [
        ("None", None),
        ("Satellite Mission", {"kind": "python", "module": "sim.mission.modules", "class_name": "SatelliteMissionModule", "params": {}}),
        ("Attitude Detumble Gate", {"kind": "python", "module": "sim.mission.modules", "class_name": "AttitudeDetumbleGateMissionModule", "params": {}}),
    ],
}
MISSION_STRATEGY_OPTIONS = {
    "rocket": [
        ("None", None),
        ("Pursuit", {"kind": "python", "module": "sim.mission.modules", "class_name": "RocketPursuitMissionStrategy", "params": {}}),
        ("Predefined Orbit", {"kind": "python", "module": "sim.mission.modules", "class_name": "RocketPredefinedOrbitMissionStrategy", "params": {}}),
    ],
    "chaser": [
        ("None", None),
        ("Pursuit", {"kind": "python", "module": "sim.mission.modules", "class_name": "PursuitMissionStrategy", "params": {}}),
        ("Evade", {"kind": "python", "module": "sim.mission.modules", "class_name": "EvadeMissionStrategy", "params": {}}),
        ("Hold", {"kind": "python", "module": "sim.mission.modules", "class_name": "HoldMissionStrategy", "params": {}}),
        ("Desired State", {"kind": "python", "module": "sim.mission.modules", "class_name": "DesiredStateMissionStrategy", "params": {}}),
        ("Mission Executive", {"kind": "python", "module": "sim.mission.modules", "class_name": "MissionExecutiveStrategy", "params": {}}),
        ("Station Keep", {"kind": "python", "module": "sim.mission.modules", "class_name": "StationKeepMissionStrategy", "params": {}}),
        ("Inspect", {"kind": "python", "module": "sim.mission.modules", "class_name": "InspectMissionStrategy", "params": {}}),
        ("Defensive", {"kind": "python", "module": "sim.mission.modules", "class_name": "DefensiveMissionStrategy", "params": {}}),
        ("Safe Hold", {"kind": "python", "module": "sim.mission.modules", "class_name": "SafeHoldMissionStrategy", "params": {}}),
    ],
    "target": [
        ("None", None),
        ("Pursuit", {"kind": "python", "module": "sim.mission.modules", "class_name": "PursuitMissionStrategy", "params": {}}),
        ("Evade", {"kind": "python", "module": "sim.mission.modules", "class_name": "EvadeMissionStrategy", "params": {}}),
        ("Hold", {"kind": "python", "module": "sim.mission.modules", "class_name": "HoldMissionStrategy", "params": {}}),
        ("Desired State", {"kind": "python", "module": "sim.mission.modules", "class_name": "DesiredStateMissionStrategy", "params": {}}),
        ("Mission Executive", {"kind": "python", "module": "sim.mission.modules", "class_name": "MissionExecutiveStrategy", "params": {}}),
        ("Station Keep", {"kind": "python", "module": "sim.mission.modules", "class_name": "StationKeepMissionStrategy", "params": {}}),
        ("Inspect", {"kind": "python", "module": "sim.mission.modules", "class_name": "InspectMissionStrategy", "params": {}}),
        ("Defensive", {"kind": "python", "module": "sim.mission.modules", "class_name": "DefensiveMissionStrategy", "params": {}}),
        ("Safe Hold", {"kind": "python", "module": "sim.mission.modules", "class_name": "SafeHoldMissionStrategy", "params": {}}),
    ],
}
MISSION_EXECUTION_OPTIONS = {
    "rocket": [
        ("None", None),
        ("Go Now", {"kind": "python", "module": "sim.mission.modules", "class_name": "RocketGoNowExecution", "params": {}}),
        ("Go When Possible", {"kind": "python", "module": "sim.mission.modules", "class_name": "RocketGoWhenPossibleExecution", "params": {}}),
        ("Wait For Optimal", {"kind": "python", "module": "sim.mission.modules", "class_name": "RocketWaitOptimalExecution", "params": {}}),
    ],
    "chaser": [
        ("None", None),
        ("Controller Pointing", {"kind": "python", "module": "sim.mission.modules", "class_name": "ControllerPointingExecution", "params": {}}),
        ("Predictive Burn", {"kind": "python", "module": "sim.mission.modules", "class_name": "PredictiveBurnExecution", "params": {}}),
        ("Integrated Command", {"kind": "python", "module": "sim.mission.modules", "class_name": "IntegratedCommandExecution", "params": {}}),
        ("Budgeted End State", {"kind": "python", "module": "sim.mission.modules", "class_name": "BudgetedEndStateExecution", "params": {}}),
        ("Direct Integrated", {"kind": "python", "module": "sim.mission.modules", "class_name": "DirectIntegratedExecution", "params": {}}),
        ("Impulsive", {"kind": "python", "module": "sim.mission.modules", "class_name": "ImpulsiveExecution", "params": {}}),
        ("Safe Hold", {"kind": "python", "module": "sim.mission.modules", "class_name": "SafeHoldExecution", "params": {}}),
    ],
    "target": [
        ("None", None),
        ("Controller Pointing", {"kind": "python", "module": "sim.mission.modules", "class_name": "ControllerPointingExecution", "params": {}}),
        ("Predictive Burn", {"kind": "python", "module": "sim.mission.modules", "class_name": "PredictiveBurnExecution", "params": {}}),
        ("Integrated Command", {"kind": "python", "module": "sim.mission.modules", "class_name": "IntegratedCommandExecution", "params": {}}),
        ("Budgeted End State", {"kind": "python", "module": "sim.mission.modules", "class_name": "BudgetedEndStateExecution", "params": {}}),
        ("Direct Integrated", {"kind": "python", "module": "sim.mission.modules", "class_name": "DirectIntegratedExecution", "params": {}}),
        ("Impulsive", {"kind": "python", "module": "sim.mission.modules", "class_name": "ImpulsiveExecution", "params": {}}),
        ("Safe Hold", {"kind": "python", "module": "sim.mission.modules", "class_name": "SafeHoldExecution", "params": {}}),
    ],
}
SATELLITE_PRESET_OPTIONS = ["BASIC_SATELLITE"]
ROCKET_PRESET_OPTIONS = ["BASIC_TWO_STAGE_STACK"]
FIGURE_ID_OPTIONS = [
    "orbit_eci",
    "trajectory_ecef",
    "trajectory_ric_rect",
    "trajectory_ric_curv",
    "trajectory_ric_rect_2d",
    "trajectory_ric_curv_2d",
    "trajectory_eci_multi",
    "trajectory_ecef_multi",
    "trajectory_ric_rect_multi",
    "trajectory_ric_curv_multi",
    "trajectory_ric_rect_2d_multi",
    "trajectory_ric_curv_2d_multi",
    "attitude",
    "quaternion_eci",
    "quaternion_ric",
    "rates_eci",
    "rates_ric",
    "relative_range",
    "knowledge_timeline",
    "control_thrust",
    "control_thrust_multi",
    "control_thrust_ric",
    "control_thrust_ric_multi",
    "quaternion_error",
    "rocket_ascent_diagnostics",
    "rocket_orbital_elements",
    "satellite_delta_v_remaining",
    "thrust_alignment_error",
]
ANIMATION_TYPE_OPTIONS = [
    "ground_track",
    "ground_track_multi",
    "ric_curv_prism_multi",
    "ric_prism_side_by_side",
]
MC_PARAMETER_CATEGORIES = {
    "Launch": [
        ("Launch Latitude", "rocket.initial_state.launch_lat_deg"),
        ("Launch Longitude", "rocket.initial_state.launch_lon_deg"),
        ("Launch Altitude", "rocket.initial_state.launch_alt_km"),
        ("Launch Azimuth", "rocket.initial_state.launch_azimuth_deg"),
    ],
    "Chaser Init": [
        ("Deploy Time", "chaser.initial_state.deploy_time_s"),
        ("Deploy dV X", "chaser.initial_state.deploy_dv_body_m_s[0]"),
        ("Deploy dV Y", "chaser.initial_state.deploy_dv_body_m_s[1]"),
        ("Deploy dV Z", "chaser.initial_state.deploy_dv_body_m_s[2]"),
        ("Relative State R", "chaser.initial_state.relative_to_target_ric.state[0]"),
        ("Relative State I", "chaser.initial_state.relative_to_target_ric.state[1]"),
        ("Relative State C", "chaser.initial_state.relative_to_target_ric.state[2]"),
        ("Relative State dR", "chaser.initial_state.relative_to_target_ric.state[3]"),
        ("Relative State dI", "chaser.initial_state.relative_to_target_ric.state[4]"),
        ("Relative State dC", "chaser.initial_state.relative_to_target_ric.state[5]"),
        ("RIC Rect R", "chaser.initial_state.relative_ric_rect[0]"),
        ("RIC Rect I", "chaser.initial_state.relative_ric_rect[1]"),
        ("RIC Rect C", "chaser.initial_state.relative_ric_rect[2]"),
        ("RIC Rect dR", "chaser.initial_state.relative_ric_rect[3]"),
        ("RIC Rect dI", "chaser.initial_state.relative_ric_rect[4]"),
        ("RIC Rect dC", "chaser.initial_state.relative_ric_rect[5]"),
        ("RIC Curv R", "chaser.initial_state.relative_ric_curv[0]"),
        ("RIC Curv I", "chaser.initial_state.relative_ric_curv[1]"),
        ("RIC Curv C", "chaser.initial_state.relative_ric_curv[2]"),
        ("RIC Curv dR", "chaser.initial_state.relative_ric_curv[3]"),
        ("RIC Curv dI", "chaser.initial_state.relative_ric_curv[4]"),
        ("RIC Curv dC", "chaser.initial_state.relative_ric_curv[5]"),
    ],
    "Target Orbit": [
        ("Semi-major Axis", "target.initial_state.coes.a_km"),
        ("Eccentricity", "target.initial_state.coes.ecc"),
        ("Inclination", "target.initial_state.coes.inc_deg"),
        ("RAAN", "target.initial_state.coes.raan_deg"),
        ("Arg Periapsis", "target.initial_state.coes.argp_deg"),
        ("True Anomaly", "target.initial_state.coes.true_anomaly_deg"),
    ],
    "Environment": [
        ("Solar Flux F10.7", "simulator.environment.atmosphere_env.solar_flux_f107"),
        ("Geomagnetic Ap", "simulator.environment.atmosphere_env.geomagnetic_ap"),
    ],
}

PARAMETER_FORM_SCHEMAS = {
    "OpenLoopPitchProgramGuidance": [
        {"key": "vertical_hold_s", "label": "Vertical Hold (s)", "kind": "float"},
        {"key": "pitch_start_s", "label": "Pitch Start (s)", "kind": "float"},
        {"key": "pitch_end_s", "label": "Pitch End (s)", "kind": "float"},
        {"key": "pitch_final_deg", "label": "Pitch Final (deg)", "kind": "float"},
        {"key": "max_throttle", "label": "Max Throttle", "kind": "float"},
        {"key": "min_throttle", "label": "Min Throttle", "kind": "float"},
    ],
    "ClosedLoopInsertionGuidance": [
        {"key": "target_altitude_km", "label": "Target Altitude (km)", "kind": "float"},
        {"key": "target_eccentricity_max", "label": "Target Max Ecc", "kind": "float"},
        {"key": "pitch_gain", "label": "Pitch Gain", "kind": "float"},
        {"key": "throttle_gain", "label": "Throttle Gain", "kind": "float"},
        {"key": "min_throttle", "label": "Min Throttle", "kind": "float"},
        {"key": "max_throttle", "label": "Max Throttle", "kind": "float"},
    ],
    "HoldAttitudeGuidance": [
        {"key": "throttle", "label": "Throttle", "kind": "float"},
        {"key": "attitude_quat_bn_cmd", "label": "Attitude Quaternion BN", "kind": "vector", "length": 4},
    ],
    "TVCSteeringGuidance": [
        {"key": "pass_through_attitude", "label": "Pass Through Attitude", "kind": "bool"},
    ],
    "MaxQThrottleLimiterGuidance": [
        {"key": "max_q_pa", "label": "Max Q (Pa)", "kind": "float"},
        {"key": "min_throttle", "label": "Min Throttle", "kind": "float"},
    ],
    "OrbitInsertionCutoffGuidance": [
        {"key": "min_cutoff_alt_km", "label": "Min Cutoff Alt (km)", "kind": "float"},
        {"key": "min_periapsis_alt_km", "label": "Min Periapsis Alt (km)", "kind": "float"},
        {"key": "apoapsis_margin_km", "label": "Apoapsis Margin (km)", "kind": "float"},
        {"key": "energy_margin_km2_s2", "label": "Energy Margin", "kind": "float"},
        {"key": "ecc_relax_factor", "label": "Ecc Relax Factor", "kind": "float"},
        {"key": "hard_escape_cutoff", "label": "Hard Escape Cutoff", "kind": "bool"},
        {"key": "near_escape_speed_margin_frac", "label": "Near Escape Margin", "kind": "float"},
    ],
    "HCWLQRController": [
        {"key": "mean_motion_rad_s", "label": "Mean Motion (rad/s)", "kind": "float"},
        {"key": "max_accel_km_s2", "label": "Max Accel (km/s^2)", "kind": "float"},
        {"key": "design_dt_s", "label": "Design dt (s)", "kind": "float"},
        {"key": "state_signs", "label": "State Signs", "kind": "vector", "length": 6},
        {"key": "q_weights", "label": "Q Weights", "kind": "vector", "length": 6},
        {"key": "r_weights", "label": "R Weights", "kind": "vector", "length": 3},
        {"key": "riccati_max_iter", "label": "Riccati Max Iter", "kind": "int"},
        {"key": "riccati_tol", "label": "Riccati Tolerance", "kind": "float"},
    ],
    "RelativeOrbitMPCController": [
        {"key": "max_accel_km_s2", "label": "Max Accel (km/s^2)", "kind": "float"},
        {"key": "horizon_steps", "label": "Horizon Steps", "kind": "int"},
        {"key": "step_dt_s", "label": "Step dt (s)", "kind": "float"},
        {"key": "gradient_method", "label": "Gradient Method", "kind": "choice", "options": ["spsa", "finite_difference"]},
        {"key": "max_iterations", "label": "Max Iterations", "kind": "int"},
        {"key": "target_rel_ric_rect", "label": "Target Rel RIC Rect", "kind": "vector", "length": 6},
        {"key": "q_weights", "label": "Q Weights", "kind": "vector", "length": 6},
        {"key": "terminal_weights", "label": "Terminal Weights", "kind": "vector", "length": 6},
        {"key": "r_weights", "label": "R Weights", "kind": "vector", "length": 3},
        {"key": "rd_weights", "label": "Rd Weights", "kind": "vector", "length": 3},
    ],
    "HCWRelativeOrbitMPCController": [
        {"key": "max_accel_km_s2", "label": "Max Accel (km/s^2)", "kind": "float"},
        {"key": "horizon_time_s", "label": "Horizon Time (s)", "kind": "float"},
        {"key": "default_model_dt_s", "label": "Default Model dt (s)", "kind": "float"},
        {"key": "model_dt_s", "label": "Model dt (s)", "kind": "optional_float"},
        {"key": "gradient_method", "label": "Gradient Method", "kind": "choice", "options": ["spsa", "finite_difference"]},
        {"key": "max_iterations", "label": "Max Iterations", "kind": "int"},
        {"key": "target_rel_ric_rect", "label": "Target Rel RIC Rect", "kind": "vector", "length": 6},
        {"key": "q_weights", "label": "Q Weights", "kind": "vector", "length": 6},
        {"key": "terminal_weights", "label": "Terminal Weights", "kind": "vector", "length": 6},
        {"key": "r_weights", "label": "R Weights", "kind": "vector", "length": 3},
        {"key": "rd_weights", "label": "Rd Weights", "kind": "vector", "length": 3},
    ],
    "QuaternionPDController": [
        {"key": "kp", "label": "Kp", "kind": "float"},
        {"key": "kd", "label": "Kd", "kind": "float"},
        {"key": "max_torque_nm", "label": "Max Torque (Nm)", "kind": "float"},
    ],
    "SurrogateSnapECIController": [
        {"key": "desired_attitude_quat_bn", "label": "Desired Quaternion BN", "kind": "vector", "length": 4},
        {"key": "cancel_rate_mag_rad_s2", "label": "Cancel Rate Mag (rad/s^2)", "kind": "float"},
        {"key": "rate_tolerance_rad_s", "label": "Rate Tolerance (rad/s)", "kind": "float"},
        {"key": "slew_time_180_s", "label": "180 deg Slew Time (s)", "kind": "float"},
        {"key": "pointing_sigma_deg", "label": "Pointing Sigma (deg)", "kind": "float"},
        {"key": "default_dt_s", "label": "Default dt (s)", "kind": "float"},
        {"key": "rng_seed", "label": "RNG Seed", "kind": "int"},
    ],
    "SurrogateSnapRICController": [
        {"key": "desired_attitude_quat_br", "label": "Desired Quaternion BR", "kind": "vector", "length": 4},
        {"key": "cancel_rate_mag_rad_s2", "label": "Cancel Rate Mag (rad/s^2)", "kind": "float"},
        {"key": "rate_tolerance_rad_s", "label": "Rate Tolerance (rad/s)", "kind": "float"},
        {"key": "slew_time_180_s", "label": "180 deg Slew Time (s)", "kind": "float"},
        {"key": "pointing_sigma_deg", "label": "Pointing Sigma (deg)", "kind": "float"},
        {"key": "default_dt_s", "label": "Default dt (s)", "kind": "float"},
        {"key": "rng_seed", "label": "RNG Seed", "kind": "int"},
    ],
    "RocketMissionModule": [
        {"key": "launch_mode", "label": "Launch Mode", "kind": "choice", "options": ["go_now", "go_when_possible", "wait_optimal_window"]},
        {"key": "orbital_goal", "label": "Orbital Goal", "kind": "choice", "options": ["pursuit", "predefined_orbit"]},
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "go_when_possible_margin_m_s", "label": "Go Margin (m/s)", "kind": "float"},
        {"key": "window_period_s", "label": "Window Period (s)", "kind": "float"},
        {"key": "window_open_duration_s", "label": "Window Open Duration (s)", "kind": "float"},
        {"key": "predef_target_alt_km", "label": "Predef Target Alt (km)", "kind": "float"},
        {"key": "predef_target_ecc", "label": "Predef Target Ecc", "kind": "float"},
    ],
    "SatelliteMissionModule": [
        {"key": "orbital_mode", "label": "Orbital Mode", "kind": "choice", "options": ["coast", "pursuit_knowledge", "evade_knowledge", "pursuit_blind", "evade_blind"]},
        {"key": "attitude_mode", "label": "Attitude Mode", "kind": "choice", "options": ["hold_eci", "hold_ric", "spotlight", "sun_track", "pursuit", "evade", "sensing"]},
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "max_accel_km_s2", "label": "Max Accel (km/s^2)", "kind": "float"},
        {"key": "blind_direction_eci", "label": "Blind Direction ECI", "kind": "vector", "length": 3},
        {"key": "hold_quat_bn", "label": "Hold Quaternion BN", "kind": "vector", "length": 4},
        {"key": "hold_quat_br", "label": "Hold Quaternion BR", "kind": "vector", "length": 4},
        {"key": "boresight_body", "label": "Boresight Body", "kind": "vector", "length": 3},
        {"key": "spotlight_lat_deg", "label": "Spotlight Lat (deg)", "kind": "float"},
        {"key": "spotlight_lon_deg", "label": "Spotlight Lon (deg)", "kind": "float"},
        {"key": "spotlight_alt_km", "label": "Spotlight Alt (km)", "kind": "float"},
        {"key": "spotlight_ric_direction", "label": "Spotlight RIC Direction", "kind": "vector", "length": 3},
        {"key": "use_knowledge_for_targeting", "label": "Use Knowledge For Targeting", "kind": "bool"},
    ],
    "PursuitMissionStrategy": [
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "use_knowledge_for_targeting", "label": "Use Knowledge", "kind": "bool"},
        {"key": "max_accel_km_s2", "label": "Fallback Max Accel (km/s^2)", "kind": "float"},
        {"key": "blind_direction_eci", "label": "Blind Direction ECI", "kind": "vector", "length": 3},
        {"key": "align_to_thrust", "label": "Align To Thrust", "kind": "bool"},
    ],
    "EvadeMissionStrategy": [
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "use_knowledge_for_targeting", "label": "Use Knowledge", "kind": "bool"},
        {"key": "max_accel_km_s2", "label": "Fallback Max Accel (km/s^2)", "kind": "float"},
        {"key": "blind_direction_eci", "label": "Blind Direction ECI", "kind": "vector", "length": 3},
        {"key": "align_to_thrust", "label": "Align To Thrust", "kind": "bool"},
    ],
    "HoldMissionStrategy": [
        {"key": "attitude_mode", "label": "Attitude Mode", "kind": "choice", "options": ["hold_eci", "hold_ric", "sun_track", "spotlight", "sensing"]},
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "use_knowledge_for_targeting", "label": "Use Knowledge", "kind": "bool"},
        {"key": "hold_quat_bn", "label": "Hold Quaternion BN", "kind": "vector", "length": 4},
        {"key": "hold_quat_br", "label": "Hold Quaternion BR", "kind": "vector", "length": 4},
        {"key": "boresight_body", "label": "Boresight Body", "kind": "vector", "length": 3},
        {"key": "spotlight_lat_deg", "label": "Spotlight Lat (deg)", "kind": "float"},
        {"key": "spotlight_lon_deg", "label": "Spotlight Lon (deg)", "kind": "float"},
        {"key": "spotlight_alt_km", "label": "Spotlight Alt (km)", "kind": "float"},
        {"key": "spotlight_ric_direction", "label": "Spotlight RIC Direction", "kind": "vector", "length": 3},
    ],
    "MissionExecutiveStrategy": [
        {"key": "initial_mode", "label": "Initial Mode", "kind": "string"},
        {"key": "modes", "label": "Modes YAML", "kind": "yaml"},
        {"key": "transitions", "label": "Transitions YAML", "kind": "yaml"},
    ],
    "DesiredStateMissionStrategy": [
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "desired_state_source", "label": "Desired State Source", "kind": "choice", "options": ["target", "explicit"]},
        {"key": "use_knowledge_for_targeting", "label": "Use Knowledge", "kind": "bool"},
        {"key": "desired_position_eci_km", "label": "Desired Position ECI", "kind": "vector", "length": 3},
        {"key": "desired_velocity_eci_km_s", "label": "Desired Velocity ECI", "kind": "vector", "length": 3},
        {"key": "align_to_thrust", "label": "Align To Thrust", "kind": "bool"},
    ],
    "StationKeepMissionStrategy": [
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "use_knowledge_for_targeting", "label": "Use Knowledge", "kind": "bool"},
        {"key": "desired_relative_ric_rect", "label": "Desired RIC Rect State", "kind": "vector", "length": 6},
        {"key": "kp_pos", "label": "Kp Pos", "kind": "float"},
        {"key": "kd_vel", "label": "Kd Vel", "kind": "float"},
        {"key": "max_accel_km_s2", "label": "Max Accel (km/s^2)", "kind": "float"},
        {"key": "align_to_thrust", "label": "Align To Thrust", "kind": "bool"},
    ],
    "InspectMissionStrategy": [
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "use_knowledge_for_targeting", "label": "Use Knowledge", "kind": "bool"},
        {"key": "desired_relative_ric_rect", "label": "Desired RIC Rect State", "kind": "vector", "length": 6},
        {"key": "boresight_body", "label": "Boresight Body", "kind": "vector", "length": 3},
        {"key": "kp_pos", "label": "Kp Pos", "kind": "float"},
        {"key": "kd_vel", "label": "Kd Vel", "kind": "float"},
        {"key": "max_accel_km_s2", "label": "Max Accel (km/s^2)", "kind": "float"},
        {"key": "align_to_thrust", "label": "Align To Thrust", "kind": "bool"},
    ],
    "DefensiveMissionStrategy": [
        {"key": "chaser_id", "label": "Chaser ID", "kind": "string"},
        {"key": "defense_mode", "label": "Defense Mode", "kind": "choice", "options": ["fixed_ric_axis", "away_from_chaser"]},
        {"key": "axis_mode", "label": "Axis Mode", "kind": "choice", "options": ["+R", "-R", "+I", "-I", "+C", "-C"]},
        {"key": "burn_accel_km_s2", "label": "Burn Accel (km/s^2)", "kind": "float"},
        {"key": "require_finite_knowledge", "label": "Require Finite Knowledge", "kind": "bool"},
        {"key": "allow_truth_fallback", "label": "Allow Truth Fallback", "kind": "bool"},
        {"key": "align_to_thrust", "label": "Align To Thrust", "kind": "bool"},
    ],
    "SafeHoldMissionStrategy": [
        {"key": "attitude_mode", "label": "Attitude Mode", "kind": "choice", "options": ["hold_current", "hold_eci", "sun_track"]},
        {"key": "hold_quat_bn", "label": "Hold Quaternion BN", "kind": "vector", "length": 4},
        {"key": "boresight_body", "label": "Boresight Body", "kind": "vector", "length": 3},
    ],
    "RocketMissionStrategy": [
        {"key": "launch_mode", "label": "Launch Mode", "kind": "choice", "options": ["go_now", "go_when_possible", "wait_optimal_window"]},
        {"key": "orbital_goal", "label": "Orbital Goal", "kind": "choice", "options": ["pursuit", "predefined_orbit"]},
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "go_when_possible_margin_m_s", "label": "Go Margin (m/s)", "kind": "float"},
        {"key": "window_period_s", "label": "Window Period (s)", "kind": "float"},
        {"key": "window_open_duration_s", "label": "Window Open Duration (s)", "kind": "float"},
        {"key": "predef_target_alt_km", "label": "Target Alt (km)", "kind": "float"},
        {"key": "predef_target_ecc", "label": "Target Ecc", "kind": "float"},
    ],
    "RocketPursuitMissionStrategy": [
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "align_to_thrust", "label": "Align To Thrust", "kind": "bool"},
    ],
    "RocketPredefinedOrbitMissionStrategy": [
        {"key": "predef_target_alt_km", "label": "Target Alt (km)", "kind": "float"},
        {"key": "predef_target_ecc", "label": "Target Ecc", "kind": "float"},
        {"key": "align_to_thrust", "label": "Align To Thrust", "kind": "bool"},
    ],
    "ControllerPointingExecution": [
        {"key": "align_thruster_to_thrust", "label": "Align Thruster To Thrust", "kind": "bool"},
        {"key": "thruster_direction_body", "label": "Thruster Direction Body", "kind": "vector", "length": 3},
        {"key": "require_attitude_alignment", "label": "Require Alignment", "kind": "bool"},
        {"key": "alignment_tolerance_deg", "label": "Alignment Tol (deg)", "kind": "float"},
        {"key": "use_strategy_fallback_thrust", "label": "Use Strategy Fallback", "kind": "bool"},
        {"key": "detumble_enter_rate_rad_s", "label": "Detumble Enter Rate", "kind": "optional_float"},
        {"key": "detumble_exit_rate_rad_s", "label": "Detumble Exit Rate", "kind": "optional_float"},
        {"key": "detumble_mode_name", "label": "Detumble Mode", "kind": "string"},
        {"key": "nominal_mode_name", "label": "Nominal Mode", "kind": "string"},
    ],
    "PredictiveBurnExecution": [
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "use_knowledge_for_targeting", "label": "Use Knowledge", "kind": "bool"},
        {"key": "lead_time_s", "label": "Lead Time (s)", "kind": "float"},
        {"key": "predict_dt_s", "label": "Predict dt (s)", "kind": "float"},
        {"key": "thruster_direction_body", "label": "Thruster Direction Body", "kind": "vector", "length": 3},
        {"key": "alignment_tolerance_deg", "label": "Alignment Tol (deg)", "kind": "float"},
        {"key": "min_burn_accel_km_s2", "label": "Min Burn Accel", "kind": "float"},
        {"key": "mu_km3_s2", "label": "Mu (km^3/s^2)", "kind": "float"},
        {"key": "orbit_controller_budget_ms", "label": "Orbit Budget (ms)", "kind": "float"},
        {"key": "attitude_controller_budget_ms", "label": "Attitude Budget (ms)", "kind": "float"},
        {"key": "planning_period_s", "label": "Planning Period (s)", "kind": "optional_float"},
        {"key": "skip_orbit_planning_in_detumble_mode", "label": "Skip In Detumble", "kind": "bool"},
        {"key": "detumble_enter_rate_rad_s", "label": "Detumble Enter Rate", "kind": "optional_float"},
        {"key": "detumble_exit_rate_rad_s", "label": "Detumble Exit Rate", "kind": "optional_float"},
        {"key": "detumble_mode_name", "label": "Detumble Mode", "kind": "string"},
        {"key": "nominal_mode_name", "label": "Nominal Mode", "kind": "string"},
    ],
    "IntegratedCommandExecution": [
        {"key": "require_attitude_alignment", "label": "Require Alignment", "kind": "bool"},
        {"key": "thruster_direction_body", "label": "Thruster Direction Body", "kind": "vector", "length": 3},
        {"key": "alignment_tolerance_deg", "label": "Alignment Tol (deg)", "kind": "float"},
        {"key": "min_burn_accel_km_s2", "label": "Min Burn Accel", "kind": "float"},
        {"key": "orbit_controller_budget_ms", "label": "Orbit Budget (ms)", "kind": "float"},
        {"key": "attitude_controller_budget_ms", "label": "Attitude Budget (ms)", "kind": "float"},
    ],
    "BudgetedEndStateExecution": [
        {"key": "strategy", "label": "Maneuver Strategy", "kind": "choice", "options": ["thrust_limited", "burn_all", "attitude_only"]},
        {"key": "max_thrust_n", "label": "Max Thrust (N)", "kind": "float"},
        {"key": "min_thrust_n", "label": "Min Thrust (N)", "kind": "float"},
        {"key": "burn_dt_s", "label": "Burn dt (s)", "kind": "float"},
        {"key": "available_delta_v_km_s", "label": "Available dV (km/s)", "kind": "float"},
        {"key": "require_attitude_alignment", "label": "Require Alignment", "kind": "bool"},
        {"key": "thruster_position_body_m", "label": "Thruster Position Body", "kind": "vector", "length": 3},
        {"key": "thruster_direction_body", "label": "Thruster Direction Body", "kind": "vector", "length": 3},
        {"key": "alignment_tolerance_deg", "label": "Alignment Tol (deg)", "kind": "float"},
        {"key": "terminate_on_velocity_tolerance_km_s", "label": "Velocity Tol", "kind": "float"},
    ],
    "DirectIntegratedExecution": [
        {"key": "align_thruster_to_thrust", "label": "Align Thruster To Thrust", "kind": "bool"},
        {"key": "thruster_direction_body", "label": "Thruster Direction Body", "kind": "vector", "length": 3},
        {"key": "use_strategy_fallback_thrust", "label": "Use Strategy Fallback", "kind": "bool"},
        {"key": "use_orbit_controller", "label": "Use Orbit Controller", "kind": "bool"},
        {"key": "orbit_controller_budget_ms", "label": "Orbit Budget (ms)", "kind": "float"},
        {"key": "attitude_controller_budget_ms", "label": "Attitude Budget (ms)", "kind": "float"},
    ],
    "ImpulsiveExecution": [
        {"key": "align_thruster_to_thrust", "label": "Align Thruster To Thrust", "kind": "bool"},
        {"key": "thruster_direction_body", "label": "Thruster Direction Body", "kind": "vector", "length": 3},
        {"key": "require_attitude_alignment", "label": "Require Alignment", "kind": "bool"},
        {"key": "alignment_tolerance_deg", "label": "Alignment Tol (deg)", "kind": "float"},
        {"key": "use_strategy_fallback_thrust", "label": "Use Strategy Fallback", "kind": "bool"},
        {"key": "pulse_period_s", "label": "Pulse Period (s)", "kind": "float"},
        {"key": "pulse_width_s", "label": "Pulse Width (s)", "kind": "float"},
        {"key": "pulse_phase_s", "label": "Pulse Phase (s)", "kind": "float"},
        {"key": "min_burn_accel_km_s2", "label": "Min Burn Accel", "kind": "float"},
        {"key": "orbit_controller_budget_ms", "label": "Orbit Budget (ms)", "kind": "float"},
        {"key": "attitude_controller_budget_ms", "label": "Attitude Budget (ms)", "kind": "float"},
    ],
    "SafeHoldExecution": [
        {"key": "attitude_controller_budget_ms", "label": "Attitude Budget (ms)", "kind": "float"},
    ],
    "RocketGoNowExecution": [],
    "RocketGoWhenPossibleExecution": [
        {"key": "target_id", "label": "Target ID", "kind": "string"},
        {"key": "go_when_possible_margin_m_s", "label": "Go Margin (m/s)", "kind": "float"},
    ],
    "RocketWaitOptimalExecution": [
        {"key": "window_period_s", "label": "Window Period (s)", "kind": "float"},
        {"key": "window_open_duration_s", "label": "Window Open Duration (s)", "kind": "float"},
    ],
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.repo_root = get_repo_root()
        self.loaded_config_path = get_default_config_path()
        self.current_config = load_config(self.loaded_config_path)
        self.mc_variations: list[dict] = []
        self._collapse_validation_on_startup = True
        self.process: QProcess | None = None
        self.preview_image_path: Path | None = None
        self.preview_zoom_factor = 1.0
        self.preview_fit_to_window = True
        self.preview_drag_active = False
        self.preview_drag_last_pos = None
        self.results_output_dir: Path | None = None
        self.preview_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.rocket_guidance_modifiers_config: list[dict] = []
        self.is_dirty = False
        self._suppress_dirty_tracking = False
        self._suppress_config_selector_load = False
        self._build_ui()
        self._connect_dirty_tracking()
        self.mc_enabled_check.toggled.connect(self._refresh_outputs_mode_ui)
        self.orbit_substep_enabled_check.toggled.connect(self._refresh_substep_visibility)
        self.attitude_substep_enabled_check.toggled.connect(self._refresh_substep_visibility)
        for combo in (
            self.target_strategy_combo,
            self.target_execution_combo,
            self.target_orbit_control_combo,
            self.target_attitude_control_combo,
            self.chaser_strategy_combo,
            self.chaser_execution_combo,
            self.chaser_orbit_control_combo,
            self.chaser_attitude_control_combo,
            self.rocket_strategy_combo,
            self.rocket_execution_combo,
            self.rocket_base_guidance_combo,
        ):
            combo.currentIndexChanged.connect(self._on_mc_catalog_source_changed)
        self._load_config_into_widgets(self.current_config)
        self._refresh_yaml()
        self._refresh_validation_state()
        self._refresh_output_files()
        self._set_dirty(False)
        self._update_window_title()

    def _build_ui(self) -> None:
        self.setWindowTitle("NonCooperativeRPO Operator Console")
        self.resize(1360, 900)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(4)
        self.config_selector = QComboBox()
        for path in list_available_configs():
            self.config_selector.addItem(str(path.relative_to(self.repo_root)), str(path))
        self.config_selector.currentIndexChanged.connect(self._on_config_selected)
        current_rel = str(self.loaded_config_path.relative_to(self.repo_root))
        idx = self.config_selector.findText(current_rel)
        if idx >= 0:
            self.config_selector.setCurrentIndex(idx)
        top_bar.addWidget(QLabel("Base Config"))
        top_bar.addWidget(self.config_selector, 1)

        self.new_button = QPushButton("New")
        self.new_button.clicked.connect(self._on_new)
        top_bar.addWidget(self.new_button)

        self.open_button = QPushButton("Open...")
        self.open_button.clicked.connect(self._on_open_file)
        top_bar.addWidget(self.open_button)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run)
        top_bar.addWidget(self.run_button)

        root.addLayout(top_bar)

        save_bar = QHBoxLayout()
        save_bar.setContentsMargins(0, 0, 0, 0)
        save_bar.setSpacing(4)
        self.save_path_edit = QLineEdit(str(self.repo_root / "configs" / "gui_working.yaml"))
        save_bar.addWidget(QLabel("Save Path"))
        save_bar.addWidget(self.save_path_edit, 1)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._on_save)
        save_bar.addWidget(self.save_button)

        self.save_as_button = QPushButton("Save As...")
        self.save_as_button.clicked.connect(self._on_save_as)
        save_bar.addWidget(self.save_as_button)

        root.addLayout(save_bar)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)
        nav_layout.addWidget(QLabel("Scenario Tree"))
        self.navigation_tree = self._build_navigation_tree()
        self.navigation_tree.itemSelectionChanged.connect(self._on_navigation_selected)
        nav_layout.addWidget(self.navigation_tree, 1)
        splitter.addWidget(nav_container)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(4)
        validation_box = QGroupBox("Validation")
        validation_layout = QVBoxLayout(validation_box)
        validation_layout.setContentsMargins(6, 6, 6, 6)
        validation_layout.setSpacing(4)
        self.validation_toggle = QPushButton("Show Details")
        self.validation_toggle.clicked.connect(self._toggle_validation_panel)
        validation_layout.addWidget(self.validation_toggle)
        self.validation_label = QLabel("No validation issues.")
        self.validation_label.setWordWrap(True)
        validation_layout.addWidget(self.validation_label)
        self.validation_panel = QPlainTextEdit()
        self.validation_panel.setReadOnly(True)
        self.validation_panel.setMaximumHeight(120)
        self.validation_panel.hide()
        validation_layout.addWidget(self.validation_panel)
        workspace_layout.addWidget(validation_box)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._sync_navigation_to_tab)
        workspace_layout.addWidget(self.tabs, 1)

        self.tabs.addTab(self._build_scenario_tab(), "Scenario")
        self.tabs.addTab(self._build_objects_tab(), "Objects")
        self.tabs.addTab(self._build_monte_carlo_tab(), "Monte Carlo")
        self.tabs.addTab(self._build_outputs_tab(), "Outputs")
        self.tabs.addTab(self._build_yaml_tab(), "Advanced YAML")
        self.tabs.addTab(self._build_results_tab(), "Results")
        self.navigation_tree.setCurrentItem(self.navigation_tree.topLevelItem(0))
        splitter.addWidget(workspace)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 1120])

        status = QStatusBar(self)
        self.setStatusBar(status)
        status.showMessage("Ready.")

    def _build_navigation_tree(self) -> QTreeWidget:
        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        scenario_item = QTreeWidgetItem(["Scenario"])
        scenario_item.setData(0, Qt.UserRole, 0)
        scenario_item.addChild(self._nav_item("Simulator", 0))

        objects_item = QTreeWidgetItem(["Objects"])
        objects_item.setData(0, Qt.UserRole, 1)
        objects_item.addChild(self._nav_item("Target", 1))
        objects_item.addChild(self._nav_item("Chaser", 1))
        objects_item.addChild(self._nav_item("Rocket", 1))

        mc_item = QTreeWidgetItem(["Monte Carlo"])
        mc_item.setData(0, Qt.UserRole, 2)
        mc_item.addChild(self._nav_item("Execution", 2))
        mc_item.addChild(self._nav_item("Variations", 2))

        outputs_item = QTreeWidgetItem(["Outputs"])
        outputs_item.setData(0, Qt.UserRole, 3)
        outputs_item.addChild(self._nav_item("Stats", 3))
        outputs_item.addChild(self._nav_item("Plots", 3))
        outputs_item.addChild(self._nav_item("Animations", 3))

        yaml_item = QTreeWidgetItem(["Advanced YAML"])
        yaml_item.setData(0, Qt.UserRole, 4)

        results_item = QTreeWidgetItem(["Results"])
        results_item.setData(0, Qt.UserRole, 5)
        results_item.addChild(self._nav_item("Console", 5))
        results_item.addChild(self._nav_item("Summary", 5))
        results_item.addChild(self._nav_item("Artifacts", 5))

        tree.addTopLevelItem(scenario_item)
        tree.addTopLevelItem(objects_item)
        tree.addTopLevelItem(mc_item)
        tree.addTopLevelItem(outputs_item)
        tree.addTopLevelItem(yaml_item)
        tree.addTopLevelItem(results_item)
        tree.expandAll()
        return tree

    def _nav_item(self, label: str, tab_index: int) -> QTreeWidgetItem:
        item = QTreeWidgetItem([label])
        item.setData(0, Qt.UserRole, tab_index)
        return item

    def _connect_dirty_tracking(self) -> None:
        line_edits = [
            self.scenario_name_edit,
            self.output_dir_edit,
            self.reference_object_edit,
            self.mc_baseline_summary_json,
            self.save_path_edit,
            self.yaml_editor,
        ]
        for widget in line_edits:
            signal = getattr(widget, "textChanged", None)
            if signal is not None:
                signal.connect(self._mark_dirty)
        combo_boxes = [
            self.output_mode_combo,
            self.chaser_init_mode,
            self.target_preset,
            self.chaser_preset,
            self.rocket_preset,
            self.target_strategy_combo,
            self.target_execution_combo,
            self.target_orbit_control_combo,
            self.target_attitude_control_combo,
            self.chaser_strategy_combo,
            self.chaser_execution_combo,
            self.chaser_orbit_control_combo,
            self.chaser_attitude_control_combo,
            self.rocket_strategy_combo,
            self.rocket_execution_combo,
            self.rocket_base_guidance_combo,
        ]
        for widget in combo_boxes:
            widget.currentIndexChanged.connect(self._mark_dirty)
        check_boxes = [
            self.mc_enabled_check,
            self.mc_parallel_check,
            self.orbit_substep_enabled_check,
            self.attitude_substep_enabled_check,
            self.attitude_enabled_check,
            self.orbit_j2_check,
            self.orbit_j3_check,
            self.orbit_j4_check,
            self.orbit_drag_check,
            self.orbit_srp_check,
            self.orbit_moon_check,
            self.orbit_sun_check,
            self.att_gg_check,
            self.att_magnetic_check,
            self.att_drag_check,
            self.att_srp_check,
            self.target_enabled,
            self.chaser_enabled,
            self.rocket_enabled,
            self.stats_enabled,
            self.stats_print_summary,
            self.stats_save_json,
            self.stats_save_csv,
            self.plots_enabled,
            self.mc_save_iteration_summaries,
            self.mc_save_aggregate_summary,
            self.mc_save_histograms,
            self.mc_display_histograms,
            self.mc_save_ops_dashboard,
            self.mc_display_ops_dashboard,
            self.mc_save_raw_runs,
            self.mc_require_rocket_insertion,
        ]
        for widget in check_boxes:
            widget.toggled.connect(self._mark_dirty)
        for widget in self.figure_id_checks.values():
            widget.toggled.connect(self._mark_dirty)
        for widget in self.animation_type_checks.values():
            widget.toggled.connect(self._mark_dirty)
        spin_boxes = [
            self.duration_spin,
            self.dt_spin,
            self.orbit_substep_spin,
            self.attitude_substep_spin,
            self.mc_iterations_spin,
            self.mc_workers_spin,
            self.mc_base_seed_spin,
            self.target_dry_mass,
            self.target_fuel_mass,
            self.target_a,
            self.target_ecc,
            self.target_inc,
            self.target_raan,
            self.target_argp,
            self.target_ta,
            self.chaser_dry_mass,
            self.chaser_fuel_mass,
            self.chaser_deploy_time,
            self.rocket_payload,
            self.rocket_launch_lat,
            self.rocket_launch_lon,
            self.rocket_launch_alt,
            self.rocket_launch_az,
            self.plots_dpi,
            self.animation_fps_spin,
            self.animation_speed_multiple_spin,
            self.animation_frame_stride_spin,
            self.mc_gate_min_closest_approach,
            self.mc_gate_max_duration,
            self.mc_gate_max_total_dv,
            self.mc_gate_max_guardrail_events,
        ]
        spin_boxes.extend(self.chaser_init_values)
        for widget in spin_boxes:
            widget.valueChanged.connect(self._mark_dirty)

    def _build_scenario_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        self.scenario_name_edit = QLineEdit()
        self.scenario_name_edit.setMinimumWidth(320)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.001, 1.0e9)
        self.duration_spin.setDecimals(3)
        self.duration_spin.setValue(3600.0)
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.000001, 1.0e6)
        self.dt_spin.setDecimals(6)
        self.dt_spin.setValue(1.0)
        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItems(OUTPUT_MODES)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setMinimumWidth(320)
        self.orbit_substep_enabled_check = QCheckBox("Orbit Substep")
        self.orbit_substep_spin = self._make_free_spinbox()
        self.attitude_substep_enabled_check = QCheckBox("Attitude Substep")
        self.attitude_substep_spin = self._make_free_spinbox()
        self.attitude_enabled_check = QCheckBox("Enable Attitude Dynamics")
        self.orbit_j2_check = QCheckBox("J2")
        self.orbit_j3_check = QCheckBox("J3")
        self.orbit_j4_check = QCheckBox("J4")
        self.orbit_drag_check = QCheckBox("Drag")
        self.orbit_srp_check = QCheckBox("SRP")
        self.orbit_moon_check = QCheckBox("Third-Body Moon")
        self.orbit_sun_check = QCheckBox("Third-Body Sun")
        self.att_gg_check = QCheckBox("Gravity Gradient")
        self.att_magnetic_check = QCheckBox("Magnetic")
        self.att_drag_check = QCheckBox("Drag Torque")
        self.att_srp_check = QCheckBox("SRP Torque")

        layout.addRow("Scenario Name", self.scenario_name_edit)
        layout.addRow("Duration (s)", self.duration_spin)
        layout.addRow("dt (s)", self.dt_spin)
        layout.addRow("Output Mode", self.output_mode_combo)
        layout.addRow("Output Directory", self.output_dir_edit)
        orbit_substep_row = QWidget()
        orbit_substep_layout = QHBoxLayout(orbit_substep_row)
        orbit_substep_layout.setContentsMargins(0, 0, 0, 0)
        orbit_substep_layout.setSpacing(4)
        orbit_substep_layout.addWidget(self.orbit_substep_enabled_check)
        orbit_substep_layout.addWidget(self.orbit_substep_spin)
        orbit_substep_layout.addStretch(1)
        layout.addRow(orbit_substep_row)
        attitude_substep_row = QWidget()
        attitude_substep_layout = QHBoxLayout(attitude_substep_row)
        attitude_substep_layout.setContentsMargins(0, 0, 0, 0)
        attitude_substep_layout.setSpacing(4)
        attitude_substep_layout.addWidget(self.attitude_substep_enabled_check)
        attitude_substep_layout.addWidget(self.attitude_substep_spin)
        attitude_substep_layout.addStretch(1)
        layout.addRow(attitude_substep_row)
        layout.addRow(self.attitude_enabled_check)
        orbit_perturbations = QWidget()
        orbit_perturbations_layout = QGridLayout(orbit_perturbations)
        orbit_perturbations_layout.setContentsMargins(0, 0, 0, 0)
        orbit_perturbations_layout.addWidget(self.orbit_j2_check, 0, 0)
        orbit_perturbations_layout.addWidget(self.orbit_j3_check, 0, 1)
        orbit_perturbations_layout.addWidget(self.orbit_j4_check, 0, 2)
        orbit_perturbations_layout.addWidget(self.orbit_drag_check, 1, 0)
        orbit_perturbations_layout.addWidget(self.orbit_srp_check, 1, 1)
        orbit_perturbations_layout.addWidget(self.orbit_moon_check, 2, 0)
        orbit_perturbations_layout.addWidget(self.orbit_sun_check, 2, 1)
        layout.addRow("Orbital Perturbations", orbit_perturbations)
        attitude_disturbances = QWidget()
        attitude_disturbances_layout = QGridLayout(attitude_disturbances)
        attitude_disturbances_layout.setContentsMargins(0, 0, 0, 0)
        attitude_disturbances_layout.addWidget(self.att_gg_check, 0, 0)
        attitude_disturbances_layout.addWidget(self.att_magnetic_check, 0, 1)
        attitude_disturbances_layout.addWidget(self.att_drag_check, 1, 0)
        attitude_disturbances_layout.addWidget(self.att_srp_check, 1, 1)
        layout.addRow("Attitude Disturbances", attitude_disturbances)
        return tab

    def _build_monte_carlo_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.mc_enabled_check = QCheckBox("Enable Monte Carlo")
        self.mc_iterations_spin = QSpinBox()
        self.mc_iterations_spin.setRange(1, 1000000)
        self.mc_parallel_check = QCheckBox("Parallel")
        self.mc_workers_spin = QSpinBox()
        self.mc_workers_spin.setRange(0, 1024)
        self.mc_base_seed_spin = QSpinBox()
        self.mc_base_seed_spin.setRange(0, 2**31 - 1)
        execution_row = QHBoxLayout()
        execution_row.addWidget(self.mc_enabled_check)
        execution_row.addWidget(QLabel("Iterations"))
        execution_row.addWidget(self.mc_iterations_spin)
        execution_row.addSpacing(12)
        execution_row.addWidget(self.mc_parallel_check)
        execution_row.addWidget(QLabel("Workers (0=auto)"))
        execution_row.addWidget(self.mc_workers_spin)
        execution_row.addSpacing(12)
        execution_row.addWidget(QLabel("Base Seed"))
        execution_row.addWidget(self.mc_base_seed_spin)
        execution_row.addStretch(1)
        layout.addLayout(execution_row)

        variations_split = QHBoxLayout()
        group_box_style = (
            "QGroupBox {"
            " font-weight: normal;"
            " margin-top: 18px;"
            "}"
            "QGroupBox::title {"
            " subcontrol-origin: margin;"
            " subcontrol-position: top left;"
            " padding: 0 4px;"
            " top: -3px;"
            "}"
        )
        variations_box = QGroupBox("Variations")
        variations_box.setStyleSheet(group_box_style)
        variations_box_layout = QVBoxLayout(variations_box)
        self.mc_variations_list = QListWidget()
        self.mc_variations_list.currentRowChanged.connect(self._on_mc_variation_selected)
        variations_box_layout.addWidget(self.mc_variations_list)
        variations_split.addWidget(variations_box, 2)

        editor_box = QGroupBox("Variation Editor")
        editor_box.setStyleSheet(group_box_style)
        editor_box_layout = QHBoxLayout(editor_box)
        editor_box_layout.setContentsMargins(8, 16, 8, 8)
        editor_box_layout.setSpacing(4)
        editor_form = QFormLayout()
        editor_form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        editor_box_layout.addLayout(editor_form, 0)
        button_column = QVBoxLayout()
        button_column.setContentsMargins(0, 0, 0, 0)
        button_column.setSpacing(4)
        self.mc_new_variation_button = QPushButton("New")
        self.mc_new_variation_button.clicked.connect(self._clear_mc_variation_editor)
        button_column.addWidget(self.mc_new_variation_button)
        self.mc_add_update_variation_button = QPushButton("Add / Update")
        self.mc_add_update_variation_button.clicked.connect(self._on_add_or_update_mc_variation)
        button_column.addWidget(self.mc_add_update_variation_button)
        self.mc_remove_variation_button = QPushButton("Remove")
        self.mc_remove_variation_button.clicked.connect(self._on_remove_mc_variation)
        button_column.addWidget(self.mc_remove_variation_button)
        button_column.addStretch(1)
        editor_box_layout.addLayout(button_column)
        editor_box_layout.addStretch(1)
        self.mc_variation_form = editor_form
        self.mc_category_combo = QComboBox()
        self.mc_category_combo.currentIndexChanged.connect(self._refresh_mc_parameter_options)
        editor_form.addRow("Category", self.mc_category_combo)

        self.mc_parameter_combo = QComboBox()
        editor_form.addRow("Parameter", self.mc_parameter_combo)

        self.mc_custom_path_edit = QLineEdit()
        self.mc_custom_path_edit.setPlaceholderText("simulator.environment.atmosphere_env.solar_flux_f107")
        editor_form.addRow("Custom Path", self.mc_custom_path_edit)

        self.mc_mode_combo = QComboBox()
        self.mc_mode_combo.addItems(["choice", "uniform", "normal"])
        self.mc_mode_combo.currentTextChanged.connect(self._refresh_mc_mode_ui)
        editor_form.addRow("Mode", self.mc_mode_combo)

        self.mc_mode_stack = QStackedWidget()
        choice_page = QWidget()
        choice_form = QFormLayout(choice_page)
        self.mc_choice_options_edit = QLineEdit()
        self.mc_choice_options_edit.setPlaceholderText("0.5, 1.0, 1.5")
        choice_form.addRow("Options", self.mc_choice_options_edit)
        self.mc_mode_stack.addWidget(choice_page)

        uniform_page = QWidget()
        uniform_form = QFormLayout(uniform_page)
        self.mc_uniform_low_spin = self._make_free_spinbox()
        self.mc_uniform_high_spin = self._make_free_spinbox()
        uniform_form.addRow("Low", self.mc_uniform_low_spin)
        uniform_form.addRow("High", self.mc_uniform_high_spin)
        self.mc_mode_stack.addWidget(uniform_page)

        normal_page = QWidget()
        normal_form = QFormLayout(normal_page)
        self.mc_normal_mean_spin = self._make_free_spinbox()
        self.mc_normal_std_spin = self._make_free_spinbox()
        normal_form.addRow("Mean", self.mc_normal_mean_spin)
        normal_form.addRow("Std", self.mc_normal_std_spin)
        self.mc_mode_stack.addWidget(normal_page)
        editor_form.addRow("Settings", self.mc_mode_stack)

        variations_split.addWidget(editor_box, 3)
        layout.addLayout(variations_split, 1)
        self._rebuild_mc_category_combo()
        self._refresh_mc_parameter_options()
        self._refresh_mc_mode_ui()
        return tab

    def _build_objects_tab(self) -> QWidget:
        tab = QWidget()
        layout = QGridLayout(tab)

        target_box = QGroupBox("Target")
        target_form = QFormLayout(target_box)
        self.target_enabled = QCheckBox("Enabled")
        self.target_preset = QComboBox()
        self._configure_compact_combo(self.target_preset)
        self.target_preset.addItems(SATELLITE_PRESET_OPTIONS)
        self.target_dry_mass = QDoubleSpinBox()
        self.target_dry_mass.setRange(0.0, 1.0e9)
        self.target_dry_mass.setDecimals(3)
        self.target_fuel_mass = QDoubleSpinBox()
        self.target_fuel_mass.setRange(0.0, 1.0e9)
        self.target_fuel_mass.setDecimals(3)
        self.target_a = QDoubleSpinBox()
        self.target_a.setRange(1.0, 1.0e9)
        self.target_a.setDecimals(3)
        self.target_ecc = QDoubleSpinBox()
        self.target_ecc.setRange(0.0, 100.0)
        self.target_ecc.setDecimals(6)
        self.target_inc = QDoubleSpinBox()
        self.target_inc.setRange(-360.0, 360.0)
        self.target_inc.setDecimals(3)
        self.target_raan = QDoubleSpinBox()
        self.target_raan.setRange(-360.0, 360.0)
        self.target_raan.setDecimals(3)
        self.target_argp = QDoubleSpinBox()
        self.target_argp.setRange(-360.0, 360.0)
        self.target_argp.setDecimals(3)
        self.target_ta = QDoubleSpinBox()
        self.target_ta.setRange(-360.0, 360.0)
        self.target_ta.setDecimals(3)
        target_form.addRow(self.target_enabled)
        target_form.addRow("Preset", self.target_preset)
        target_form.addRow("Dry Mass (kg)", self.target_dry_mass)
        target_form.addRow("Fuel Mass (kg)", self.target_fuel_mass)
        self.target_initial_state_button = self._make_section_toggle_button("target")
        target_form.addRow("Initial State", self.target_initial_state_button)
        self.target_initial_state_container = QWidget()
        target_initial_state_form = QFormLayout(self.target_initial_state_container)
        target_initial_state_form.setContentsMargins(0, 0, 0, 0)
        target_initial_state_form.addRow("a_km", self.target_a)
        target_initial_state_form.addRow("ecc", self.target_ecc)
        target_initial_state_form.addRow("inc_deg", self.target_inc)
        target_initial_state_form.addRow("raan_deg", self.target_raan)
        target_initial_state_form.addRow("argp_deg", self.target_argp)
        target_initial_state_form.addRow("true_anomaly_deg", self.target_ta)
        target_form.addRow(self.target_initial_state_container)
        self.target_strategy_combo = self._make_pointer_combo(MISSION_STRATEGY_OPTIONS["target"])
        self.target_execution_combo = self._make_pointer_combo(MISSION_EXECUTION_OPTIONS["target"])
        self.target_orbit_control_combo = self._make_pointer_combo(ORBIT_CONTROL_OPTIONS["target"])
        self.target_attitude_control_combo = self._make_pointer_combo(ATTITUDE_CONTROL_OPTIONS["target"])
        target_form.addRow("Mission Strategy", self._make_pointer_editor_row(self.target_strategy_combo, "target", "mission_strategy"))
        target_form.addRow("Mission Execution", self._make_pointer_editor_row(self.target_execution_combo, "target", "mission_execution"))
        target_form.addRow("Orbit Control", self._make_pointer_editor_row(self.target_orbit_control_combo, "target", "orbit_control"))
        target_form.addRow("Attitude Control", self._make_pointer_editor_row(self.target_attitude_control_combo, "target", "attitude_control"))
        layout.addWidget(target_box, 0, 0)

        chaser_box = QGroupBox("Chaser")
        chaser_form = QFormLayout(chaser_box)
        self.chaser_enabled = QCheckBox("Enabled")
        self.chaser_preset = QComboBox()
        self._configure_compact_combo(self.chaser_preset)
        self.chaser_preset.addItems(SATELLITE_PRESET_OPTIONS)
        self.chaser_dry_mass = QDoubleSpinBox()
        self.chaser_dry_mass.setRange(0.0, 1.0e9)
        self.chaser_dry_mass.setDecimals(3)
        self.chaser_fuel_mass = QDoubleSpinBox()
        self.chaser_fuel_mass.setRange(0.0, 1.0e9)
        self.chaser_fuel_mass.setDecimals(3)
        self.chaser_init_mode = QComboBox()
        self.chaser_init_mode.addItems(["rocket_deployment", "relative_ric_rect", "relative_ric_curv"])
        self.chaser_init_values = [self._make_free_spinbox() for _ in range(6)]
        self.chaser_deploy_time = self._make_free_spinbox()
        chaser_form.addRow(self.chaser_enabled)
        chaser_form.addRow("Preset", self.chaser_preset)
        chaser_form.addRow("Dry Mass (kg)", self.chaser_dry_mass)
        chaser_form.addRow("Fuel Mass (kg)", self.chaser_fuel_mass)
        self.chaser_initial_state_button = self._make_section_toggle_button("chaser")
        chaser_form.addRow("Initial State", self.chaser_initial_state_button)
        self.chaser_initial_state_container = QWidget()
        chaser_initial_state_form = QFormLayout(self.chaser_initial_state_container)
        chaser_initial_state_form.setContentsMargins(0, 0, 0, 0)
        chaser_initial_state_form.addRow("Init Mode", self.chaser_init_mode)
        chaser_initial_state_form.addRow("Deploy Time (s)", self.chaser_deploy_time)
        for i, widget in enumerate(self.chaser_init_values):
            chaser_initial_state_form.addRow(f"Init[{i}]", widget)
        chaser_form.addRow(self.chaser_initial_state_container)
        self.chaser_strategy_combo = self._make_pointer_combo(MISSION_STRATEGY_OPTIONS["chaser"])
        self.chaser_execution_combo = self._make_pointer_combo(MISSION_EXECUTION_OPTIONS["chaser"])
        self.chaser_orbit_control_combo = self._make_pointer_combo(ORBIT_CONTROL_OPTIONS["chaser"])
        self.chaser_attitude_control_combo = self._make_pointer_combo(ATTITUDE_CONTROL_OPTIONS["chaser"])
        chaser_form.addRow("Mission Strategy", self._make_pointer_editor_row(self.chaser_strategy_combo, "chaser", "mission_strategy"))
        chaser_form.addRow("Mission Execution", self._make_pointer_editor_row(self.chaser_execution_combo, "chaser", "mission_execution"))
        chaser_form.addRow("Orbit Control", self._make_pointer_editor_row(self.chaser_orbit_control_combo, "chaser", "orbit_control"))
        chaser_form.addRow("Attitude Control", self._make_pointer_editor_row(self.chaser_attitude_control_combo, "chaser", "attitude_control"))
        layout.addWidget(chaser_box, 0, 1)

        rocket_box = QGroupBox("Rocket")
        rocket_form = QFormLayout(rocket_box)
        self.rocket_enabled = QCheckBox("Enabled")
        self.rocket_preset = QComboBox()
        self._configure_compact_combo(self.rocket_preset)
        self.rocket_preset.addItems(ROCKET_PRESET_OPTIONS)
        self.rocket_payload = QDoubleSpinBox()
        self.rocket_payload.setRange(0.0, 1.0e9)
        self.rocket_payload.setDecimals(3)
        self.rocket_launch_lat = self._make_free_spinbox()
        self.rocket_launch_lon = self._make_free_spinbox()
        self.rocket_launch_alt = self._make_free_spinbox()
        self.rocket_launch_az = self._make_free_spinbox()
        self.rocket_launch_lat.setMaximumWidth(self.rocket_payload.maximumWidth())
        self.rocket_launch_lon.setMaximumWidth(self.rocket_payload.maximumWidth())
        self.rocket_launch_alt.setMaximumWidth(self.rocket_payload.maximumWidth())
        self.rocket_launch_az.setMaximumWidth(self.rocket_payload.maximumWidth())
        rocket_form.addRow(self.rocket_enabled)
        rocket_form.addRow("Stack Preset", self.rocket_preset)
        rocket_form.addRow("Payload Mass (kg)", self.rocket_payload)
        self.rocket_initial_state_button = self._make_section_toggle_button("rocket")
        rocket_form.addRow("Initial State", self.rocket_initial_state_button)
        self.rocket_initial_state_container = QWidget()
        rocket_initial_state_form = QFormLayout(self.rocket_initial_state_container)
        rocket_initial_state_form.setContentsMargins(0, 0, 0, 0)
        rocket_initial_state_form.addRow("Launch Lat (deg)", self.rocket_launch_lat)
        rocket_initial_state_form.addRow("Launch Lon (deg)", self.rocket_launch_lon)
        rocket_initial_state_form.addRow("Launch Alt (km)", self.rocket_launch_alt)
        rocket_initial_state_form.addRow("Launch Azimuth (deg)", self.rocket_launch_az)
        rocket_form.addRow(self.rocket_initial_state_container)
        self.rocket_strategy_combo = self._make_pointer_combo(MISSION_STRATEGY_OPTIONS["rocket"])
        self.rocket_execution_combo = self._make_pointer_combo(MISSION_EXECUTION_OPTIONS["rocket"])
        self.rocket_base_guidance_combo = self._make_pointer_combo(BASE_GUIDANCE_OPTIONS["rocket"])
        self.rocket_guidance_modifiers_label = QLabel("None")
        self.rocket_guidance_modifiers_label.setWordWrap(True)
        self.rocket_guidance_modifiers_button = QPushButton("+")
        self.rocket_guidance_modifiers_button.setFixedWidth(28)
        self.rocket_guidance_modifiers_button.setToolTip("Edit ordered rocket guidance modifiers")
        self.rocket_guidance_modifiers_button.clicked.connect(self._edit_rocket_guidance_modifiers)
        rocket_form.addRow("Mission Strategy", self._make_pointer_editor_row(self.rocket_strategy_combo, "rocket", "mission_strategy"))
        rocket_form.addRow("Mission Execution", self._make_pointer_editor_row(self.rocket_execution_combo, "rocket", "mission_execution"))
        rocket_form.addRow("Base Guidance", self._make_pointer_editor_row(self.rocket_base_guidance_combo, "rocket", "base_guidance"))
        rocket_guidance_modifiers_row = QWidget()
        rocket_guidance_modifiers_layout = QHBoxLayout(rocket_guidance_modifiers_row)
        rocket_guidance_modifiers_layout.setContentsMargins(0, 0, 0, 0)
        rocket_guidance_modifiers_layout.addWidget(self.rocket_guidance_modifiers_label, 1)
        rocket_guidance_modifiers_layout.addWidget(self.rocket_guidance_modifiers_button)
        rocket_form.addRow("Guidance Modifiers", rocket_guidance_modifiers_row)
        layout.addWidget(rocket_box, 0, 2)
        self._set_initial_state_section_visible("target", False)
        self._set_initial_state_section_visible("chaser", False)
        self._set_initial_state_section_visible("rocket", False)
        return tab

    def _build_outputs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.outputs_mode_label = QLabel("")
        self.outputs_mode_label.setWordWrap(True)
        layout.addWidget(self.outputs_mode_label)

        self.outputs_stack = QStackedWidget()
        layout.addWidget(self.outputs_stack, 1)

        single_run_page = QWidget()
        single_run_layout = QVBoxLayout(single_run_page)
        single_run_layout.setContentsMargins(0, 0, 0, 0)
        single_run_layout.setSpacing(8)
        self.stats_enabled = QCheckBox("Enable Stats")
        self.stats_print_summary = QCheckBox("Print Summary")
        self.stats_save_json = QCheckBox("Save JSON")
        self.stats_save_csv = QCheckBox("Save CSV")
        self.plots_enabled = QCheckBox("Enable Plots")
        self.plots_dpi = QSpinBox()
        self.plots_dpi.setRange(50, 2000)
        self.reference_object_edit = QLineEdit()
        self.figure_id_checks: dict[str, QCheckBox] = {}
        figure_ids_widget = QWidget()
        figure_ids_layout = QGridLayout(figure_ids_widget)
        figure_ids_layout.setContentsMargins(0, 0, 0, 0)
        for idx, figure_id in enumerate(FIGURE_ID_OPTIONS):
            check = QCheckBox(figure_id)
            self.figure_id_checks[figure_id] = check
            figure_ids_layout.addWidget(check, idx // 3, idx % 3)
        figure_ids_scroll = QScrollArea()
        figure_ids_scroll.setWidgetResizable(True)
        figure_ids_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        figure_ids_scroll.setWidget(figure_ids_widget)
        figure_ids_scroll.setMaximumHeight(140)
        figure_ids_panel = QWidget()
        figure_ids_panel_layout = QVBoxLayout(figure_ids_panel)
        figure_ids_panel_layout.setContentsMargins(0, 0, 0, 0)
        figure_ids_panel_layout.setSpacing(4)
        figure_ids_header = QHBoxLayout()
        figure_ids_header.setContentsMargins(0, 0, 0, 0)
        figure_ids_header.addWidget(QLabel("Figure IDs"))
        figure_ids_header.addStretch(1)
        plot_dpi_widget = QWidget()
        plot_dpi_layout = QVBoxLayout(plot_dpi_widget)
        plot_dpi_layout.setContentsMargins(0, 0, 0, 0)
        plot_dpi_layout.setSpacing(4)
        plot_dpi_layout.addWidget(QLabel("Plot DPI"))
        plot_dpi_layout.addWidget(self.plots_dpi)
        plot_dpi_layout.addStretch(1)
        figure_ids_header.addWidget(plot_dpi_widget)
        figure_ids_panel_layout.addLayout(figure_ids_header)
        figure_ids_panel_layout.addWidget(figure_ids_scroll, 1)
        self.animation_type_checks: dict[str, QCheckBox] = {}
        animation_types_widget = QWidget()
        animation_types_layout = QGridLayout(animation_types_widget)
        animation_types_layout.setContentsMargins(0, 0, 0, 0)
        for idx, anim_type in enumerate(ANIMATION_TYPE_OPTIONS):
            check = QCheckBox(anim_type)
            self.animation_type_checks[anim_type] = check
            animation_types_layout.addWidget(check, idx // 2, idx % 2)
        animation_types_scroll = QScrollArea()
        animation_types_scroll.setWidgetResizable(True)
        animation_types_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        animation_types_scroll.setWidget(animation_types_widget)
        animation_types_scroll.setMaximumHeight(140)
        self.animation_fps_spin = QDoubleSpinBox()
        self.animation_fps_spin.setRange(1.0, 240.0)
        self.animation_fps_spin.setDecimals(1)
        self.animation_fps_spin.setValue(30.0)
        self.animation_speed_multiple_spin = QDoubleSpinBox()
        self.animation_speed_multiple_spin.setRange(0.1, 1000.0)
        self.animation_speed_multiple_spin.setDecimals(2)
        self.animation_speed_multiple_spin.setValue(10.0)
        self.animation_frame_stride_spin = QSpinBox()
        self.animation_frame_stride_spin.setRange(1, 100000)
        self.animation_frame_stride_spin.setValue(1)
        left_controls = QWidget()
        left_controls_layout = QVBoxLayout(left_controls)
        left_controls_layout.setContentsMargins(0, 0, 0, 0)
        left_controls_layout.setSpacing(4)
        left_controls_layout.addWidget(self.stats_enabled)
        left_controls_layout.addWidget(self.stats_print_summary)
        left_controls_layout.addWidget(self.stats_save_json)
        left_controls_layout.addWidget(self.stats_save_csv)
        left_controls_layout.addWidget(self.plots_enabled)
        left_controls_layout.addWidget(figure_ids_panel)
        single_run_layout.addWidget(left_controls, 0, Qt.AlignTop)

        lower_fields = QWidget()
        lower_fields_layout = QVBoxLayout(lower_fields)
        lower_fields_layout.setContentsMargins(0, 0, 0, 0)
        lower_fields_layout.setSpacing(6)
        lower_fields_layout.addWidget(QLabel("RIC Reference Object"))
        lower_fields_layout.addWidget(self.reference_object_edit, 0, Qt.AlignLeft)
        animation_row = QHBoxLayout()
        animation_row.setContentsMargins(0, 0, 0, 0)
        animation_row.setSpacing(12)
        animation_left = QWidget()
        animation_left_layout = QVBoxLayout(animation_left)
        animation_left_layout.setContentsMargins(0, 0, 0, 0)
        animation_left_layout.setSpacing(6)
        animation_left_layout.addWidget(QLabel("Animation Types"))
        animation_left_layout.addWidget(animation_types_scroll, 0, Qt.AlignLeft)
        animation_settings_form = QFormLayout()
        animation_settings_form.addRow("Animation FPS", self.animation_fps_spin)
        animation_settings_form.addRow("Speed Multiple", self.animation_speed_multiple_spin)
        animation_settings_form.addRow("Frame Stride", self.animation_frame_stride_spin)
        animation_row.addWidget(animation_left, 0, Qt.AlignTop)
        animation_row.addLayout(animation_settings_form)
        animation_row.addStretch(1)
        lower_fields_layout.addLayout(animation_row)
        single_run_layout.addWidget(lower_fields, 0, Qt.AlignLeft)
        single_run_layout.addStretch(1)
        self.outputs_stack.addWidget(single_run_page)

        mc_page = QWidget()
        mc_layout = QFormLayout(mc_page)
        self.mc_save_iteration_summaries = QCheckBox("Save Iteration Summaries")
        self.mc_save_aggregate_summary = QCheckBox("Save Aggregate Summary")
        self.mc_save_histograms = QCheckBox("Save Histograms")
        self.mc_display_histograms = QCheckBox("Display Histograms")
        self.mc_save_ops_dashboard = QCheckBox("Save Ops Dashboard")
        self.mc_display_ops_dashboard = QCheckBox("Display Ops Dashboard")
        self.mc_save_raw_runs = QCheckBox("Save Raw Runs")
        self.mc_require_rocket_insertion = QCheckBox("Require Rocket Insertion For Pass")
        self.mc_baseline_summary_json = QLineEdit()
        self.mc_gate_min_closest_approach = self._make_free_spinbox()
        self.mc_gate_max_duration = self._make_free_spinbox()
        self.mc_gate_max_total_dv = self._make_free_spinbox()
        self.mc_gate_max_guardrail_events = self._make_free_spinbox()
        mc_layout.addRow(self.mc_save_iteration_summaries)
        mc_layout.addRow(self.mc_save_aggregate_summary)
        mc_layout.addRow(self.mc_save_histograms)
        mc_layout.addRow(self.mc_display_histograms)
        mc_layout.addRow(self.mc_save_ops_dashboard)
        mc_layout.addRow(self.mc_display_ops_dashboard)
        mc_layout.addRow(self.mc_save_raw_runs)
        mc_layout.addRow(self.mc_require_rocket_insertion)
        mc_layout.addRow("Baseline Summary JSON", self.mc_baseline_summary_json)
        mc_layout.addRow("Gate: Min Closest Approach (km)", self.mc_gate_min_closest_approach)
        mc_layout.addRow("Gate: Max Duration (s)", self.mc_gate_max_duration)
        mc_layout.addRow("Gate: Max Total dV (m/s)", self.mc_gate_max_total_dv)
        mc_layout.addRow("Gate: Max Guardrail Events", self.mc_gate_max_guardrail_events)
        self.outputs_stack.addWidget(mc_page)
        return tab

    def _build_yaml_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        buttons = QHBoxLayout()
        self.refresh_yaml_button = QPushButton("Refresh From Form")
        self.refresh_yaml_button.clicked.connect(self._refresh_yaml)
        buttons.addWidget(self.refresh_yaml_button)
        self.apply_yaml_button = QPushButton("Apply YAML To Form")
        self.apply_yaml_button.clicked.connect(self._apply_yaml_to_form)
        buttons.addWidget(self.apply_yaml_button)
        self.validate_yaml_button = QPushButton("Validate YAML")
        self.validate_yaml_button.clicked.connect(self._validate_yaml)
        buttons.addWidget(self.validate_yaml_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.yaml_editor = QPlainTextEdit()
        layout.addWidget(self.yaml_editor, 1)
        return tab

    def _build_results_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs, 1)

        console_tab = QWidget()
        console_layout = QVBoxLayout(console_tab)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        console_layout.addWidget(self.console)
        self.results_tabs.addTab(console_tab, "Console")

        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.results_summary = QPlainTextEdit()
        self.results_summary.setReadOnly(True)
        summary_layout.addWidget(self.results_summary)
        self.results_tabs.addTab(summary_tab, "Summary")

        artifacts_tab = QWidget()
        artifacts_layout = QGridLayout(artifacts_tab)
        self.output_files = QListWidget()
        self.output_files.currentTextChanged.connect(self._on_output_file_selected)
        artifacts_layout.addWidget(QLabel("Output Files"), 0, 0)
        artifacts_layout.addWidget(self.output_files, 1, 0)

        self.preview_title = QLabel("Select an artifact to preview.")
        artifacts_layout.addWidget(self.preview_title, 0, 1)

        self.preview_stack = QTabWidget()
        self.preview_image = QLabel("No image selected.")
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setMinimumSize(320, 240)
        self.preview_image.installEventFilter(self)
        preview_controls = QWidget()
        preview_controls_layout = QHBoxLayout(preview_controls)
        preview_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.zoom_out_button = QPushButton("Zoom -")
        self.zoom_out_button.clicked.connect(self._zoom_out_preview)
        preview_controls_layout.addWidget(self.zoom_out_button)
        self.zoom_in_button = QPushButton("Zoom +")
        self.zoom_in_button.clicked.connect(self._zoom_in_preview)
        preview_controls_layout.addWidget(self.zoom_in_button)
        self.zoom_fit_button = QPushButton("Fit")
        self.zoom_fit_button.clicked.connect(self._fit_preview_image)
        preview_controls_layout.addWidget(self.zoom_fit_button)
        self.zoom_actual_button = QPushButton("1:1")
        self.zoom_actual_button.clicked.connect(self._actual_size_preview)
        preview_controls_layout.addWidget(self.zoom_actual_button)
        self.zoom_label = QLabel("Fit")
        preview_controls_layout.addWidget(self.zoom_label)
        preview_controls_layout.addStretch(1)
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setWidget(self.preview_image)
        self.preview_scroll.viewport().installEventFilter(self)
        self.preview_text = QPlainTextEdit()
        self.preview_text.setReadOnly(True)
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        image_layout.addWidget(preview_controls)
        image_layout.addWidget(self.preview_scroll, 1)
        self.preview_stack.addTab(image_tab, "Image")
        self.preview_stack.addTab(self.preview_text, "Text")
        artifacts_layout.addWidget(self.preview_stack, 1, 1)
        artifacts_layout.setColumnStretch(0, 2)
        artifacts_layout.setColumnStretch(1, 3)
        self.results_tabs.addTab(artifacts_tab, "Artifacts")
        return tab

    def _make_free_spinbox(self) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(-1.0e12, 1.0e12)
        widget.setDecimals(6)
        return widget

    def _configure_compact_spinbox(self, widget: QDoubleSpinBox, width: int = 84) -> None:
        widget.setMaximumWidth(width)

    def _make_pointer_combo(self, options: list[tuple[str, dict | None]]) -> QComboBox:
        combo = QComboBox()
        combo.setMinimumContentsLength(12)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMaximumWidth(126)
        for label, pointer in options:
            combo.addItem(label, copy.deepcopy(pointer))
        return combo

    def _configure_compact_combo(self, combo: QComboBox) -> None:
        combo.setMinimumContentsLength(12)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMaximumWidth(126)

    def _make_section_toggle_button(self, object_key: str) -> QPushButton:
        button = QPushButton("+")
        button.setFixedWidth(28)
        button.clicked.connect(lambda: self._toggle_initial_state_section(object_key))
        return button

    def _make_pointer_editor_row(self, combo: QComboBox, object_key: str, pointer_kind: str) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(combo, 1)
        button = QPushButton("+")
        button.setFixedWidth(28)
        button.setToolTip(f"Edit params for {object_key}.{pointer_kind}")
        button.clicked.connect(lambda: self._edit_pointer_params(object_key, pointer_kind, combo))
        row_layout.addWidget(button)
        return row

    def _pointer_signature(self, pointer: dict | None) -> tuple[str, str, str]:
        if not isinstance(pointer, dict):
            return ("", "", "")
        return (
            str(pointer.get("module", "") or ""),
            str(pointer.get("class_name", "") or ""),
            str(pointer.get("function", "") or ""),
        )

    def _set_pointer_combo_value(self, combo: QComboBox, pointer: dict | None) -> None:
        target_sig = self._pointer_signature(pointer)
        for i in range(combo.count()):
            candidate = combo.itemData(i)
            if self._pointer_signature(candidate) == target_sig:
                combo.setCurrentIndex(i)
                return
        label = "None" if pointer is None else f"Custom: {target_sig[0]}.{target_sig[1] or target_sig[2]}".strip(".")
        combo.addItem(label, copy.deepcopy(pointer))
        combo.setCurrentIndex(combo.count() - 1)

    def _combo_pointer_value(self, combo: QComboBox, existing: dict | None = None) -> dict | None:
        selected = combo.currentData()
        if selected is None:
            return None
        selected_copy = copy.deepcopy(selected)
        if existing is not None and self._pointer_signature(existing) == self._pointer_signature(selected_copy):
            merged = copy.deepcopy(existing)
            merged["kind"] = str(selected_copy.get("kind", merged.get("kind", "python")))
            merged["module"] = selected_copy.get("module")
            merged["class_name"] = selected_copy.get("class_name")
            if "function" in selected_copy:
                merged["function"] = selected_copy.get("function")
            return merged
        return selected_copy

    def _set_combo_text_or_append(self, combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
            return
        combo.addItem(value)
        combo.setCurrentIndex(combo.count() - 1)

    def _get_existing_pointer_for_editor(self, object_key: str, pointer_kind: str) -> dict | None:
        cfg = self._collect_config_from_widgets()
        section = dict(cfg.get(object_key, {}) or {})
        pointer = section.get(pointer_kind)
        return dict(pointer) if isinstance(pointer, dict) else None

    def _edit_pointer_params(self, object_key: str, pointer_kind: str, combo: QComboBox) -> None:
        pointer = self._combo_pointer_value(combo, existing=self._get_existing_pointer_for_editor(object_key, pointer_kind))
        if pointer is None:
            self.statusBar().showMessage("No pointer selected for parameter editing.", 5000)
            return
        params = dict(pointer.get("params", {}) or {})
        if self._edit_pointer_params_structured(pointer, combo, object_key, pointer_kind):
            self._on_mc_catalog_source_changed()
            return
        self._edit_pointer_params_yaml(pointer, combo, object_key, pointer_kind, params)
        self._on_mc_catalog_source_changed()

    def _edit_pointer_params_yaml(self, pointer: dict, combo: QComboBox, object_key: str, pointer_kind: str, params: dict) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Params: {object_key}.{pointer_kind}")
        dialog.resize(520, 420)
        layout = QVBoxLayout(dialog)
        header = QLabel(
            f"{pointer.get('module', '')}.{pointer.get('class_name', '') or pointer.get('function', '')}".strip(".")
        )
        header.setWordWrap(True)
        layout.addWidget(header)
        editor = QPlainTextEdit()
        editor.setPlainText(yaml.safe_dump({"params": params}, sort_keys=False, allow_unicode=False))
        layout.addWidget(editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            parsed = yaml.safe_load(editor.toPlainText()) or {}
            if not isinstance(parsed, dict):
                raise ValueError("Params editor content must be a YAML mapping/object.")
            new_params = dict(parsed.get("params", {}) or {})
            pointer["params"] = new_params
            current_index = combo.currentIndex()
            combo.setItemData(current_index, pointer)
            self._mark_dirty()
            self.statusBar().showMessage(f"Updated params for {object_key}.{pointer_kind}.", 5000)
        except Exception as exc:
            self._show_error("Invalid Params YAML", str(exc))
            return

    def _refresh_rocket_guidance_modifiers_label(self) -> None:
        names = [str(item.get("class_name", "") or item.get("function", "") or "Unknown") for item in self.rocket_guidance_modifiers_config]
        self.rocket_guidance_modifiers_label.setText(", ".join(names) if names else "None")

    def _edit_rocket_guidance_modifiers(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Rocket Guidance Modifiers")
        dialog.resize(560, 420)
        layout = QVBoxLayout(dialog)
        hint = QLabel(
            "Enter an ordered YAML list of guidance modifier pointers. Available classes: "
            + ", ".join(label for label, _ in GUIDANCE_MODIFIER_OPTIONS)
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        editor = QPlainTextEdit()
        editor.setPlainText(yaml.safe_dump(self.rocket_guidance_modifiers_config, sort_keys=False, allow_unicode=False))
        layout.addWidget(editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            parsed = yaml.safe_load(editor.toPlainText()) or []
            if not isinstance(parsed, list):
                raise ValueError("Guidance modifiers must be a YAML list.")
            normalized: list[dict] = []
            for item in parsed:
                if not isinstance(item, dict):
                    raise ValueError("Each guidance modifier must be a mapping/object.")
                normalized.append(dict(item))
            self.rocket_guidance_modifiers_config = normalized
            self._refresh_rocket_guidance_modifiers_label()
            self._mark_dirty()
            self._on_mc_catalog_source_changed()
            self.statusBar().showMessage("Updated rocket guidance modifiers.", 5000)
        except Exception as exc:
            self._show_error("Invalid Guidance Modifiers YAML", str(exc))

    def _on_mc_catalog_source_changed(self) -> None:
        self._rebuild_mc_category_combo()
        self._refresh_mc_parameter_options()
        self._refresh_mc_variations_list()

    def _pointer_form_schema(self, pointer: dict) -> list[dict] | None:
        class_name = str(pointer.get("class_name", "") or "")
        return PARAMETER_FORM_SCHEMAS.get(class_name)

    def _pointer_display_name(self, pointer: dict) -> str:
        return f"{pointer.get('module', '')}.{pointer.get('class_name', '') or pointer.get('function', '')}".strip(".")

    def _default_params_for_pointer(self, pointer: dict) -> dict:
        module_name = str(pointer.get("module", "") or "")
        class_name = str(pointer.get("class_name", "") or "")
        if not module_name or not class_name:
            return {}
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            instance = cls()
            return copy.deepcopy(getattr(instance, "__dict__", {}) or {})
        except Exception:
            return {}

    def _normalize_form_value(self, field_spec: dict, params: dict, defaults: dict) -> object:
        key = field_spec["key"]
        if key in params:
            return params.get(key)
        return defaults.get(key)

    def _format_vector_text(self, value: object, length: int | None = None) -> str:
        if value is None:
            values = [0.0] * int(length or 0)
        else:
            values = list(value) if isinstance(value, (list, tuple)) else [value]
        if length is not None and len(values) < length:
            values = values + [0.0] * (length - len(values))
        return ", ".join(str(v) for v in values)

    def _parse_vector_text(self, text: str, length: int | None = None) -> list[float]:
        raw = text.strip()
        if not raw:
            values: list[float] = [0.0] * int(length or 0)
        else:
            parsed = yaml.safe_load(f"[{raw}]") if "[" not in raw else yaml.safe_load(raw)
            if not isinstance(parsed, list):
                raise ValueError("Vector values must be a comma-separated list.")
            values = [float(v) for v in parsed]
        if length is not None and len(values) != length:
            raise ValueError(f"Expected {length} values.")
        return values

    def _format_yaml_text(self, value: object) -> str:
        if value is None:
            payload: object = []
        else:
            payload = value
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False).strip()
        return text if text else "[]"

    def _parse_yaml_text(self, text: str) -> object:
        raw = text.strip()
        if not raw:
            return []
        return yaml.safe_load(raw)

    def _edit_pointer_params_structured(self, pointer: dict, combo: QComboBox, object_key: str, pointer_kind: str) -> bool:
        schema = self._pointer_form_schema(pointer)
        if not schema:
            return False
        params = dict(pointer.get("params", {}) or {})
        defaults = self._default_params_for_pointer(pointer)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Params: {object_key}.{pointer_kind}")
        dialog.resize(560, 480)
        layout = QVBoxLayout(dialog)
        header = QLabel(self._pointer_display_name(pointer))
        header.setWordWrap(True)
        layout.addWidget(header)
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)
        editors: dict[str, QWidget] = {}
        for field_spec in schema:
            key = field_spec["key"]
            kind = field_spec["kind"]
            value = self._normalize_form_value(field_spec, params, defaults)
            if kind == "float":
                widget = self._make_free_spinbox()
                widget.setValue(float(value or 0.0))
            elif kind == "optional_float":
                widget = QLineEdit("" if value is None else str(value))
                widget.setPlaceholderText("Leave blank for default/None")
            elif kind == "int":
                widget = QSpinBox()
                widget.setRange(-10**9, 10**9)
                widget.setValue(int(value or 0))
            elif kind == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(value))
            elif kind == "string":
                widget = QLineEdit("" if value is None else str(value))
            elif kind == "choice":
                widget = QComboBox()
                for option in field_spec.get("options", []):
                    widget.addItem(str(option))
                current_text = "" if value is None else str(value)
                idx = widget.findText(current_text)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                elif current_text:
                    widget.addItem(current_text)
                    widget.setCurrentIndex(widget.count() - 1)
            elif kind == "vector":
                widget = QLineEdit(self._format_vector_text(value, field_spec.get("length")))
                widget.setPlaceholderText("comma-separated values")
            elif kind == "yaml":
                widget = QPlainTextEdit(self._format_yaml_text(value))
                widget.setPlaceholderText("Enter YAML")
                widget.setMinimumHeight(140)
            else:
                continue
            editors[key] = widget
            form.addRow(field_spec["label"], widget)
        layout.addWidget(form_widget, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return True
        try:
            new_params: dict[str, object] = {}
            for field_spec in schema:
                key = field_spec["key"]
                kind = field_spec["kind"]
                widget = editors[key]
                if kind == "float":
                    new_params[key] = float(widget.value())  # type: ignore[attr-defined]
                elif kind == "optional_float":
                    raw = widget.text().strip()  # type: ignore[attr-defined]
                    new_params[key] = None if not raw else float(raw)
                elif kind == "int":
                    new_params[key] = int(widget.value())  # type: ignore[attr-defined]
                elif kind == "bool":
                    new_params[key] = bool(widget.isChecked())  # type: ignore[attr-defined]
                elif kind == "string":
                    raw = widget.text().strip()  # type: ignore[attr-defined]
                    new_params[key] = raw or None
                elif kind == "choice":
                    new_params[key] = widget.currentText()  # type: ignore[attr-defined]
                elif kind == "vector":
                    new_params[key] = self._parse_vector_text(widget.text(), field_spec.get("length"))  # type: ignore[attr-defined]
                elif kind == "yaml":
                    new_params[key] = self._parse_yaml_text(widget.toPlainText())  # type: ignore[attr-defined]
            pointer["params"] = new_params
            current_index = combo.currentIndex()
            combo.setItemData(current_index, pointer)
            self._mark_dirty()
            self.statusBar().showMessage(f"Updated params for {object_key}.{pointer_kind}.", 5000)
        except Exception as exc:
            self._show_error("Invalid Params", str(exc))
        return True

    def _set_initial_state_section_visible(self, object_key: str, visible: bool) -> None:
        container = getattr(self, f"{object_key}_initial_state_container")
        button = getattr(self, f"{object_key}_initial_state_button")
        container.setVisible(bool(visible))
        button.setText("-" if visible else "+")

    def _toggle_initial_state_section(self, object_key: str) -> None:
        container = getattr(self, f"{object_key}_initial_state_container")
        self._set_initial_state_section_visible(object_key, not container.isVisible())

    def _rebuild_mc_category_combo(self) -> None:
        current_text = self.mc_category_combo.currentText() if hasattr(self, "mc_category_combo") else ""
        self.mc_category_combo.blockSignals(True)
        self.mc_category_combo.clear()
        for category_name in MC_PARAMETER_CATEGORIES:
            self.mc_category_combo.addItem(category_name)
        controller_params = self._get_controller_param_variation_options()
        if controller_params:
            self.mc_category_combo.addItem("Controller Params")
        self.mc_category_combo.addItem("Custom Path")
        if current_text:
            idx = self.mc_category_combo.findText(current_text)
            if idx >= 0:
                self.mc_category_combo.setCurrentIndex(idx)
        self.mc_category_combo.blockSignals(False)

    def _get_controller_param_variation_options(self) -> list[tuple[str, str]]:
        try:
            cfg = self._collect_config_from_widgets()
        except Exception:
            cfg = copy.deepcopy(self.current_config)
        options: list[tuple[str, str]] = []
        pointer_paths = [
            ("target", "mission_strategy", cfg.get("target", {}).get("mission_strategy")),
            ("target", "mission_execution", cfg.get("target", {}).get("mission_execution")),
            ("target", "orbit_control", cfg.get("target", {}).get("orbit_control")),
            ("target", "attitude_control", cfg.get("target", {}).get("attitude_control")),
            ("chaser", "mission_strategy", cfg.get("chaser", {}).get("mission_strategy")),
            ("chaser", "mission_execution", cfg.get("chaser", {}).get("mission_execution")),
            ("chaser", "orbit_control", cfg.get("chaser", {}).get("orbit_control")),
            ("chaser", "attitude_control", cfg.get("chaser", {}).get("attitude_control")),
            ("rocket", "mission_strategy", cfg.get("rocket", {}).get("mission_strategy")),
            ("rocket", "mission_execution", cfg.get("rocket", {}).get("mission_execution")),
            ("rocket", "base_guidance", cfg.get("rocket", {}).get("base_guidance") or cfg.get("rocket", {}).get("guidance")),
        ]
        for object_key, pointer_key, pointer in pointer_paths:
            if not isinstance(pointer, dict):
                continue
            params = pointer.get("params")
            if not isinstance(params, (dict, list)):
                continue
            class_name = str(pointer.get("class_name", "") or pointer.get("function", "") or pointer_key)
            base_label = f"{object_key}.{class_name}"
            base_path = f"{object_key}.{pointer_key}.params"
            options.extend(self._flatten_mc_parameter_options(params, base_path, base_label))
        for idx, pointer in enumerate(cfg.get("rocket", {}).get("guidance_modifiers", []) or []):
            if not isinstance(pointer, dict):
                continue
            params = pointer.get("params")
            if not isinstance(params, (dict, list)):
                continue
            class_name = str(pointer.get("class_name", "") or pointer.get("function", "") or f"modifier_{idx}")
            base_label = f"rocket.{class_name}"
            base_path = f"rocket.guidance_modifiers[{idx}].params"
            options.extend(self._flatten_mc_parameter_options(params, base_path, base_label))
        return options

    def _flatten_mc_parameter_options(self, value: object, path_prefix: str, label_prefix: str) -> list[tuple[str, str]]:
        if isinstance(value, dict):
            out: list[tuple[str, str]] = []
            for key, child in value.items():
                child_path = f"{path_prefix}.{key}"
                child_label = f"{label_prefix}.{key}"
                out.extend(self._flatten_mc_parameter_options(child, child_path, child_label))
            return out
        if isinstance(value, list):
            out = []
            for idx, child in enumerate(value):
                child_path = f"{path_prefix}[{idx}]"
                child_label = f"{label_prefix}[{idx}]"
                out.extend(self._flatten_mc_parameter_options(child, child_path, child_label))
            return out
        return [(label_prefix, path_prefix)]

    def _current_mc_category_options(self) -> list[tuple[str, str]]:
        category = self.mc_category_combo.currentText()
        if category == "Controller Params":
            return self._get_controller_param_variation_options()
        return list(MC_PARAMETER_CATEGORIES.get(category, []))

    def _refresh_mc_parameter_options(self) -> None:
        category = self.mc_category_combo.currentText()
        current_path = self.mc_parameter_combo.currentData()
        self.mc_parameter_combo.blockSignals(True)
        self.mc_parameter_combo.clear()
        for label, path in self._current_mc_category_options():
            self.mc_parameter_combo.addItem(label, path)
        if current_path:
            idx = self.mc_parameter_combo.findData(current_path)
            if idx >= 0:
                self.mc_parameter_combo.setCurrentIndex(idx)
        self.mc_parameter_combo.blockSignals(False)
        custom = category == "Custom Path"
        self.mc_parameter_combo.setEnabled(not custom)
        self.mc_custom_path_edit.setVisible(custom)
        parameter_label = self.mc_variation_form.labelForField(self.mc_parameter_combo)
        custom_label = self.mc_variation_form.labelForField(self.mc_custom_path_edit)
        if parameter_label is not None:
            parameter_label.setVisible(not custom)
        if custom_label is not None:
            custom_label.setVisible(custom)
        self.mc_parameter_combo.setVisible(not custom)
        self._refresh_mc_variation_button_state()

    def _refresh_mc_mode_ui(self) -> None:
        mode = self.mc_mode_combo.currentText().strip().lower()
        mode_to_index = {"choice": 0, "uniform": 1, "normal": 2}
        self.mc_mode_stack.setCurrentIndex(mode_to_index.get(mode, 0))

    def _refresh_mc_variation_button_state(self) -> None:
        has_selection = self.mc_variations_list.currentRow() >= 0
        self.mc_add_update_variation_button.setText("Update" if has_selection else "Add")
        self.mc_remove_variation_button.setEnabled(has_selection)

    def _mc_path_display_name(self, path: str) -> str:
        for entries in MC_PARAMETER_CATEGORIES.values():
            for label, candidate_path in entries:
                if candidate_path == path:
                    return label
        for label, candidate_path in self._get_controller_param_variation_options():
            if candidate_path == path:
                return label
        return path

    def _format_mc_variation_label(self, variation: dict) -> str:
        path = str(variation.get("parameter_path", "") or "")
        mode = str(variation.get("mode", "choice") or "choice").lower()
        label = self._mc_path_display_name(path)
        if mode == "choice":
            options = list(variation.get("options", []) or [])
            return f"{label} | choice | {len(options)} option(s)"
        if mode == "uniform":
            return f"{label} | uniform | {variation.get('low')} to {variation.get('high')}"
        if mode == "normal":
            return f"{label} | normal | mean {variation.get('mean')} std {variation.get('std')}"
        return f"{label} | {mode}"

    def _refresh_mc_variations_list(self) -> None:
        self.mc_variations_list.blockSignals(True)
        selected_row = self.mc_variations_list.currentRow()
        self.mc_variations_list.clear()
        for variation in self.mc_variations:
            self.mc_variations_list.addItem(self._format_mc_variation_label(variation))
        if 0 <= selected_row < self.mc_variations_list.count():
            self.mc_variations_list.setCurrentRow(selected_row)
        self.mc_variations_list.blockSignals(False)
        self._refresh_mc_variation_button_state()

    def _set_mc_variation_path_selection(self, parameter_path: str) -> None:
        matched_category: str | None = None
        for category_name, entries in MC_PARAMETER_CATEGORIES.items():
            if any(candidate_path == parameter_path for _, candidate_path in entries):
                matched_category = category_name
                break
        if matched_category is None:
            controller_paths = {candidate_path for _, candidate_path in self._get_controller_param_variation_options()}
            matched_category = "Controller Params" if parameter_path in controller_paths else "Custom Path"
        category_index = self.mc_category_combo.findText(matched_category)
        if category_index >= 0:
            self.mc_category_combo.setCurrentIndex(category_index)
        self._refresh_mc_parameter_options()
        if matched_category == "Custom Path":
            self.mc_custom_path_edit.setText(parameter_path)
            return
        param_index = self.mc_parameter_combo.findData(parameter_path)
        if param_index >= 0:
            self.mc_parameter_combo.setCurrentIndex(param_index)
        else:
            custom_index = self.mc_category_combo.findText("Custom Path")
            if custom_index >= 0:
                self.mc_category_combo.setCurrentIndex(custom_index)
            self._refresh_mc_parameter_options()
            self.mc_custom_path_edit.setText(parameter_path)

    def _on_mc_variation_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.mc_variations):
            self._refresh_mc_variation_button_state()
            return
        variation = dict(self.mc_variations[row] or {})
        self._set_mc_variation_path_selection(str(variation.get("parameter_path", "") or ""))
        mode = str(variation.get("mode", "choice") or "choice").lower()
        self.mc_mode_combo.setCurrentText(mode if mode in {"choice", "uniform", "normal"} else "choice")
        self.mc_choice_options_edit.setText(", ".join(str(v) for v in list(variation.get("options", []) or [])))
        self.mc_uniform_low_spin.setValue(float(variation.get("low", 0.0) or 0.0))
        self.mc_uniform_high_spin.setValue(float(variation.get("high", 0.0) or 0.0))
        self.mc_normal_mean_spin.setValue(float(variation.get("mean", 0.0) or 0.0))
        self.mc_normal_std_spin.setValue(float(variation.get("std", 0.0) or 0.0))
        self._refresh_mc_mode_ui()
        self._refresh_mc_variation_button_state()

    def _clear_mc_variation_editor(self) -> None:
        self.mc_variations_list.blockSignals(True)
        self.mc_variations_list.clearSelection()
        self.mc_variations_list.setCurrentRow(-1)
        self.mc_variations_list.blockSignals(False)
        if self.mc_category_combo.count() > 0:
            self.mc_category_combo.setCurrentIndex(0)
        self._refresh_mc_parameter_options()
        self.mc_custom_path_edit.clear()
        self.mc_mode_combo.setCurrentText("choice")
        self.mc_choice_options_edit.clear()
        self.mc_uniform_low_spin.setValue(0.0)
        self.mc_uniform_high_spin.setValue(0.0)
        self.mc_normal_mean_spin.setValue(0.0)
        self.mc_normal_std_spin.setValue(0.0)
        self._refresh_mc_mode_ui()
        self._refresh_mc_variation_button_state()

    def _parse_mc_choice_options(self) -> list[object]:
        raw = self.mc_choice_options_edit.text().strip()
        if not raw:
            raise ValueError("Choice mode requires at least one option.")
        options: list[object] = []
        for token in raw.split(","):
            item = token.strip()
            if not item:
                continue
            options.append(yaml.safe_load(item))
        if not options:
            raise ValueError("Choice mode requires at least one option.")
        return options

    def _build_mc_variation_from_editor(self) -> dict:
        category = self.mc_category_combo.currentText()
        if category == "Custom Path":
            parameter_path = self.mc_custom_path_edit.text().strip()
        else:
            parameter_path = str(self.mc_parameter_combo.currentData() or "").strip()
        if not parameter_path:
            raise ValueError("Select a Monte Carlo parameter or enter a custom path.")
        mode = self.mc_mode_combo.currentText().strip().lower()
        variation: dict[str, object] = {"parameter_path": parameter_path, "mode": mode}
        if mode == "choice":
            variation["options"] = self._parse_mc_choice_options()
        elif mode == "uniform":
            variation["low"] = float(self.mc_uniform_low_spin.value())
            variation["high"] = float(self.mc_uniform_high_spin.value())
        elif mode == "normal":
            variation["mean"] = float(self.mc_normal_mean_spin.value())
            variation["std"] = float(self.mc_normal_std_spin.value())
        else:
            raise ValueError(f"Unsupported Monte Carlo mode '{mode}'.")
        return variation

    def _on_add_or_update_mc_variation(self) -> None:
        try:
            variation = self._build_mc_variation_from_editor()
            row = self.mc_variations_list.currentRow()
            if 0 <= row < len(self.mc_variations):
                self.mc_variations[row] = variation
                action = "Updated"
            else:
                self.mc_variations.append(variation)
                row = len(self.mc_variations) - 1
                action = "Added"
            self._refresh_mc_variations_list()
            self.mc_variations_list.setCurrentRow(row)
            self._mark_dirty()
            self.statusBar().showMessage(f"{action} Monte Carlo variation.", 5000)
        except Exception as exc:
            self._show_error("Invalid Monte Carlo Variation", str(exc))

    def _on_remove_mc_variation(self) -> None:
        row = self.mc_variations_list.currentRow()
        if row < 0 or row >= len(self.mc_variations):
            return
        self.mc_variations.pop(row)
        self._refresh_mc_variations_list()
        if self.mc_variations:
            self.mc_variations_list.setCurrentRow(min(row, len(self.mc_variations) - 1))
        else:
            self._clear_mc_variation_editor()
        self._mark_dirty()
        self.statusBar().showMessage("Removed Monte Carlo variation.", 5000)

    def _load_config_into_widgets(self, cfg: dict) -> None:
        self._suppress_dirty_tracking = True
        sim = cfg.get("simulator", {})
        outputs = cfg.get("outputs", {})
        mc = cfg.get("monte_carlo", {})
        target = cfg.get("target", {})
        chaser = cfg.get("chaser", {})
        rocket = cfg.get("rocket", {})

        self.scenario_name_edit.setText(str(cfg.get("scenario_name", "")))
        self.duration_spin.setValue(float(sim.get("duration_s", 3600.0)))
        self.dt_spin.setValue(float(sim.get("dt_s", 1.0)))
        self.output_mode_combo.setCurrentText(str(outputs.get("mode", "interactive")))
        self.output_dir_edit.setText(str(outputs.get("output_dir", "outputs/gui_run")))
        self.mc_enabled_check.setChecked(bool(mc.get("enabled", False)))
        self.mc_iterations_spin.setValue(int(mc.get("iterations", 1)))
        self.mc_parallel_check.setChecked(bool(mc.get("parallel_enabled", False)))
        self.mc_workers_spin.setValue(int(mc.get("parallel_workers", 0)))
        self.mc_base_seed_spin.setValue(int(mc.get("base_seed", 0)))
        self.mc_variations = [dict(v or {}) for v in list(mc.get("variations", []) or [])]
        self._rebuild_mc_category_combo()
        self._refresh_mc_parameter_options()
        self._refresh_mc_variations_list()
        self._clear_mc_variation_editor()
        dynamics = dict(sim.get("dynamics", {}) or {})
        orbit_dyn = dict(dynamics.get("orbit", {}) or {})
        att_dyn = dict(dynamics.get("attitude", {}) or {})
        disturbance_torques = dict(att_dyn.get("disturbance_torques", {}) or {})
        orbit_substep_val = orbit_dyn.get("orbit_substep_s")
        attitude_substep_val = att_dyn.get("attitude_substep_s")
        self.orbit_substep_enabled_check.setChecked(orbit_substep_val is not None)
        self.attitude_substep_enabled_check.setChecked(attitude_substep_val is not None)
        self.orbit_substep_spin.setValue(float(orbit_substep_val or 0.0))
        self.attitude_substep_spin.setValue(float(attitude_substep_val or 0.0))
        self.attitude_enabled_check.setChecked(bool(att_dyn.get("enabled", True)))
        self.orbit_j2_check.setChecked(bool(orbit_dyn.get("j2", False)))
        self.orbit_j3_check.setChecked(bool(orbit_dyn.get("j3", False)))
        self.orbit_j4_check.setChecked(bool(orbit_dyn.get("j4", False)))
        self.orbit_drag_check.setChecked(bool(orbit_dyn.get("drag", False)))
        self.orbit_srp_check.setChecked(bool(orbit_dyn.get("srp", False)))
        self.orbit_moon_check.setChecked(bool(orbit_dyn.get("third_body_moon", False)))
        self.orbit_sun_check.setChecked(bool(orbit_dyn.get("third_body_sun", False)))
        self.att_gg_check.setChecked(bool(disturbance_torques.get("gravity_gradient", False)))
        self.att_magnetic_check.setChecked(bool(disturbance_torques.get("magnetic", False)))
        self.att_drag_check.setChecked(bool(disturbance_torques.get("drag", False)))
        self.att_srp_check.setChecked(bool(disturbance_torques.get("srp", False)))
        self._refresh_substep_visibility()

        target_specs = dict(target.get("specs", {}) or {})
        target_coes = dict(target.get("initial_state", {}).get("coes", {}) or {})
        self.target_enabled.setChecked(bool(target.get("enabled", True)))
        self._set_combo_text_or_append(self.target_preset, str(target_specs.get("preset_satellite", "") or SATELLITE_PRESET_OPTIONS[0]))
        target_mass_fallback = float(target_specs.get("mass_kg", 400.0) or 0.0)
        self.target_dry_mass.setValue(float(target_specs.get("dry_mass_kg", target_mass_fallback) or 0.0))
        self.target_fuel_mass.setValue(float(target_specs.get("fuel_mass_kg", 0.0) or 0.0))
        self.target_a.setValue(float(target_coes.get("a_km", 7000.0) or 7000.0))
        self.target_ecc.setValue(float(target_coes.get("ecc", 0.001) or 0.0))
        self.target_inc.setValue(float(target_coes.get("inc_deg", 45.0) or 0.0))
        self.target_raan.setValue(float(target_coes.get("raan_deg", 0.0) or 0.0))
        self.target_argp.setValue(float(target_coes.get("argp_deg", 0.0) or 0.0))
        self.target_ta.setValue(float(target_coes.get("true_anomaly_deg", 0.0) or 0.0))
        self._set_pointer_combo_value(self.target_strategy_combo, dict(target.get("mission_strategy", {}) or {}) if target.get("mission_strategy") else None)
        self._set_pointer_combo_value(self.target_execution_combo, dict(target.get("mission_execution", {}) or {}) if target.get("mission_execution") else None)
        self._set_pointer_combo_value(self.target_orbit_control_combo, dict(target.get("orbit_control", {}) or {}) if target.get("orbit_control") else None)
        self._set_pointer_combo_value(self.target_attitude_control_combo, dict(target.get("attitude_control", {}) or {}) if target.get("attitude_control") else None)

        chaser_specs = dict(chaser.get("specs", {}) or {})
        chaser_init = dict(chaser.get("initial_state", {}) or {})
        self.chaser_enabled.setChecked(bool(chaser.get("enabled", False)))
        self._set_combo_text_or_append(self.chaser_preset, str(chaser_specs.get("preset_satellite", "") or SATELLITE_PRESET_OPTIONS[0]))
        chaser_mass_fallback = float(chaser_specs.get("mass_kg", 200.0) or 0.0)
        self.chaser_dry_mass.setValue(float(chaser_specs.get("dry_mass_kg", chaser_mass_fallback) or 0.0))
        self.chaser_fuel_mass.setValue(float(chaser_specs.get("fuel_mass_kg", 0.0) or 0.0))
        rel_block = dict(chaser_init.get("relative_to_target_ric", {}) or {})
        if rel_block:
            frame = str(rel_block.get("frame", "rect") or "rect").strip().lower()
            self.chaser_init_mode.setCurrentText("relative_ric_curv" if frame == "curv" else "relative_ric_rect")
            values = list(rel_block.get("state", [0.0] * 6))
        elif "relative_ric_rect" in chaser_init:
            self.chaser_init_mode.setCurrentText("relative_ric_rect")
            values = list(chaser_init.get("relative_ric_rect", [0.0] * 6))
        elif "relative_ric_curv" in chaser_init:
            self.chaser_init_mode.setCurrentText("relative_ric_curv")
            values = list(chaser_init.get("relative_ric_curv", [0.0] * 6))
        else:
            self.chaser_init_mode.setCurrentText("rocket_deployment")
            values = list(chaser_init.get("deploy_dv_body_m_s", [10.0, 0.0, 0.0])) + [0.0, 0.0, 0.0]
        self.chaser_deploy_time.setValue(float(chaser_init.get("deploy_time_s", 900.0) or 0.0))
        for i, widget in enumerate(self.chaser_init_values):
            widget.setValue(float(values[i] if i < len(values) else 0.0))
        self._set_pointer_combo_value(self.chaser_strategy_combo, dict(chaser.get("mission_strategy", {}) or {}) if chaser.get("mission_strategy") else None)
        self._set_pointer_combo_value(self.chaser_execution_combo, dict(chaser.get("mission_execution", {}) or {}) if chaser.get("mission_execution") else None)
        self._set_pointer_combo_value(self.chaser_orbit_control_combo, dict(chaser.get("orbit_control", {}) or {}) if chaser.get("orbit_control") else None)
        self._set_pointer_combo_value(self.chaser_attitude_control_combo, dict(chaser.get("attitude_control", {}) or {}) if chaser.get("attitude_control") else None)

        rocket_specs = dict(rocket.get("specs", {}) or {})
        rocket_init = dict(rocket.get("initial_state", {}) or {})
        self.rocket_enabled.setChecked(bool(rocket.get("enabled", False)))
        self._set_combo_text_or_append(self.rocket_preset, str(rocket_specs.get("preset_stack", "") or ROCKET_PRESET_OPTIONS[0]))
        self.rocket_payload.setValue(float(rocket_specs.get("payload_mass_kg", 150.0) or 0.0))
        self.rocket_launch_lat.setValue(float(rocket_init.get("launch_lat_deg", 28.5) or 0.0))
        self.rocket_launch_lon.setValue(float(rocket_init.get("launch_lon_deg", -80.6) or 0.0))
        self.rocket_launch_alt.setValue(float(rocket_init.get("launch_alt_km", 0.0) or 0.0))
        self.rocket_launch_az.setValue(float(rocket_init.get("launch_azimuth_deg", 90.0) or 0.0))
        self._set_pointer_combo_value(self.rocket_strategy_combo, dict(rocket.get("mission_strategy", {}) or {}) if rocket.get("mission_strategy") else None)
        self._set_pointer_combo_value(self.rocket_execution_combo, dict(rocket.get("mission_execution", {}) or {}) if rocket.get("mission_execution") else None)
        rocket_base_guidance = dict(rocket.get("base_guidance", {}) or {}) if rocket.get("base_guidance") else None
        if rocket_base_guidance is None and rocket.get("guidance"):
            rocket_base_guidance = dict(rocket.get("guidance", {}) or {})
        self._set_pointer_combo_value(self.rocket_base_guidance_combo, rocket_base_guidance)
        self.rocket_guidance_modifiers_config = copy.deepcopy(list(rocket.get("guidance_modifiers", []) or []))
        self._refresh_rocket_guidance_modifiers_label()
        stats = dict(outputs.get("stats", {}) or {})
        plots = dict(outputs.get("plots", {}) or {})
        animations = dict(outputs.get("animations", {}) or {})
        mc_outputs = dict(outputs.get("monte_carlo", {}) or {})
        self.stats_enabled.setChecked(bool(stats.get("enabled", True)))
        self.stats_print_summary.setChecked(bool(stats.get("print_summary", True)))
        self.stats_save_json.setChecked(bool(stats.get("save_json", True)))
        self.stats_save_csv.setChecked(bool(stats.get("save_csv", False)))
        self.plots_enabled.setChecked(bool(plots.get("enabled", True)))
        self.plots_dpi.setValue(int(plots.get("dpi", 150) or 150))
        for check in self.figure_id_checks.values():
            check.setChecked(False)
        for figure_id in list(plots.get("figure_ids", []) or []):
            if figure_id in self.figure_id_checks:
                self.figure_id_checks[figure_id].setChecked(True)
        self.reference_object_edit.setText(str(plots.get("reference_object_id", "") or ""))
        for check in self.animation_type_checks.values():
            check.setChecked(False)
        for anim_type in list(animations.get("types", []) or []):
            if anim_type in self.animation_type_checks:
                self.animation_type_checks[anim_type].setChecked(True)
        self.animation_fps_spin.setValue(float(animations.get("fps", 30.0) or 30.0))
        self.animation_speed_multiple_spin.setValue(float(animations.get("speed_multiple", 10.0) or 10.0))
        self.animation_frame_stride_spin.setValue(int(animations.get("frame_stride", 1) or 1))
        self.mc_save_iteration_summaries.setChecked(bool(mc_outputs.get("save_iteration_summaries", False)))
        self.mc_save_aggregate_summary.setChecked(bool(mc_outputs.get("save_aggregate_summary", True)))
        self.mc_save_histograms.setChecked(bool(mc_outputs.get("save_histograms", False)))
        self.mc_display_histograms.setChecked(bool(mc_outputs.get("display_histograms", False)))
        self.mc_save_ops_dashboard.setChecked(bool(mc_outputs.get("save_ops_dashboard", True)))
        self.mc_display_ops_dashboard.setChecked(bool(mc_outputs.get("display_ops_dashboard", False)))
        self.mc_save_raw_runs.setChecked(bool(mc_outputs.get("save_raw_runs", False)))
        self.mc_require_rocket_insertion.setChecked(bool(mc_outputs.get("require_rocket_insertion", False)))
        self.mc_baseline_summary_json.setText(str(mc_outputs.get("baseline_summary_json", "") or ""))
        gates = dict(mc_outputs.get("gates", {}) or {})
        self.mc_gate_min_closest_approach.setValue(float(gates.get("min_closest_approach_km", 0.0) or 0.0))
        self.mc_gate_max_duration.setValue(float(gates.get("max_duration_s", 0.0) or 0.0))
        self.mc_gate_max_total_dv.setValue(float(gates.get("max_total_dv_m_s", 0.0) or 0.0))
        self.mc_gate_max_guardrail_events.setValue(float(gates.get("max_guardrail_events", 0.0) or 0.0))
        self._refresh_outputs_mode_ui()
        self._suppress_dirty_tracking = False

    def _collect_config_from_widgets(self) -> dict:
        cfg = dict(self.current_config)
        cfg["scenario_name"] = self.scenario_name_edit.text().strip()
        sim = cfg.setdefault("simulator", {})
        outputs = cfg.setdefault("outputs", {})
        mc = cfg.setdefault("monte_carlo", {})
        target = cfg.setdefault("target", {})
        chaser = cfg.setdefault("chaser", {})
        rocket = cfg.setdefault("rocket", {})

        sim["duration_s"] = float(self.duration_spin.value())
        sim["dt_s"] = float(self.dt_spin.value())
        dynamics = sim.setdefault("dynamics", {})
        orbit_dyn = dynamics.setdefault("orbit", {})
        att_dyn = dynamics.setdefault("attitude", {})
        disturbance_torques = att_dyn.setdefault("disturbance_torques", {})

        orbit_substep = float(self.orbit_substep_spin.value())
        attitude_substep = float(self.attitude_substep_spin.value())
        orbit_dyn["orbit_substep_s"] = orbit_substep if (self.orbit_substep_enabled_check.isChecked() and orbit_substep > 0.0) else None
        att_dyn["attitude_substep_s"] = attitude_substep if (self.attitude_substep_enabled_check.isChecked() and attitude_substep > 0.0) else None
        att_dyn["enabled"] = bool(self.attitude_enabled_check.isChecked())
        orbit_dyn["j2"] = bool(self.orbit_j2_check.isChecked())
        orbit_dyn["j3"] = bool(self.orbit_j3_check.isChecked())
        orbit_dyn["j4"] = bool(self.orbit_j4_check.isChecked())
        orbit_dyn["drag"] = bool(self.orbit_drag_check.isChecked())
        orbit_dyn["srp"] = bool(self.orbit_srp_check.isChecked())
        orbit_dyn["third_body_moon"] = bool(self.orbit_moon_check.isChecked())
        orbit_dyn["third_body_sun"] = bool(self.orbit_sun_check.isChecked())
        disturbance_torques["gravity_gradient"] = bool(self.att_gg_check.isChecked())
        disturbance_torques["magnetic"] = bool(self.att_magnetic_check.isChecked())
        disturbance_torques["drag"] = bool(self.att_drag_check.isChecked())
        disturbance_torques["srp"] = bool(self.att_srp_check.isChecked())

        outputs["mode"] = self.output_mode_combo.currentText()
        outputs["output_dir"] = self.output_dir_edit.text().strip()
        mc["enabled"] = bool(self.mc_enabled_check.isChecked())
        mc["iterations"] = int(self.mc_iterations_spin.value())
        mc["parallel_enabled"] = bool(self.mc_parallel_check.isChecked())
        mc["parallel_workers"] = int(self.mc_workers_spin.value())
        mc["base_seed"] = int(self.mc_base_seed_spin.value())
        mc["variations"] = copy.deepcopy(self.mc_variations)

        target["enabled"] = bool(self.target_enabled.isChecked())
        target.setdefault("specs", {})["preset_satellite"] = self.target_preset.currentText().strip()
        target["specs"]["dry_mass_kg"] = float(self.target_dry_mass.value())
        target["specs"]["fuel_mass_kg"] = float(self.target_fuel_mass.value())
        target["specs"].pop("mass_kg", None)
        target.setdefault("initial_state", {})["coes"] = {
            "a_km": float(self.target_a.value()),
            "ecc": float(self.target_ecc.value()),
            "inc_deg": float(self.target_inc.value()),
            "raan_deg": float(self.target_raan.value()),
            "argp_deg": float(self.target_argp.value()),
            "true_anomaly_deg": float(self.target_ta.value()),
        }
        target["mission_strategy"] = self._combo_pointer_value(self.target_strategy_combo, existing=dict(target.get("mission_strategy", {}) or {}) if target.get("mission_strategy") else None)
        target["mission_execution"] = self._combo_pointer_value(self.target_execution_combo, existing=dict(target.get("mission_execution", {}) or {}) if target.get("mission_execution") else None)
        target.pop("guidance", None)
        target["orbit_control"] = self._combo_pointer_value(self.target_orbit_control_combo, existing=dict(target.get("orbit_control", {}) or {}) if target.get("orbit_control") else None)
        target["attitude_control"] = self._combo_pointer_value(self.target_attitude_control_combo, existing=dict(target.get("attitude_control", {}) or {}) if target.get("attitude_control") else None)

        chaser["enabled"] = bool(self.chaser_enabled.isChecked())
        chaser.setdefault("specs", {})["preset_satellite"] = self.chaser_preset.currentText().strip()
        chaser["specs"]["dry_mass_kg"] = float(self.chaser_dry_mass.value())
        chaser["specs"]["fuel_mass_kg"] = float(self.chaser_fuel_mass.value())
        chaser["specs"].pop("mass_kg", None)
        init_mode = self.chaser_init_mode.currentText()
        chaser_initial_state = dict(chaser.get("initial_state", {}) or {})
        if init_mode == "rocket_deployment":
            chaser_initial_state["source"] = "rocket_deployment"
            chaser_initial_state["deploy_time_s"] = float(self.chaser_deploy_time.value())
            chaser_initial_state["deploy_dv_body_m_s"] = [float(self.chaser_init_values[i].value()) for i in range(3)]
            chaser_initial_state.pop("relative_to_target_ric", None)
            chaser_initial_state.pop("relative_ric_rect", None)
            chaser_initial_state.pop("relative_ric_curv", None)
        else:
            chaser_initial_state.pop("source", None)
            chaser_initial_state.pop("deploy_time_s", None)
            chaser_initial_state.pop("deploy_dv_body_m_s", None)
            chaser_initial_state["relative_to_target_ric"] = {
                "frame": "rect" if init_mode == "relative_ric_rect" else "curv",
                "state": [float(widget.value()) for widget in self.chaser_init_values],
            }
            chaser_initial_state.pop("relative_ric_rect", None)
            chaser_initial_state.pop("relative_ric_curv", None)
        chaser["initial_state"] = chaser_initial_state
        chaser["mission_strategy"] = self._combo_pointer_value(self.chaser_strategy_combo, existing=dict(chaser.get("mission_strategy", {}) or {}) if chaser.get("mission_strategy") else None)
        chaser["mission_execution"] = self._combo_pointer_value(self.chaser_execution_combo, existing=dict(chaser.get("mission_execution", {}) or {}) if chaser.get("mission_execution") else None)
        chaser.pop("guidance", None)
        chaser["orbit_control"] = self._combo_pointer_value(self.chaser_orbit_control_combo, existing=dict(chaser.get("orbit_control", {}) or {}) if chaser.get("orbit_control") else None)
        chaser["attitude_control"] = self._combo_pointer_value(self.chaser_attitude_control_combo, existing=dict(chaser.get("attitude_control", {}) or {}) if chaser.get("attitude_control") else None)

        rocket["enabled"] = bool(self.rocket_enabled.isChecked())
        rocket.setdefault("specs", {})["preset_stack"] = self.rocket_preset.currentText().strip()
        rocket["specs"]["payload_mass_kg"] = float(self.rocket_payload.value())
        rocket["initial_state"] = {
            "launch_lat_deg": float(self.rocket_launch_lat.value()),
            "launch_lon_deg": float(self.rocket_launch_lon.value()),
            "launch_alt_km": float(self.rocket_launch_alt.value()),
            "launch_azimuth_deg": float(self.rocket_launch_az.value()),
        }
        rocket["mission_strategy"] = self._combo_pointer_value(self.rocket_strategy_combo, existing=dict(rocket.get("mission_strategy", {}) or {}) if rocket.get("mission_strategy") else None)
        rocket["mission_execution"] = self._combo_pointer_value(self.rocket_execution_combo, existing=dict(rocket.get("mission_execution", {}) or {}) if rocket.get("mission_execution") else None)
        rocket["base_guidance"] = self._combo_pointer_value(
            self.rocket_base_guidance_combo,
            existing=dict(rocket.get("base_guidance", {}) or {}) if rocket.get("base_guidance") else (
                dict(rocket.get("guidance", {}) or {}) if rocket.get("guidance") else None
            ),
        )
        rocket["guidance_modifiers"] = copy.deepcopy(self.rocket_guidance_modifiers_config)
        rocket.pop("guidance", None)
        rocket.pop("orbit_control", None)
        rocket.pop("attitude_control", None)

        stats = outputs.setdefault("stats", {})
        plots = outputs.setdefault("plots", {})
        animations = outputs.setdefault("animations", {})
        mc_outputs = outputs.setdefault("monte_carlo", {})
        stats["enabled"] = bool(self.stats_enabled.isChecked())
        stats["print_summary"] = bool(self.stats_print_summary.isChecked())
        stats["save_json"] = bool(self.stats_save_json.isChecked())
        stats["save_csv"] = bool(self.stats_save_csv.isChecked())
        plots["enabled"] = bool(self.plots_enabled.isChecked())
        plots["dpi"] = int(self.plots_dpi.value())
        figure_ids = [figure_id for figure_id, check in self.figure_id_checks.items() if check.isChecked()]
        plots["figure_ids"] = figure_ids
        ref_obj = self.reference_object_edit.text().strip()
        if ref_obj:
            plots["reference_object_id"] = ref_obj
        else:
            plots.pop("reference_object_id", None)
        animation_types = [anim_type for anim_type, check in self.animation_type_checks.items() if check.isChecked()]
        animations["enabled"] = bool(animation_types)
        animations["types"] = animation_types
        animations["fps"] = float(self.animation_fps_spin.value())
        animations["speed_multiple"] = float(self.animation_speed_multiple_spin.value())
        animations["frame_stride"] = int(self.animation_frame_stride_spin.value())
        mc_outputs["save_iteration_summaries"] = bool(self.mc_save_iteration_summaries.isChecked())
        mc_outputs["save_aggregate_summary"] = bool(self.mc_save_aggregate_summary.isChecked())
        mc_outputs["save_histograms"] = bool(self.mc_save_histograms.isChecked())
        mc_outputs["display_histograms"] = bool(self.mc_display_histograms.isChecked())
        mc_outputs["save_ops_dashboard"] = bool(self.mc_save_ops_dashboard.isChecked())
        mc_outputs["display_ops_dashboard"] = bool(self.mc_display_ops_dashboard.isChecked())
        mc_outputs["save_raw_runs"] = bool(self.mc_save_raw_runs.isChecked())
        mc_outputs["require_rocket_insertion"] = bool(self.mc_require_rocket_insertion.isChecked())
        mc_outputs["baseline_summary_json"] = self.mc_baseline_summary_json.text().strip()
        gates = {}
        if float(self.mc_gate_min_closest_approach.value()) != 0.0:
            gates["min_closest_approach_km"] = float(self.mc_gate_min_closest_approach.value())
        if float(self.mc_gate_max_duration.value()) != 0.0:
            gates["max_duration_s"] = float(self.mc_gate_max_duration.value())
        if float(self.mc_gate_max_total_dv.value()) != 0.0:
            gates["max_total_dv_m_s"] = float(self.mc_gate_max_total_dv.value())
        if float(self.mc_gate_max_guardrail_events.value()) != 0.0:
            gates["max_guardrail_events"] = float(self.mc_gate_max_guardrail_events.value())
        if gates:
            mc_outputs["gates"] = gates
        else:
            mc_outputs.pop("gates", None)
        return cfg

    def _refresh_yaml(self) -> None:
        try:
            self.current_config = self._collect_config_from_widgets()
        except Exception:
            pass
        self._suppress_dirty_tracking = True
        self.yaml_editor.setPlainText(dump_config_text(self.current_config))
        self._suppress_dirty_tracking = False

    def _apply_yaml_to_form(self) -> None:
        try:
            cfg = parse_config_text(self.yaml_editor.toPlainText())
            validate_config(cfg)
            self.current_config = cfg
            self.results_output_dir = None
            self._load_config_into_widgets(cfg)
            self._refresh_validation_state()
            self._refresh_output_files()
            self.statusBar().showMessage("Applied YAML to form.", 5000)
        except Exception as exc:
            self._show_error("Apply YAML Failed", str(exc))

    def _validate_yaml(self) -> None:
        try:
            cfg = parse_config_text(self.yaml_editor.toPlainText())
            validate_config(cfg)
            self.statusBar().showMessage("YAML is valid.", 5000)
        except Exception as exc:
            self._show_error("Validation Failed", str(exc))

    def _refresh_validation_state(self) -> None:
        try:
            cfg = validate_config(self._collect_config_from_widgets())
            self.current_config = cfg.to_dict()
            self._refresh_validation_panel(valid=True, issues=self._collect_validation_messages(self.current_config))
        except Exception as exc:
            self._refresh_validation_panel(valid=False, issues=[str(exc)])

    def _refresh_output_files(self) -> None:
        self.output_files.clear()
        self.preview_image_path = None
        self.preview_title.setText("Select an artifact to preview.")
        self.preview_image.setText("No image selected.")
        self.preview_image.setPixmap(QPixmap())
        self.preview_zoom_factor = 1.0
        self.preview_fit_to_window = True
        self.zoom_label.setText("Fit")
        self.preview_text.clear()
        self.results_summary.clear()
        try:
            cfg = validate_config(self._collect_config_from_widgets())
            output_dir = self.results_output_dir or self._resolve_output_dir(cfg.outputs.output_dir)
            files = get_output_files(output_dir)
            for path in files:
                label = self._artifact_label(path)
                item_text = label
                self.output_files.addItem(item_text)
                self.output_files.item(self.output_files.count() - 1).setData(Qt.UserRole, str(path))
            self._refresh_results_summary(output_dir, files, used_temp_dir=self.results_output_dir is not None)
        except Exception:
            return

    def _refresh_results_summary(self, output_dir: Path, files: list[Path], *, used_temp_dir: bool) -> None:
        summary_path = output_dir / "master_run_summary.json"
        mc_summary_path = output_dir / "master_monte_carlo_summary.json"
        text = []
        if used_temp_dir:
            text.append(
                "GUI preview cache\n"
                "=================\n"
                "This run used `outputs.mode: interactive`, so the GUI redirected plot artifacts to a temporary "
                f"preview directory instead of your normal output folder:\n{output_dir}"
            )
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                text.append(self._format_json_summary("Single Run Summary", data))
            except Exception as exc:
                text.append(f"Failed to read {summary_path.name}: {exc}")
        if mc_summary_path.exists():
            try:
                data = json.loads(mc_summary_path.read_text(encoding="utf-8"))
                text.append(self._format_json_summary("Monte Carlo Summary", data))
            except Exception as exc:
                text.append(f"Failed to read {mc_summary_path.name}: {exc}")
        if not text:
            png_count = sum(1 for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            json_count = sum(1 for p in files if p.suffix.lower() == ".json")
            text.append(
                f"Output directory: {output_dir}\n"
                f"Artifacts found: {len(files)}\n"
                f"Images: {png_count}\n"
                f"JSON files: {json_count}\n"
                "No recognized run summary file found yet."
            )
        self.results_summary.setPlainText("\n\n".join(text))

    def _format_json_summary(self, title: str, data: dict) -> str:
        lines = [title, "=" * len(title)]
        for key in (
            "scenario_name",
            "samples",
            "dt_s",
            "duration_s",
            "terminated_early",
            "termination_reason",
            "termination_time_s",
            "termination_object_id",
            "rocket_insertion_achieved",
            "rocket_insertion_time_s",
            "pass_rate",
            "fail_rate",
            "duration_s_mean",
            "closest_approach_km_mean",
            "p_catastrophic_outcome",
        ):
            if key in data:
                lines.append(f"{key}: {data.get(key)}")
        remaining = {k: v for k, v in data.items() if k not in {line.split(':', 1)[0] for line in lines[2:] if ': ' in line}}
        if remaining:
            lines.append("")
            lines.append(json.dumps(remaining, indent=2)[:4000])
        return "\n".join(lines)

    def _on_output_file_selected(self, _text: str) -> None:
        item = self.output_files.currentItem()
        if item is None:
            return
        path_data = item.data(Qt.UserRole)
        if not path_data:
            return
        path = Path(str(path_data))
        self.preview_title.setText(self._artifact_label(path))
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
            self.preview_stack.setCurrentIndex(0)
            self.preview_text.clear()
            self.preview_image_path = path
            self.preview_zoom_factor = 1.0
            self.preview_fit_to_window = True
            self._update_image_preview()
            return
        self.preview_stack.setCurrentIndex(1)
        self.preview_image_path = None
        self.preview_image.setText("Preview available in Text tab.")
        self.preview_image.setPixmap(QPixmap())
        if suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.preview_text.setPlainText(json.dumps(data, indent=2)[:20000])
            except Exception as exc:
                self.preview_text.setPlainText(f"Failed to read JSON: {exc}")
            return
        try:
            self.preview_text.setPlainText(path.read_text(encoding="utf-8")[:20000])
        except Exception as exc:
            self.preview_text.setPlainText(f"Preview not available: {exc}")

    def _update_image_preview(self) -> None:
        if self.preview_image_path is None or not self.preview_image_path.exists():
            self.preview_image.setText("Image file not found.")
            self.preview_image.setPixmap(QPixmap())
            return
        pixmap = QPixmap(str(self.preview_image_path))
        if pixmap.isNull():
            self.preview_image.setText("Could not render image preview.")
            self.preview_image.setPixmap(QPixmap())
            return
        if self.preview_fit_to_window:
            viewport = self.preview_scroll.viewport().size()
            scaled = pixmap.scaled(
                max(viewport.width() - 16, 100),
                max(viewport.height() - 16, 100),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.zoom_label.setText("Fit")
            self.preview_image.setCursor(Qt.ArrowCursor)
        else:
            width = max(int(pixmap.width() * self.preview_zoom_factor), 1)
            height = max(int(pixmap.height() * self.preview_zoom_factor), 1)
            scaled = pixmap.scaled(
                width,
                height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.zoom_label.setText(f"{int(self.preview_zoom_factor * 100)}%")
            self.preview_image.setCursor(Qt.OpenHandCursor)
        self.preview_image.setPixmap(scaled)
        self.preview_image.resize(scaled.size())
        self.preview_image.setMinimumSize(scaled.size())

    def _resolve_output_dir(self, output_dir: str) -> Path:
        path = Path(output_dir)
        return path if path.is_absolute() else (self.repo_root / path)

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo_root))
        except ValueError:
            return str(path)

    def _artifact_label(self, path: Path) -> str:
        stem = path.stem.replace("_", " ").strip()
        if not stem:
            stem = path.name
        label = stem.title()
        if path.suffix:
            label = f"{label} ({path.suffix.lower().lstrip('.')})"
        return label

    def _path_from_display(self, path_text: str) -> Path:
        path = Path(path_text)
        return path if path.is_absolute() else (self.repo_root / path)

    def _load_selected_config_path(self, path: Path) -> None:
        if not self._prompt_discard_changes():
            self._restore_config_selector()
            return
        self.loaded_config_path = path
        self.current_config = load_config(path)
        self.save_path_edit.setText(str(path))
        self.results_output_dir = None
        self._load_config_into_widgets(self.current_config)
        self._refresh_yaml()
        self._refresh_validation_state()
        self._refresh_output_files()
        self._set_dirty(False)
        self._update_window_title()
        self.statusBar().showMessage(f"Loaded {path}", 5000)

    def _on_config_selected(self, index: int) -> None:
        if self._suppress_config_selector_load:
            return
        if index < 0:
            return
        data = self.config_selector.itemData(index)
        if not data:
            return
        path = Path(data)
        if path == self.loaded_config_path:
            return
        self._load_selected_config_path(path)

    def _on_open_file(self) -> None:
        if not self._prompt_discard_changes():
            return
        path_str, _ = QFileDialog.getOpenFileName(self, "Open Config", str(self.repo_root), "YAML Files (*.yaml *.yml)")
        if not path_str:
            return
        path = Path(path_str)
        self.loaded_config_path = path
        self.current_config = load_config(path)
        self.save_path_edit.setText(str(path))
        self.results_output_dir = None
        self._sync_config_selector_to_path(path)
        self._load_config_into_widgets(self.current_config)
        self._refresh_yaml()
        self._refresh_validation_state()
        self._refresh_output_files()
        self._set_dirty(False)
        self._update_window_title()
        self.statusBar().showMessage(f"Loaded {path}", 5000)

    def _on_save(self) -> None:
        try:
            cfg_dict = self._collect_config_from_widgets()
            save_path = save_config(self.save_path_edit.text().strip(), cfg_dict)
            self.loaded_config_path = save_path
            self.current_config = load_config(save_path)
            self.save_path_edit.setText(str(save_path))
            self.results_output_dir = None
            self._sync_config_selector_to_path(save_path)
            self._refresh_yaml()
            self._refresh_validation_state()
            self._set_dirty(False)
            self._update_window_title()
            self.statusBar().showMessage(f"Saved {save_path}", 5000)
        except Exception as exc:
            self._show_error("Save Failed", str(exc))

    def _on_save_as(self) -> None:
        start_dir = str(self.loaded_config_path.parent if self.loaded_config_path.exists() else (self.repo_root / "configs"))
        path_str, _ = QFileDialog.getSaveFileName(self, "Save Config As", start_dir, "YAML Files (*.yaml *.yml)")
        if not path_str:
            return
        self.save_path_edit.setText(path_str)
        self._on_save()

    def _on_new(self) -> None:
        if not self._prompt_discard_changes():
            return
        new_cfg = load_config(get_default_config_path())
        self.loaded_config_path = self.repo_root / "configs" / "untitled_gui_config.yaml"
        self.current_config = new_cfg
        self.save_path_edit.setText(str(self.loaded_config_path))
        self.results_output_dir = None
        self._load_config_into_widgets(new_cfg)
        self._refresh_yaml()
        self._refresh_validation_state()
        self._refresh_output_files()
        self._set_dirty(True)
        self._update_window_title()
        self.statusBar().showMessage("Started a new config from the default template.", 5000)

    def _on_run(self) -> None:
        if self.process is not None:
            self._show_error("Run In Progress", "A simulation is already running.")
            return
        try:
            cfg_dict = self._collect_config_from_widgets()
            save_path = save_config(self.save_path_edit.text().strip(), cfg_dict)
            run_config_path = save_path
            self.results_output_dir = None
            if str(cfg_dict.get("outputs", {}).get("mode", "")).strip().lower() == "interactive":
                run_config_path = self._build_preview_run_config(cfg_dict, save_path)
            cmd = build_cli_run_command(run_config_path)
            self.console.clear()
            self.output_files.clear()
            self.process = QProcess(self)
            self.process.setWorkingDirectory(str(self.repo_root))
            self.process.setProcessChannelMode(QProcess.MergedChannels)
            process_env = QProcessEnvironment.systemEnvironment()
            process_env.insert("NONCOOP_GUI", "1")
            self.process.setProcessEnvironment(process_env)
            self.process.readyReadStandardOutput.connect(self._append_process_output)
            self.process.finished.connect(self._on_process_finished)
            self.process.start(cmd[0], cmd[1:])
            self.run_button.setEnabled(False)
            self.statusBar().showMessage("Simulation running...")
            self.tabs.setCurrentIndex(4)
        except Exception as exc:
            self._show_error("Run Failed", str(exc))

    def _append_process_output(self) -> None:
        if self.process is None:
            return
        txt = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if txt:
            self.console.moveCursor(QTextCursor.End)
            self.console.insertPlainText(txt)
            self.console.ensureCursorVisible()

    def _on_process_finished(self, exit_code: int, _status) -> None:
        self.run_button.setEnabled(True)
        self.statusBar().showMessage(f"Simulation finished with code {exit_code}.", 10000)
        self._refresh_output_files()
        self.process = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.preview_image_path is not None:
            self._update_image_preview()

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 10000)

    def _build_preview_run_config(self, cfg_dict: dict, saved_config_path: Path) -> Path:
        self.preview_temp_dir = tempfile.TemporaryDirectory(prefix="noncoop_gui_preview_")
        preview_root = Path(self.preview_temp_dir.name)
        run_cfg = copy.deepcopy(cfg_dict)
        run_cfg.setdefault("outputs", {})
        run_cfg["outputs"]["mode"] = "save"
        run_cfg["outputs"]["output_dir"] = str(preview_root / "artifacts")
        temp_config_path = preview_root / f"{saved_config_path.stem}_preview.yaml"
        save_config(temp_config_path, run_cfg)
        self.results_output_dir = preview_root / "artifacts"
        self.statusBar().showMessage(
            "Interactive run redirected to a temporary preview cache so plots can be shown in the GUI.",
            10000,
        )
        return temp_config_path

    def _monte_carlo_path_warning(self, cfg_dict: dict) -> str:
        mc = dict(cfg_dict.get("monte_carlo", {}) or {})
        if not bool(mc.get("enabled", False)):
            return ""
        for variation in list(mc.get("variations", []) or []):
            path = str(dict(variation or {}).get("parameter_path", "") or "").strip()
            if path and not self._path_exists(cfg_dict, path):
                return f"Warning: MC path missing: {path}"
        return ""

    def _path_exists(self, root: dict, path: str) -> bool:
        cur = root
        for tok in path.split("."):
            if "[" in tok and tok.endswith("]"):
                key, idx_txt = tok[:-1].split("[", 1)
                idx = int(idx_txt)
                if key:
                    if not isinstance(cur, dict) or key not in cur:
                        return False
                    cur = cur[key]
                if not isinstance(cur, list) or idx >= len(cur):
                    return False
                cur = cur[idx]
                continue
            if not isinstance(cur, dict) or tok not in cur:
                return False
            cur = cur[tok]
        return True

    def _mark_dirty(self, *_args) -> None:
        if self._suppress_dirty_tracking:
            return
        self._set_dirty(True)

    def _set_dirty(self, is_dirty: bool) -> None:
        self.is_dirty = bool(is_dirty)
        self._update_window_title()

    def _update_window_title(self) -> None:
        marker = "*" if self.is_dirty else ""
        self.setWindowTitle(f"{marker}{self.loaded_config_path.name} - NonCooperativeRPO Operator Console")

    def _prompt_discard_changes(self) -> bool:
        if not self.is_dirty:
            return True
        result = QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Discard them?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return result == QMessageBox.Yes

    def _sync_config_selector_to_path(self, path: Path) -> None:
        try:
            rel = str(path.resolve().relative_to(self.repo_root))
        except ValueError:
            return
        idx = self.config_selector.findText(rel)
        if idx >= 0:
            self._suppress_config_selector_load = True
            self.config_selector.setCurrentIndex(idx)
            self._suppress_config_selector_load = False

    def _restore_config_selector(self) -> None:
        self._sync_config_selector_to_path(self.loaded_config_path)

    def _zoom_in_preview(self) -> None:
        if self.preview_image_path is None:
            return
        if self.preview_fit_to_window:
            self.preview_fit_to_window = False
            self.preview_zoom_factor = 1.0
        self.preview_zoom_factor = min(self.preview_zoom_factor * 1.25, 8.0)
        self._update_image_preview()

    def _zoom_out_preview(self) -> None:
        if self.preview_image_path is None:
            return
        if self.preview_fit_to_window:
            self.preview_fit_to_window = False
            self.preview_zoom_factor = 1.0
        self.preview_zoom_factor = max(self.preview_zoom_factor / 1.25, 0.1)
        self._update_image_preview()

    def _fit_preview_image(self) -> None:
        if self.preview_image_path is None:
            return
        self.preview_fit_to_window = True
        self._update_image_preview()

    def _actual_size_preview(self) -> None:
        if self.preview_image_path is None:
            return
        self.preview_fit_to_window = False
        self.preview_zoom_factor = 1.0
        self._update_image_preview()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.preview_scroll.viewport() and self.preview_image_path is not None:
            if event.type() == QEvent.Wheel:
                delta_y = event.angleDelta().y()
                if delta_y > 0:
                    self._zoom_in_preview()
                elif delta_y < 0:
                    self._zoom_out_preview()
                return True
        if watched is self.preview_image and self.preview_image_path is not None and not self.preview_fit_to_window:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.preview_drag_active = True
                self.preview_drag_last_pos = event.globalPosition().toPoint()
                self.preview_image.setCursor(Qt.ClosedHandCursor)
                return True
            if event.type() == QEvent.MouseMove and self.preview_drag_active and self.preview_drag_last_pos is not None:
                current_pos = event.globalPosition().toPoint()
                delta = current_pos - self.preview_drag_last_pos
                self.preview_drag_last_pos = current_pos
                hbar = self.preview_scroll.horizontalScrollBar()
                vbar = self.preview_scroll.verticalScrollBar()
                hbar.setValue(hbar.value() - delta.x())
                vbar.setValue(vbar.value() - delta.y())
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.preview_drag_active = False
                self.preview_drag_last_pos = None
                self.preview_image.setCursor(Qt.OpenHandCursor)
                return True
        if watched is self.preview_image and event.type() == QEvent.Enter and self.preview_image_path is not None:
            self.preview_image.setCursor(Qt.OpenHandCursor if not self.preview_fit_to_window else Qt.ArrowCursor)
        if watched is self.preview_image and event.type() == QEvent.Leave and not self.preview_drag_active:
            self.preview_image.setCursor(Qt.ArrowCursor)
        return super().eventFilter(watched, event)

    def _on_navigation_selected(self) -> None:
        if not hasattr(self, "tabs"):
            return
        item = self.navigation_tree.currentItem()
        if item is None:
            return
        tab_index = item.data(0, Qt.UserRole)
        if isinstance(tab_index, int) and 0 <= tab_index < self.tabs.count():
            self.tabs.setCurrentIndex(tab_index)

    def _sync_navigation_to_tab(self, tab_index: int) -> None:
        root_item = self.navigation_tree.topLevelItem(tab_index) if 0 <= tab_index < self.navigation_tree.topLevelItemCount() else None
        if root_item is not None:
            self.navigation_tree.setCurrentItem(root_item)

    def _collect_validation_messages(self, cfg_dict: dict) -> list[str]:
        issues: list[str] = []
        mc_warn = self._monte_carlo_path_warning(cfg_dict)
        if mc_warn:
            issues.append(mc_warn)
        mode = str(cfg_dict.get("outputs", {}).get("mode", "")).strip().lower()
        if mode == "interactive":
            issues.append("Interactive mode in the GUI uses a temporary preview cache for plots.")
        return issues

    def _refresh_validation_panel(self, *, valid: bool, issues: list[str]) -> None:
        if valid and not issues:
            self.validation_label.setText("Config valid. No warnings.")
            self.validation_panel.setPlainText("")
            self.validation_panel.hide()
            self.validation_toggle.setText("Show Details")
            self._collapse_validation_on_startup = False
            return
        if valid:
            self.validation_label.setText(f"Config valid with {len(issues)} warning(s).")
        else:
            self.validation_label.setText("Config invalid.")
        self.validation_panel.setPlainText("\n".join(issues))
        if self._collapse_validation_on_startup:
            self.validation_panel.hide()
            self.validation_toggle.setText("Show Details")
            self._collapse_validation_on_startup = False
            return
        self.validation_panel.show()
        self.validation_toggle.setText("Hide Details")

    def _toggle_validation_panel(self) -> None:
        if self.validation_panel.isVisible():
            self.validation_panel.hide()
            self.validation_toggle.setText("Show Details")
        else:
            self.validation_panel.show()
            self.validation_toggle.setText("Hide Details")

    def _refresh_outputs_mode_ui(self) -> None:
        mc_enabled = bool(self.mc_enabled_check.isChecked())
        if mc_enabled:
            self.outputs_stack.setCurrentIndex(1)
            self.outputs_mode_label.setText(
                "Monte Carlo is enabled. Configure campaign outputs, dashboards, and pass/fail gates here."
            )
        else:
            self.outputs_stack.setCurrentIndex(0)
            self.outputs_mode_label.setText(
                "Monte Carlo is disabled. Configure single-run plots, stats, and animations here."
            )

    def _refresh_substep_visibility(self) -> None:
        self.orbit_substep_spin.setVisible(bool(self.orbit_substep_enabled_check.isChecked()))
        self.attitude_substep_spin.setVisible(bool(self.attitude_substep_enabled_check.isChecked()))
