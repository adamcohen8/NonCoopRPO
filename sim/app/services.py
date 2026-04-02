from __future__ import annotations

import importlib
from pathlib import Path
import time
from typing import Any

from sim.app.models import ConfigSummary
from sim.app.models import AnalysisUiProfile
from sim.app.models import GuiCapabilities
from sim.app.models import RunResult
from sim.app.io import (
    CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    build_run_command,
    dump_yaml_text,
    ensure_sections,
    list_config_files,
    list_output_files,
    load_config_dict,
    parse_yaml_text,
    read_yaml_file,
    run_simulation_cli,
    save_config_dict,
    validate_config_dict,
)
from sim.api import SimulationSession
from sim.config.scenario_yaml import SimulationScenarioConfig
from sim.master_outputs import AVAILABLE_ANIMATION_TYPES, AVAILABLE_FIGURE_IDS

PARAMETER_FORM_SCHEMAS: dict[str, list[dict[str, Any]]] = {
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

MONTE_CARLO_PARAMETER_CATEGORIES: dict[str, list[tuple[str, str]]] = {
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

ANALYSIS_UI_PROFILES: dict[str, AnalysisUiProfile] = {
    "monte_carlo": AnalysisUiProfile(
        count_label="Iterations",
        seed_label="Base Seed",
        inputs_title="Monte Carlo Variations",
        editor_title="Variation Editor",
        help_text="Configure random variations, iteration count, and campaign execution settings for Monte Carlo studies.",
        mode_label="Mode",
    ),
    "sensitivity_lhs": AnalysisUiProfile(
        count_label="Samples",
        seed_label="Seed",
        inputs_title="LHS Parameters",
        editor_title="Distribution Editor",
        help_text="Latin hypercube sampling draws one stratified sample per parameter per run. Use uniform or normal distributions.",
        mode_label="Distribution",
    ),
    "sensitivity_one_at_a_time": AnalysisUiProfile(
        count_label="Runs (auto)",
        seed_label="Seed",
        inputs_title="Sensitivity Parameters",
        editor_title="Parameter Editor",
        help_text="One-at-a-time sensitivity varies one parameter at a time. Choice mode uses explicit values; uniform/normal provide quick helpers.",
        mode_label="Mode",
    ),
}


def _pointer(module: str, class_name: str) -> dict[str, Any]:
    return {
        "kind": "python",
        "module": module,
        "class_name": class_name,
        "params": {},
    }


def _discover_pointer_options(
    specs: list[tuple[str, str, str]],
    *,
    include_none: bool = False,
) -> list[tuple[str, dict[str, Any] | None]]:
    options: list[tuple[str, dict[str, Any] | None]] = [("None", None)] if include_none else []
    for label, module_name, class_name in specs:
        try:
            module = importlib.import_module(module_name)
            if getattr(module, class_name, None) is None:
                continue
        except Exception:
            continue
        options.append((label, _pointer(module_name, class_name)))
    return options


def _discover_named_presets(module_name: str, type_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    preset_type = getattr(module, type_name)
    names = [
        name
        for name, value in vars(module).items()
        if name.isupper() and isinstance(value, preset_type)
    ]
    return sorted(names)


def get_gui_capabilities() -> GuiCapabilities:
    return GuiCapabilities(
        output_modes=["interactive", "save", "both"],
        orbit_integrators=["rk4", "rkf78", "dopri5"],
        analysis_study_types=[
            ("monte_carlo", "Monte Carlo"),
            ("sensitivity", "Sensitivity"),
        ],
        sensitivity_methods=[
            ("one_at_a_time", "One-at-a-Time"),
            ("lhs", "Latin Hypercube"),
        ],
        monte_carlo_modes=["choice", "uniform", "normal"],
        monte_carlo_lhs_modes=["uniform", "normal"],
        chaser_init_modes=["rocket_deployment", "relative_ric_rect", "relative_ric_curv"],
        satellite_presets=_discover_named_presets("presets.satellites", "SatellitePreset"),
        rocket_preset_stacks=_discover_named_presets("presets.rockets", "RocketStackPreset"),
        figure_ids=list(AVAILABLE_FIGURE_IDS),
        animation_types=list(AVAILABLE_ANIMATION_TYPES),
        base_guidance_options={
            "rocket": _discover_pointer_options(
                [
                    ("Open Loop Pitch Program", "sim.rocket.guidance", "OpenLoopPitchProgramGuidance"),
                    ("Closed Loop Insertion", "sim.rocket.guidance", "ClosedLoopInsertionGuidance"),
                    ("Hold Attitude", "sim.rocket.guidance", "HoldAttitudeGuidance"),
                ]
            ),
        },
        guidance_modifier_options=_discover_pointer_options(
            [
                ("TVC Steering", "sim.rocket.guidance", "TVCSteeringGuidance"),
                ("Max Q Throttle Limiter", "sim.rocket.guidance", "MaxQThrottleLimiterGuidance"),
                ("Orbit Insertion Cutoff", "sim.rocket.guidance", "OrbitInsertionCutoffGuidance"),
            ]
        ),
        orbit_control_options={
            "rocket": _discover_pointer_options(
                [("Zero Controller", "sim.control.orbit.zero_controller", "ZeroController")]
            ),
            "chaser": _discover_pointer_options(
                [
                    ("Zero Controller", "sim.control.orbit.zero_controller", "ZeroController"),
                    ("HCW LQR", "sim.control.orbit.lqr", "HCWLQRController"),
                    ("Relative Orbit MPC", "sim.control.orbit.relative_mpc", "RelativeOrbitMPCController"),
                    ("HCW Relative MPC", "sim.control.orbit.hcw_mpc", "HCWRelativeOrbitMPCController"),
                    ("Stationkeeping", "sim.control.orbit.baseline", "StationkeepingController"),
                ]
            ),
            "target": _discover_pointer_options(
                [
                    ("Zero Controller", "sim.control.orbit.zero_controller", "ZeroController"),
                    ("Stationkeeping", "sim.control.orbit.baseline", "StationkeepingController"),
                ]
            ),
        },
        attitude_control_options={
            "rocket": _discover_pointer_options(
                [("Zero Torque", "sim.control.attitude.zero_torque", "ZeroTorqueController")]
            ),
            "chaser": _discover_pointer_options(
                [
                    ("Zero Torque", "sim.control.attitude.zero_torque", "ZeroTorqueController"),
                    ("Surrogate Snap ECI", "sim.control.attitude.surrogate_snap", "SurrogateSnapECIController"),
                    ("Surrogate Snap RIC", "sim.control.attitude.surrogate_snap", "SurrogateSnapRICController"),
                    ("RIC Detumble PD", "sim.control.attitude.detumble_pd", "RICDetumblePDController"),
                    ("Quaternion PD", "sim.control.attitude.baseline", "QuaternionPDController"),
                ]
            ),
            "target": _discover_pointer_options(
                [
                    ("Zero Torque", "sim.control.attitude.zero_torque", "ZeroTorqueController"),
                    ("Surrogate Snap ECI", "sim.control.attitude.surrogate_snap", "SurrogateSnapECIController"),
                    ("Quaternion PD", "sim.control.attitude.baseline", "QuaternionPDController"),
                ]
            ),
        },
        mission_strategy_options={
            "rocket": _discover_pointer_options(
                [
                    ("Pursuit", "sim.mission.modules", "RocketPursuitMissionStrategy"),
                    ("Predefined Orbit", "sim.mission.modules", "RocketPredefinedOrbitMissionStrategy"),
                ],
                include_none=True,
            ),
            "chaser": _discover_pointer_options(
                [
                    ("Pursuit", "sim.mission.modules", "PursuitMissionStrategy"),
                    ("Evade", "sim.mission.modules", "EvadeMissionStrategy"),
                    ("Hold", "sim.mission.modules", "HoldMissionStrategy"),
                    ("Desired State", "sim.mission.modules", "DesiredStateMissionStrategy"),
                    ("Mission Executive", "sim.mission.modules", "MissionExecutiveStrategy"),
                    ("Station Keep", "sim.mission.modules", "StationKeepMissionStrategy"),
                    ("Inspect", "sim.mission.modules", "InspectMissionStrategy"),
                    ("Defensive", "sim.mission.modules", "DefensiveMissionStrategy"),
                    ("Safe Hold", "sim.mission.modules", "SafeHoldMissionStrategy"),
                ],
                include_none=True,
            ),
            "target": _discover_pointer_options(
                [
                    ("Pursuit", "sim.mission.modules", "PursuitMissionStrategy"),
                    ("Evade", "sim.mission.modules", "EvadeMissionStrategy"),
                    ("Hold", "sim.mission.modules", "HoldMissionStrategy"),
                    ("Desired State", "sim.mission.modules", "DesiredStateMissionStrategy"),
                    ("Mission Executive", "sim.mission.modules", "MissionExecutiveStrategy"),
                    ("Station Keep", "sim.mission.modules", "StationKeepMissionStrategy"),
                    ("Inspect", "sim.mission.modules", "InspectMissionStrategy"),
                    ("Defensive", "sim.mission.modules", "DefensiveMissionStrategy"),
                    ("Safe Hold", "sim.mission.modules", "SafeHoldMissionStrategy"),
                ],
                include_none=True,
            ),
        },
        mission_execution_options={
            "rocket": _discover_pointer_options(
                [
                    ("Go Now", "sim.mission.modules", "RocketGoNowExecution"),
                    ("Go When Possible", "sim.mission.modules", "RocketGoWhenPossibleExecution"),
                    ("Wait For Optimal", "sim.mission.modules", "RocketWaitOptimalExecution"),
                ],
                include_none=True,
            ),
            "chaser": _discover_pointer_options(
                [
                    ("Controller Pointing", "sim.mission.modules", "ControllerPointingExecution"),
                    ("Predictive Burn", "sim.mission.modules", "PredictiveBurnExecution"),
                    ("Integrated Command", "sim.mission.modules", "IntegratedCommandExecution"),
                    ("Budgeted End State", "sim.mission.modules", "BudgetedEndStateExecution"),
                    ("Direct Integrated", "sim.mission.modules", "DirectIntegratedExecution"),
                    ("Impulsive", "sim.mission.modules", "ImpulsiveExecution"),
                    ("Safe Hold", "sim.mission.modules", "SafeHoldExecution"),
                ],
                include_none=True,
            ),
            "target": _discover_pointer_options(
                [
                    ("Controller Pointing", "sim.mission.modules", "ControllerPointingExecution"),
                    ("Predictive Burn", "sim.mission.modules", "PredictiveBurnExecution"),
                    ("Integrated Command", "sim.mission.modules", "IntegratedCommandExecution"),
                    ("Budgeted End State", "sim.mission.modules", "BudgetedEndStateExecution"),
                    ("Direct Integrated", "sim.mission.modules", "DirectIntegratedExecution"),
                    ("Impulsive", "sim.mission.modules", "ImpulsiveExecution"),
                    ("Safe Hold", "sim.mission.modules", "SafeHoldExecution"),
                ],
                include_none=True,
            ),
        },
        monte_carlo_parameter_categories=MONTE_CARLO_PARAMETER_CATEGORIES,
        parameter_form_schemas=PARAMETER_FORM_SCHEMAS,
        analysis_ui_profiles=ANALYSIS_UI_PROFILES,
    )


def get_repo_root() -> Path:
    return REPO_ROOT


def get_default_config_path() -> Path:
    return DEFAULT_CONFIG_PATH


def get_config_dir() -> Path:
    return CONFIG_DIR


def list_available_configs() -> list[Path]:
    return list_config_files()


def load_config(path: str | Path) -> dict[str, Any]:
    return ensure_sections(load_config_dict(path))


def load_config_text(path: str | Path) -> str:
    return read_yaml_file(path)


def parse_config_text(yaml_text: str) -> dict[str, Any]:
    return ensure_sections(parse_yaml_text(yaml_text))


def validate_config(data: dict[str, Any]) -> SimulationScenarioConfig:
    return validate_config_dict(ensure_sections(data))


def dump_config_text(data: dict[str, Any]) -> str:
    return dump_yaml_text(ensure_sections(data))


def save_config(path: str | Path, data: dict[str, Any]) -> Path:
    cfg = validate_config(data)
    return save_config_dict(path, cfg.to_dict())


def summarize_config(cfg: SimulationScenarioConfig) -> ConfigSummary:
    objects = [
        object_id
        for object_id, section in (("rocket", cfg.rocket), ("chaser", cfg.chaser), ("target", cfg.target))
        if bool(section.enabled)
    ]
    analysis_enabled = bool(cfg.analysis.enabled)
    analysis_study_type = str(cfg.analysis.study_type if analysis_enabled else "single_run")
    return ConfigSummary(
        scenario_name=cfg.scenario_name,
        scenario_type=cfg.simulator.scenario_type,
        duration_s=float(cfg.simulator.duration_s),
        dt_s=float(cfg.simulator.dt_s),
        objects=objects,
        output_dir=cfg.outputs.output_dir,
        output_mode=cfg.outputs.mode,
        analysis_enabled=analysis_enabled,
        analysis_study_type=analysis_study_type,
        monte_carlo_enabled=bool(cfg.monte_carlo.enabled),
        mc_iterations=int(cfg.monte_carlo.iterations),
    )


def build_cli_run_command(config_path: str | Path) -> list[str]:
    return build_run_command(config_path)


def run_config_via_cli(config_path: str | Path) -> dict[str, Any]:
    return run_simulation_cli(config_path)


def run_config_via_api(
    config_path: str | Path,
    *,
    step_callback: Any | None = None,
) -> RunResult:
    path = Path(config_path).expanduser().resolve()
    started = time.perf_counter()
    session = SimulationSession.from_yaml(path)
    result = session.run(step_callback=step_callback)
    elapsed_s = time.perf_counter() - started
    summary = result.summary
    output_dir = str(result.config.scenario.outputs.output_dir)
    lines = [
        f"Scenario: {result.config.scenario_name}",
        f"Mode: {result.analysis_study_type}",
        f"Elapsed: {elapsed_s:.2f}s",
    ]
    if summary:
        if "samples" in summary:
            lines.append(f"Samples: {summary.get('samples')}")
        if "duration_s" in summary:
            lines.append(f"Duration: {summary.get('duration_s')} s")
        if "terminated_early" in summary:
            lines.append(f"Terminated Early: {summary.get('terminated_early')}")
    return RunResult(
        command=["api", str(path)],
        returncode=0,
        stdout="\n".join(lines) + "\n",
        stderr="",
        elapsed_s=float(elapsed_s),
        output_dir=output_dir,
        scenario_name=result.config.scenario_name,
    )


def get_output_files(output_dir: str | Path, limit: int = 200) -> list[Path]:
    return list_output_files(output_dir, limit=limit)
