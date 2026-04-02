from __future__ import annotations

import copy
from typing import Any


class GuiConfigAdapter:
    def load_into_window(self, window: Any, cfg: dict[str, Any]) -> None:
        window._suppress_dirty_tracking = True
        sim = dict(cfg.get("simulator", {}) or {})
        outputs = dict(cfg.get("outputs", {}) or {})
        mc = dict(cfg.get("monte_carlo", {}) or {})
        analysis = dict(cfg.get("analysis", {}) or {})
        target = dict(cfg.get("target", {}) or {})
        chaser = dict(cfg.get("chaser", {}) or {})
        rocket = dict(cfg.get("rocket", {}) or {})

        window.scenario_name_edit.setText(str(cfg.get("scenario_name", "")))
        window.scenario_description_edit.setText(str(cfg.get("scenario_description", "") or ""))
        window.duration_spin.setValue(float(sim.get("duration_s", 3600.0) or 3600.0))
        window.dt_spin.setValue(float(sim.get("dt_s", 1.0) or 1.0))
        window.output_dir_edit.setText(str(outputs.get("output_dir", "outputs/gui_run") or "outputs/gui_run"))
        window._set_combo_text_or_append(window.output_mode_combo, str(outputs.get("mode", "interactive") or "interactive"))

        analysis_enabled = bool(analysis.get("enabled", False) or mc.get("enabled", False))
        study_type = str(analysis.get("study_type", "monte_carlo") or "monte_carlo").strip().lower()
        sensitivity = dict(analysis.get("sensitivity", {}) or {})
        sensitivity_method = str(sensitivity.get("method", "one_at_a_time") or "one_at_a_time").strip().lower()
        window.mc_enabled_check.setChecked(analysis_enabled)
        window._set_combo_data_or_text(window.analysis_study_type_combo, "sensitivity" if study_type == "sensitivity" else "monte_carlo")
        window._set_combo_data_or_text(window.sensitivity_method_combo, "lhs" if sensitivity_method == "lhs" else "one_at_a_time")
        window.analysis_metrics_edit.setPlainText(window._format_analysis_metrics_text(list(analysis.get("metrics", []) or [])))
        baseline = dict(analysis.get("baseline", {}) or {})
        execution = dict(analysis.get("execution", {}) or {})
        window.analysis_baseline_enable_check.setChecked(bool(baseline.get("enabled", False)))
        window.analysis_baseline_path_edit.setText(str(baseline.get("summary_json", "") or ""))
        window.mc_parallel_check.setChecked(bool(execution.get("parallel_enabled", mc.get("parallel_enabled", False))))
        window.mc_workers_spin.setValue(int(execution.get("parallel_workers", mc.get("parallel_workers", 0)) or 0))
        if study_type == "sensitivity":
            window.mc_iterations_spin.setValue(max(int(sensitivity.get("samples", 1) or 1), 1))
            window.mc_base_seed_spin.setValue(int(sensitivity.get("seed", 0) or 0))
            window.analysis_lhs_samples_spin.setValue(max(int(sensitivity.get("samples", 1) or 1), 1))
            window.mc_variations = []
            for param in list(sensitivity.get("parameters", []) or []):
                parameter_path = str(dict(param or {}).get("parameter_path", "") or "")
                if not parameter_path:
                    continue
                distribution = str(dict(param or {}).get("distribution", "uniform") or "uniform").strip().lower()
                variation: dict[str, Any] = {"parameter_path": parameter_path, "mode": distribution}
                if distribution == "normal":
                    variation["mean"] = float(dict(param or {}).get("mean", 0.0) or 0.0)
                    variation["std"] = float(dict(param or {}).get("std", 0.0) or 0.0)
                elif distribution == "uniform":
                    variation["low"] = float(dict(param or {}).get("low", 0.0) or 0.0)
                    variation["high"] = float(dict(param or {}).get("high", 0.0) or 0.0)
                else:
                    values = list(dict(param or {}).get("values", []) or [])
                    if values:
                        variation["mode"] = "choice"
                        variation["options"] = values
                    else:
                        variation["mode"] = "uniform"
                        variation["low"] = 0.0
                        variation["high"] = 0.0
                window.mc_variations.append(variation)
        else:
            window.mc_iterations_spin.setValue(int(mc.get("iterations", 1)))
            window.mc_parallel_check.setChecked(bool(mc.get("parallel_enabled", False)))
            window.mc_workers_spin.setValue(int(mc.get("parallel_workers", 0)))
            window.mc_base_seed_spin.setValue(int(mc.get("base_seed", 0)))
            window.analysis_lhs_samples_spin.setValue(max(int(sensitivity.get("samples", 0) or 0), 1))
            window.mc_variations = [dict(v or {}) for v in list(mc.get("variations", []) or [])]
        window._rebuild_mc_category_combo()
        window._refresh_mc_parameter_options()
        window._refresh_mc_variations_list()
        window._clear_mc_variation_editor()
        window._refresh_analysis_editor_ui()

        dynamics = dict(sim.get("dynamics", {}) or {})
        orbit_dyn = dict(dynamics.get("orbit", {}) or {})
        att_dyn = dict(dynamics.get("attitude", {}) or {})
        disturbance_torques = dict(att_dyn.get("disturbance_torques", {}) or {})
        orbit_substep_val = orbit_dyn.get("orbit_substep_s")
        attitude_substep_val = att_dyn.get("attitude_substep_s")
        window._set_combo_text_or_append(window.orbit_integrator_combo, str(orbit_dyn.get("integrator", "rk4") or "rk4"))
        window.orbit_adaptive_atol_spin.setValue(float(orbit_dyn.get("adaptive_atol", 1.0e-9) or 0.0))
        window.orbit_adaptive_rtol_spin.setValue(float(orbit_dyn.get("adaptive_rtol", 1.0e-7) or 0.0))
        window.orbit_substep_enabled_check.setChecked(orbit_substep_val is not None)
        window.attitude_substep_enabled_check.setChecked(attitude_substep_val is not None)
        window.orbit_substep_spin.setValue(float(orbit_substep_val or 0.0))
        window.attitude_substep_spin.setValue(float(attitude_substep_val or 0.0))
        window.attitude_enabled_check.setChecked(bool(att_dyn.get("enabled", True)))
        window.orbit_j2_check.setChecked(bool(orbit_dyn.get("j2", False)))
        window.orbit_j3_check.setChecked(bool(orbit_dyn.get("j3", False)))
        window.orbit_j4_check.setChecked(bool(orbit_dyn.get("j4", False)))
        window.orbit_drag_check.setChecked(bool(orbit_dyn.get("drag", False)))
        window.orbit_srp_check.setChecked(bool(orbit_dyn.get("srp", False)))
        window.orbit_moon_check.setChecked(bool(orbit_dyn.get("third_body_moon", False)))
        window.orbit_sun_check.setChecked(bool(orbit_dyn.get("third_body_sun", False)))
        window.att_gg_check.setChecked(bool(disturbance_torques.get("gravity_gradient", False)))
        window.att_magnetic_check.setChecked(bool(disturbance_torques.get("magnetic", False)))
        window.att_drag_check.setChecked(bool(disturbance_torques.get("drag", False)))
        window.att_srp_check.setChecked(bool(disturbance_torques.get("srp", False)))
        window._refresh_substep_visibility()
        window._refresh_integrator_visibility()

        target_specs = dict(target.get("specs", {}) or {})
        target_coes = dict(target.get("initial_state", {}).get("coes", {}) or {})
        window.target_enabled.setChecked(bool(target.get("enabled", True)))
        target_default = window.target_preset.itemText(0) if window.target_preset.count() else ""
        window._set_combo_text_or_append(window.target_preset, str(target_specs.get("preset_satellite", "") or target_default))
        target_mass_fallback = float(target_specs.get("mass_kg", 400.0) or 0.0)
        window.target_dry_mass.setValue(float(target_specs.get("dry_mass_kg", target_mass_fallback) or 0.0))
        window.target_fuel_mass.setValue(float(target_specs.get("fuel_mass_kg", 0.0) or 0.0))
        window.target_a.setValue(float(target_coes.get("a_km", 7000.0) or 7000.0))
        window.target_ecc.setValue(float(target_coes.get("ecc", 0.001) or 0.0))
        window.target_inc.setValue(float(target_coes.get("inc_deg", 45.0) or 0.0))
        window.target_raan.setValue(float(target_coes.get("raan_deg", 0.0) or 0.0))
        window.target_argp.setValue(float(target_coes.get("argp_deg", 0.0) or 0.0))
        window.target_ta.setValue(float(target_coes.get("true_anomaly_deg", 0.0) or 0.0))
        window._set_pointer_combo_value(window.target_strategy_combo, dict(target.get("mission_strategy", {}) or {}) if target.get("mission_strategy") else None)
        window._set_pointer_combo_value(window.target_execution_combo, dict(target.get("mission_execution", {}) or {}) if target.get("mission_execution") else None)
        window._set_pointer_combo_value(window.target_orbit_control_combo, dict(target.get("orbit_control", {}) or {}) if target.get("orbit_control") else None)
        window._set_pointer_combo_value(window.target_attitude_control_combo, dict(target.get("attitude_control", {}) or {}) if target.get("attitude_control") else None)
        self.load_knowledge_into_window(window, "target", dict(target.get("knowledge", {}) or {}))

        chaser_specs = dict(chaser.get("specs", {}) or {})
        chaser_init = dict(chaser.get("initial_state", {}) or {})
        window.chaser_enabled.setChecked(bool(chaser.get("enabled", False)))
        chaser_default = window.chaser_preset.itemText(0) if window.chaser_preset.count() else ""
        window._set_combo_text_or_append(window.chaser_preset, str(chaser_specs.get("preset_satellite", "") or chaser_default))
        chaser_mass_fallback = float(chaser_specs.get("mass_kg", 200.0) or 0.0)
        window.chaser_dry_mass.setValue(float(chaser_specs.get("dry_mass_kg", chaser_mass_fallback) or 0.0))
        window.chaser_fuel_mass.setValue(float(chaser_specs.get("fuel_mass_kg", 0.0) or 0.0))
        rel_block = dict(chaser_init.get("relative_to_target_ric", {}) or {})
        if rel_block:
            frame = str(rel_block.get("frame", "rect") or "rect").strip().lower()
            window._set_combo_data_or_text(window.chaser_init_mode, "relative_ric_curv" if frame == "curv" else "relative_ric_rect")
            values = list(rel_block.get("state", [0.0] * 6))
        elif "relative_ric_rect" in chaser_init:
            window._set_combo_data_or_text(window.chaser_init_mode, "relative_ric_rect")
            values = list(chaser_init.get("relative_ric_rect", [0.0] * 6))
        elif "relative_ric_curv" in chaser_init:
            window._set_combo_data_or_text(window.chaser_init_mode, "relative_ric_curv")
            values = list(chaser_init.get("relative_ric_curv", [0.0] * 6))
        else:
            window._set_combo_data_or_text(window.chaser_init_mode, "rocket_deployment")
            values = list(chaser_init.get("deploy_dv_body_m_s", [10.0, 0.0, 0.0])) + [0.0, 0.0, 0.0]
        window.chaser_deploy_time.setValue(float(chaser_init.get("deploy_time_s", 900.0) or 0.0))
        for i, widget in enumerate(window.chaser_init_values):
            widget.setValue(float(values[i] if i < len(values) else 0.0))
        window._set_pointer_combo_value(window.chaser_strategy_combo, dict(chaser.get("mission_strategy", {}) or {}) if chaser.get("mission_strategy") else None)
        window._set_pointer_combo_value(window.chaser_execution_combo, dict(chaser.get("mission_execution", {}) or {}) if chaser.get("mission_execution") else None)
        window._set_pointer_combo_value(window.chaser_orbit_control_combo, dict(chaser.get("orbit_control", {}) or {}) if chaser.get("orbit_control") else None)
        window._set_pointer_combo_value(window.chaser_attitude_control_combo, dict(chaser.get("attitude_control", {}) or {}) if chaser.get("attitude_control") else None)
        self.load_knowledge_into_window(window, "chaser", dict(chaser.get("knowledge", {}) or {}))

        rocket_specs = dict(rocket.get("specs", {}) or {})
        rocket_init = dict(rocket.get("initial_state", {}) or {})
        window.rocket_enabled.setChecked(bool(rocket.get("enabled", False)))
        rocket_default = window.rocket_preset.itemText(0) if window.rocket_preset.count() else ""
        window._set_combo_text_or_append(window.rocket_preset, str(rocket_specs.get("preset_stack", "") or rocket_default))
        window.rocket_payload.setValue(float(rocket_specs.get("payload_mass_kg", 150.0) or 0.0))
        window.rocket_launch_lat.setValue(float(rocket_init.get("launch_lat_deg", 28.5) or 0.0))
        window.rocket_launch_lon.setValue(float(rocket_init.get("launch_lon_deg", -80.6) or 0.0))
        window.rocket_launch_alt.setValue(float(rocket_init.get("launch_alt_km", 0.0) or 0.0))
        window.rocket_launch_az.setValue(float(rocket_init.get("launch_azimuth_deg", 90.0) or 0.0))
        window._set_pointer_combo_value(window.rocket_strategy_combo, dict(rocket.get("mission_strategy", {}) or {}) if rocket.get("mission_strategy") else None)
        window._set_pointer_combo_value(window.rocket_execution_combo, dict(rocket.get("mission_execution", {}) or {}) if rocket.get("mission_execution") else None)
        rocket_base_guidance = dict(rocket.get("base_guidance", {}) or {}) if rocket.get("base_guidance") else None
        if rocket_base_guidance is None and rocket.get("guidance"):
            rocket_base_guidance = dict(rocket.get("guidance", {}) or {})
        window._set_pointer_combo_value(window.rocket_base_guidance_combo, rocket_base_guidance)
        window.rocket_guidance_modifiers_config = copy.deepcopy(list(rocket.get("guidance_modifiers", []) or []))
        window._refresh_rocket_guidance_modifiers_label()
        self.load_knowledge_into_window(window, "rocket", dict(rocket.get("knowledge", {}) or {}))

        stats = dict(outputs.get("stats", {}) or {})
        plots = dict(outputs.get("plots", {}) or {})
        animations = dict(outputs.get("animations", {}) or {})
        mc_outputs = dict(outputs.get("monte_carlo", {}) or {})
        window.stats_enabled.setChecked(bool(stats.get("enabled", True)))
        window.stats_print_summary.setChecked(bool(stats.get("print_summary", True)))
        window.stats_save_json.setChecked(bool(stats.get("save_json", True)))
        window.stats_save_csv.setChecked(bool(stats.get("save_csv", False)))
        window.plots_enabled.setChecked(bool(plots.get("enabled", True)))
        window.plots_dpi.setValue(int(plots.get("dpi", 150) or 150))
        for check in window.figure_id_checks.values():
            check.setChecked(False)
        for figure_id in list(plots.get("figure_ids", []) or []):
            if figure_id in window.figure_id_checks:
                window.figure_id_checks[figure_id].setChecked(True)
        window.reference_object_edit.setText(str(plots.get("reference_object_id", "") or ""))
        for check in window.animation_type_checks.values():
            check.setChecked(False)
        for anim_type in list(animations.get("types", []) or []):
            if anim_type in window.animation_type_checks:
                window.animation_type_checks[anim_type].setChecked(True)
        window.animation_fps_spin.setValue(float(animations.get("fps", 30.0) or 30.0))
        window.animation_speed_multiple_spin.setValue(float(animations.get("speed_multiple", 10.0) or 10.0))
        window.animation_frame_stride_spin.setValue(int(animations.get("frame_stride", 1) or 1))
        window.mc_save_iteration_summaries.setChecked(bool(mc_outputs.get("save_iteration_summaries", False)))
        window.mc_save_aggregate_summary.setChecked(bool(mc_outputs.get("save_aggregate_summary", True)))
        window.mc_save_histograms.setChecked(bool(mc_outputs.get("save_histograms", False)))
        window.mc_display_histograms.setChecked(bool(mc_outputs.get("display_histograms", False)))
        window.mc_save_ops_dashboard.setChecked(bool(mc_outputs.get("save_ops_dashboard", True)))
        window.mc_display_ops_dashboard.setChecked(bool(mc_outputs.get("display_ops_dashboard", False)))
        window.mc_save_raw_runs.setChecked(bool(mc_outputs.get("save_raw_runs", False)))
        window.mc_require_rocket_insertion.setChecked(bool(mc_outputs.get("require_rocket_insertion", False)))
        window.mc_baseline_summary_json.setText(str(mc_outputs.get("baseline_summary_json", "") or ""))
        gates = dict(mc_outputs.get("gates", {}) or {})
        window.mc_gate_min_closest_approach.setValue(float(gates.get("min_closest_approach_km", 0.0) or 0.0))
        window.mc_gate_max_duration.setValue(float(gates.get("max_duration_s", 0.0) or 0.0))
        window.mc_gate_max_total_dv.setValue(float(gates.get("max_total_dv_m_s", 0.0) or 0.0))
        window.mc_gate_max_guardrail_events.setValue(float(gates.get("max_guardrail_events", 0.0) or 0.0))
        window._refresh_outputs_mode_ui()
        window._suppress_dirty_tracking = False

    def collect_from_window(self, window: Any, current_config: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(current_config)
        cfg["scenario_name"] = window.scenario_name_edit.text().strip()
        cfg["scenario_description"] = window.scenario_description_edit.text().strip()
        sim = cfg.setdefault("simulator", {})
        outputs = cfg.setdefault("outputs", {})
        mc = cfg.setdefault("monte_carlo", {})
        analysis = cfg.setdefault("analysis", {})
        target = cfg.setdefault("target", {})
        chaser = cfg.setdefault("chaser", {})
        rocket = cfg.setdefault("rocket", {})

        sim["duration_s"] = float(window.duration_spin.value())
        sim["dt_s"] = float(window.dt_spin.value())
        dynamics = sim.setdefault("dynamics", {})
        orbit_dyn = dynamics.setdefault("orbit", {})
        att_dyn = dynamics.setdefault("attitude", {})
        disturbance_torques = att_dyn.setdefault("disturbance_torques", {})

        orbit_substep = float(window.orbit_substep_spin.value())
        attitude_substep = float(window.attitude_substep_spin.value())
        orbit_dyn["integrator"] = window.orbit_integrator_combo.currentText()
        orbit_dyn["adaptive_atol"] = float(window.orbit_adaptive_atol_spin.value())
        orbit_dyn["adaptive_rtol"] = float(window.orbit_adaptive_rtol_spin.value())
        orbit_dyn["orbit_substep_s"] = orbit_substep if (window.orbit_substep_enabled_check.isChecked() and orbit_substep > 0.0) else None
        att_dyn["attitude_substep_s"] = attitude_substep if (window.attitude_substep_enabled_check.isChecked() and attitude_substep > 0.0) else None
        att_dyn["enabled"] = bool(window.attitude_enabled_check.isChecked())
        orbit_dyn["j2"] = bool(window.orbit_j2_check.isChecked())
        orbit_dyn["j3"] = bool(window.orbit_j3_check.isChecked())
        orbit_dyn["j4"] = bool(window.orbit_j4_check.isChecked())
        orbit_dyn["drag"] = bool(window.orbit_drag_check.isChecked())
        orbit_dyn["srp"] = bool(window.orbit_srp_check.isChecked())
        orbit_dyn["third_body_moon"] = bool(window.orbit_moon_check.isChecked())
        orbit_dyn["third_body_sun"] = bool(window.orbit_sun_check.isChecked())
        disturbance_torques["gravity_gradient"] = bool(window.att_gg_check.isChecked())
        disturbance_torques["magnetic"] = bool(window.att_magnetic_check.isChecked())
        disturbance_torques["drag"] = bool(window.att_drag_check.isChecked())
        disturbance_torques["srp"] = bool(window.att_srp_check.isChecked())

        outputs["mode"] = window.output_mode_combo.currentText()
        output_dir = window.output_dir_edit.text().strip() or "outputs/gui_run"
        outputs["output_dir"] = output_dir
        if window.output_dir_edit.text().strip() != output_dir:
            window.output_dir_edit.setText(output_dir)
        analysis_enabled = bool(window.mc_enabled_check.isChecked())
        study_type = window._selected_analysis_study_type()
        mc["enabled"] = bool(analysis_enabled and study_type == "monte_carlo")
        mc["iterations"] = int(window.mc_iterations_spin.value())
        mc["parallel_enabled"] = bool(window.mc_parallel_check.isChecked())
        mc["parallel_workers"] = int(window.mc_workers_spin.value())
        mc["base_seed"] = int(window.mc_base_seed_spin.value())
        mc["variations"] = copy.deepcopy(window.mc_variations)
        analysis["enabled"] = analysis_enabled
        analysis["study_type"] = study_type
        analysis["execution"] = {
            "parallel_enabled": bool(window.mc_parallel_check.isChecked()),
            "parallel_workers": int(window.mc_workers_spin.value()),
        }
        analysis["metrics"] = window._parse_analysis_metrics_text()
        analysis["baseline"] = {
            "enabled": bool(window.analysis_baseline_enable_check.isChecked()),
            "summary_json": window.analysis_baseline_path_edit.text().strip(),
        }
        analysis["monte_carlo"] = {
            "iterations": int(window.mc_iterations_spin.value()),
            "base_seed": int(window.mc_base_seed_spin.value()),
            "variations": copy.deepcopy(window.mc_variations),
        }
        sensitivity_method = window._selected_sensitivity_method()
        params = []
        for variation in window.mc_variations:
            parameter_path = str(dict(variation or {}).get("parameter_path", "") or "")
            if not parameter_path:
                continue
            mode = str(dict(variation or {}).get("mode", "choice") or "choice").strip().lower()
            if sensitivity_method == "lhs":
                param_entry = {
                    "parameter_path": parameter_path,
                    "distribution": mode if mode in {"uniform", "normal"} else "uniform",
                }
                if mode == "normal":
                    param_entry["mean"] = float(dict(variation or {}).get("mean", 0.0) or 0.0)
                    param_entry["std"] = float(dict(variation or {}).get("std", 0.0) or 0.0)
                else:
                    param_entry["low"] = float(dict(variation or {}).get("low", 0.0) or 0.0)
                    param_entry["high"] = float(dict(variation or {}).get("high", 0.0) or 0.0)
            else:
                values = list(dict(variation or {}).get("options", []) or [])
                if mode == "uniform":
                    values = [dict(variation or {}).get("low"), dict(variation or {}).get("high")]
                elif mode == "normal":
                    mean = dict(variation or {}).get("mean")
                    std = dict(variation or {}).get("std")
                    if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
                        values = [float(mean) - float(std), float(mean), float(mean) + float(std)]
                param_entry = {
                    "parameter_path": parameter_path,
                    "values": [v for v in values if v is not None],
                    "distribution": mode if mode in {"uniform", "normal"} else "uniform",
                }
                if mode == "uniform":
                    param_entry["low"] = float(dict(variation or {}).get("low", 0.0) or 0.0)
                    param_entry["high"] = float(dict(variation or {}).get("high", 0.0) or 0.0)
                elif mode == "normal":
                    param_entry["mean"] = float(dict(variation or {}).get("mean", 0.0) or 0.0)
                    param_entry["std"] = float(dict(variation or {}).get("std", 0.0) or 0.0)
            params.append(param_entry)
        analysis["sensitivity"] = {
            "method": sensitivity_method,
            "samples": int(window.mc_iterations_spin.value()),
            "seed": int(window.mc_base_seed_spin.value()),
            "parameters": params,
        }

        target["enabled"] = bool(window.target_enabled.isChecked())
        target.setdefault("specs", {})["preset_satellite"] = window.target_preset.currentText().strip()
        target["specs"]["dry_mass_kg"] = float(window.target_dry_mass.value())
        target["specs"]["fuel_mass_kg"] = float(window.target_fuel_mass.value())
        target["specs"].pop("mass_kg", None)
        target.setdefault("initial_state", {})["coes"] = {
            "a_km": float(window.target_a.value()),
            "ecc": float(window.target_ecc.value()),
            "inc_deg": float(window.target_inc.value()),
            "raan_deg": float(window.target_raan.value()),
            "argp_deg": float(window.target_argp.value()),
            "true_anomaly_deg": float(window.target_ta.value()),
        }
        target["mission_strategy"] = window._combo_pointer_value(window.target_strategy_combo, existing=dict(target.get("mission_strategy", {}) or {}) if target.get("mission_strategy") else None)
        target["mission_execution"] = window._combo_pointer_value(window.target_execution_combo, existing=dict(target.get("mission_execution", {}) or {}) if target.get("mission_execution") else None)
        target.pop("guidance", None)
        target["orbit_control"] = window._combo_pointer_value(window.target_orbit_control_combo, existing=dict(target.get("orbit_control", {}) or {}) if target.get("orbit_control") else None)
        target["attitude_control"] = window._combo_pointer_value(window.target_attitude_control_combo, existing=dict(target.get("attitude_control", {}) or {}) if target.get("attitude_control") else None)
        target["knowledge"] = self.collect_knowledge_from_window(window, "target", existing=dict(target.get("knowledge", {}) or {}))

        chaser["enabled"] = bool(window.chaser_enabled.isChecked())
        chaser.setdefault("specs", {})["preset_satellite"] = window.chaser_preset.currentText().strip()
        chaser["specs"]["dry_mass_kg"] = float(window.chaser_dry_mass.value())
        chaser["specs"]["fuel_mass_kg"] = float(window.chaser_fuel_mass.value())
        chaser["specs"].pop("mass_kg", None)
        init_mode = str(window.chaser_init_mode.currentData() or window.chaser_init_mode.currentText())
        chaser_initial_state = dict(chaser.get("initial_state", {}) or {})
        if init_mode == "rocket_deployment":
            chaser_initial_state["source"] = "rocket_deployment"
            chaser_initial_state["deploy_time_s"] = float(window.chaser_deploy_time.value())
            chaser_initial_state["deploy_dv_body_m_s"] = [float(window.chaser_init_values[i].value()) for i in range(3)]
            chaser_initial_state.pop("relative_to_target_ric", None)
            chaser_initial_state.pop("relative_ric_rect", None)
            chaser_initial_state.pop("relative_ric_curv", None)
        else:
            chaser_initial_state.pop("source", None)
            chaser_initial_state.pop("deploy_time_s", None)
            chaser_initial_state.pop("deploy_dv_body_m_s", None)
            chaser_initial_state["relative_to_target_ric"] = {
                "frame": "rect" if init_mode == "relative_ric_rect" else "curv",
                "state": [float(widget.value()) for widget in window.chaser_init_values],
            }
            chaser_initial_state.pop("relative_ric_rect", None)
            chaser_initial_state.pop("relative_ric_curv", None)
        chaser["initial_state"] = chaser_initial_state
        chaser["mission_strategy"] = window._combo_pointer_value(window.chaser_strategy_combo, existing=dict(chaser.get("mission_strategy", {}) or {}) if chaser.get("mission_strategy") else None)
        chaser["mission_execution"] = window._combo_pointer_value(window.chaser_execution_combo, existing=dict(chaser.get("mission_execution", {}) or {}) if chaser.get("mission_execution") else None)
        chaser.pop("guidance", None)
        chaser["orbit_control"] = window._combo_pointer_value(window.chaser_orbit_control_combo, existing=dict(chaser.get("orbit_control", {}) or {}) if chaser.get("orbit_control") else None)
        chaser["attitude_control"] = window._combo_pointer_value(window.chaser_attitude_control_combo, existing=dict(chaser.get("attitude_control", {}) or {}) if chaser.get("attitude_control") else None)
        chaser["knowledge"] = self.collect_knowledge_from_window(window, "chaser", existing=dict(chaser.get("knowledge", {}) or {}))

        rocket["enabled"] = bool(window.rocket_enabled.isChecked())
        rocket.setdefault("specs", {})["preset_stack"] = window.rocket_preset.currentText().strip()
        rocket["specs"]["payload_mass_kg"] = float(window.rocket_payload.value())
        rocket["initial_state"] = {
            "launch_lat_deg": float(window.rocket_launch_lat.value()),
            "launch_lon_deg": float(window.rocket_launch_lon.value()),
            "launch_alt_km": float(window.rocket_launch_alt.value()),
            "launch_azimuth_deg": float(window.rocket_launch_az.value()),
        }
        rocket["mission_strategy"] = window._combo_pointer_value(window.rocket_strategy_combo, existing=dict(rocket.get("mission_strategy", {}) or {}) if rocket.get("mission_strategy") else None)
        rocket["mission_execution"] = window._combo_pointer_value(window.rocket_execution_combo, existing=dict(rocket.get("mission_execution", {}) or {}) if rocket.get("mission_execution") else None)
        rocket["base_guidance"] = window._combo_pointer_value(
            window.rocket_base_guidance_combo,
            existing=dict(rocket.get("base_guidance", {}) or {}) if rocket.get("base_guidance") else (
                dict(rocket.get("guidance", {}) or {}) if rocket.get("guidance") else None
            ),
        )
        rocket["guidance_modifiers"] = copy.deepcopy(window.rocket_guidance_modifiers_config)
        rocket.pop("guidance", None)
        rocket.pop("orbit_control", None)
        rocket.pop("attitude_control", None)
        rocket["knowledge"] = self.collect_knowledge_from_window(window, "rocket", existing=dict(rocket.get("knowledge", {}) or {}))

        stats = outputs.setdefault("stats", {})
        plots = outputs.setdefault("plots", {})
        animations = outputs.setdefault("animations", {})
        mc_outputs = outputs.setdefault("monte_carlo", {})
        stats["enabled"] = bool(window.stats_enabled.isChecked())
        stats["print_summary"] = bool(window.stats_print_summary.isChecked())
        stats["save_json"] = bool(window.stats_save_json.isChecked())
        stats["save_csv"] = bool(window.stats_save_csv.isChecked())
        plots["enabled"] = bool(window.plots_enabled.isChecked())
        plots["dpi"] = int(window.plots_dpi.value())
        figure_ids = [figure_id for figure_id, check in window.figure_id_checks.items() if check.isChecked()]
        plots["figure_ids"] = figure_ids
        ref_obj = window.reference_object_edit.text().strip()
        if ref_obj:
            plots["reference_object_id"] = ref_obj
        else:
            plots.pop("reference_object_id", None)
        animation_types = [anim_type for anim_type, check in window.animation_type_checks.items() if check.isChecked()]
        animations["enabled"] = bool(animation_types)
        animations["types"] = animation_types
        animations["fps"] = float(window.animation_fps_spin.value())
        animations["speed_multiple"] = float(window.animation_speed_multiple_spin.value())
        animations["frame_stride"] = int(window.animation_frame_stride_spin.value())
        mc_outputs["save_iteration_summaries"] = bool(window.mc_save_iteration_summaries.isChecked())
        mc_outputs["save_aggregate_summary"] = bool(window.mc_save_aggregate_summary.isChecked())
        mc_outputs["save_histograms"] = bool(window.mc_save_histograms.isChecked())
        mc_outputs["display_histograms"] = bool(window.mc_display_histograms.isChecked())
        mc_outputs["save_ops_dashboard"] = bool(window.mc_save_ops_dashboard.isChecked())
        mc_outputs["display_ops_dashboard"] = bool(window.mc_display_ops_dashboard.isChecked())
        mc_outputs["save_raw_runs"] = bool(window.mc_save_raw_runs.isChecked())
        mc_outputs["require_rocket_insertion"] = bool(window.mc_require_rocket_insertion.isChecked())
        mc_outputs["baseline_summary_json"] = window.mc_baseline_summary_json.text().strip()
        gates = {}
        if float(window.mc_gate_min_closest_approach.value()) != 0.0:
            gates["min_closest_approach_km"] = float(window.mc_gate_min_closest_approach.value())
        if float(window.mc_gate_max_duration.value()) != 0.0:
            gates["max_duration_s"] = float(window.mc_gate_max_duration.value())
        if float(window.mc_gate_max_total_dv.value()) != 0.0:
            gates["max_total_dv_m_s"] = float(window.mc_gate_max_total_dv.value())
        if float(window.mc_gate_max_guardrail_events.value()) != 0.0:
            gates["max_guardrail_events"] = float(window.mc_gate_max_guardrail_events.value())
        if gates:
            mc_outputs["gates"] = gates
        else:
            mc_outputs.pop("gates", None)
        return cfg

    def load_knowledge_into_window(self, window: Any, object_key: str, knowledge: dict[str, Any]) -> None:
        conditions = dict(knowledge.get("conditions", {}) or {})
        targets = list(knowledge.get("targets", []) or [])
        getattr(window, f"{object_key}_knowledge_targets_edit").setText(", ".join(str(t) for t in targets))
        getattr(window, f"{object_key}_knowledge_refresh_rate").setValue(float(knowledge.get("refresh_rate_s", 1.0) or 0.0))
        getattr(window, f"{object_key}_knowledge_max_range").setValue(float(conditions.get("max_range_km", 0.0) or 0.0))
        getattr(window, f"{object_key}_knowledge_dropout_prob").setValue(float(conditions.get("dropout_prob", 0.0) or 0.0))
        getattr(window, f"{object_key}_knowledge_solid_angle").setValue(float(conditions.get("solid_angle_sr", 4.0 * 3.141592653589793) or 0.0))
        getattr(window, f"{object_key}_knowledge_require_los").setChecked(bool(conditions.get("require_line_of_sight", False)))
        sensor_pos = list(conditions.get("sensor_position_body_m", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
        sensor_bore = list(conditions.get("sensor_boresight_body", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
        for axis, value in zip(("x", "y", "z"), sensor_pos):
            getattr(window, f"{object_key}_knowledge_sensor_pos_{axis}").setValue(float(value or 0.0))
        for axis, value in zip(("x", "y", "z"), sensor_bore):
            getattr(window, f"{object_key}_knowledge_sensor_bore_{axis}").setValue(float(value or 0.0))
        window._refresh_knowledge_summary_label(object_key)

    def collect_knowledge_from_window(self, window: Any, object_key: str, existing: dict[str, Any] | None = None) -> dict[str, Any]:
        knowledge = copy.deepcopy(existing or {})
        raw_targets = getattr(window, f"{object_key}_knowledge_targets_edit").text().strip()
        knowledge["targets"] = [tok.strip() for tok in raw_targets.split(",") if tok.strip()]
        knowledge["refresh_rate_s"] = float(getattr(window, f"{object_key}_knowledge_refresh_rate").value())
        conditions = dict(knowledge.get("conditions", {}) or {})
        max_range = float(getattr(window, f"{object_key}_knowledge_max_range").value())
        conditions["max_range_km"] = max_range if max_range > 0.0 else None
        conditions["dropout_prob"] = float(getattr(window, f"{object_key}_knowledge_dropout_prob").value())
        solid_angle = float(getattr(window, f"{object_key}_knowledge_solid_angle").value())
        conditions["solid_angle_sr"] = solid_angle if solid_angle > 0.0 else None
        conditions["require_line_of_sight"] = bool(getattr(window, f"{object_key}_knowledge_require_los").isChecked())
        conditions["sensor_position_body_m"] = [
            float(getattr(window, f"{object_key}_knowledge_sensor_pos_x").value()),
            float(getattr(window, f"{object_key}_knowledge_sensor_pos_y").value()),
            float(getattr(window, f"{object_key}_knowledge_sensor_pos_z").value()),
        ]
        sensor_bore = [
            float(getattr(window, f"{object_key}_knowledge_sensor_bore_x").value()),
            float(getattr(window, f"{object_key}_knowledge_sensor_bore_y").value()),
            float(getattr(window, f"{object_key}_knowledge_sensor_bore_z").value()),
        ]
        conditions["sensor_boresight_body"] = sensor_bore if any(abs(v) > 0.0 for v in sensor_bore) else None
        knowledge["conditions"] = conditions
        return knowledge


GUI_CONFIG_ADAPTER = GuiConfigAdapter()
