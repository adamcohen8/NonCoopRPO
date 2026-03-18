from __future__ import annotations

from pathlib import Path
import traceback

import streamlit as st
import yaml

from sim.gui.helpers import (
    CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    build_run_command,
    dump_yaml_text,
    ensure_sections,
    list_config_files,
    list_output_files,
    load_config_dict,
    read_yaml_file,
    run_simulation_cli,
    save_config_dict,
    summarize_config,
    validate_config_dict,
    validate_yaml_text,
)


SCENARIO_TYPES = [
    "auto",
    "asat_phased_engagement",
    "rocket_ascent",
    "free_tumble_one_orbit",
    "free_tumble_one_orbit_ric",
]
OUTPUT_MODES = ["interactive", "save", "both"]
SUPPORTED_FIGURES = [
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
]
SUPPORTED_ANIMATIONS = ["ground_track", "ground_track_multi"]


def _default_save_path() -> str:
    return str(CONFIG_DIR / "gui_working.yaml")


def _load_into_session(path: Path) -> None:
    cfg = ensure_sections(load_config_dict(path))
    st.session_state.editor_config = cfg
    st.session_state.yaml_text = dump_yaml_text(cfg)
    st.session_state.loaded_config_path = str(path)
    st.session_state.validation_message = f"Loaded {path}"


def _sync_yaml_from_config() -> None:
    st.session_state.editor_config = ensure_sections(st.session_state.editor_config)
    st.session_state.yaml_text = dump_yaml_text(st.session_state.editor_config)


def _load_yaml_text_into_config(yaml_text: str) -> None:
    cfg = ensure_sections(yaml.safe_load(yaml_text) or {})
    validate_config_dict(cfg)
    st.session_state.editor_config = cfg
    st.session_state.yaml_text = dump_yaml_text(cfg)
    st.session_state.validation_message = "YAML applied to form state."


def _init_state() -> None:
    if "editor_config" not in st.session_state:
        _load_into_session(DEFAULT_CONFIG_PATH)
    st.session_state.setdefault("save_path", _default_save_path())
    st.session_state.setdefault("run_result", None)
    st.session_state.setdefault("validation_message", "")


def _edit_pointer_form(section_key: str, pointer_key: str, title: str) -> None:
    cfg = st.session_state.editor_config
    section = cfg.setdefault(section_key, {})
    pointer = dict(section.get(pointer_key, {}) or {})
    st.markdown(f"**{title}**")
    with st.form(f"{section_key}_{pointer_key}_form"):
        kind = st.text_input("Kind", value=str(pointer.get("kind", "python")))
        module = st.text_input("Module", value=str(pointer.get("module", "") or ""))
        class_name = st.text_input("Class Name", value=str(pointer.get("class_name", "") or ""))
        function = st.text_input("Function", value=str(pointer.get("function", "") or ""))
        file_path = st.text_input("File", value=str(pointer.get("file", "") or ""))
        params_text = st.text_area(
            "Params YAML",
            value=yaml.safe_dump(dict(pointer.get("params", {}) or {}), sort_keys=False, allow_unicode=False),
            height=140,
        )
        submitted = st.form_submit_button(f"Apply {title}")
        if submitted:
            try:
                params = yaml.safe_load(params_text) or {}
                if not isinstance(params, dict):
                    raise ValueError("Pointer params must be a mapping/object.")
                updated = {
                    "kind": kind.strip() or "python",
                    "params": params,
                }
                if module.strip():
                    updated["module"] = module.strip()
                if class_name.strip():
                    updated["class_name"] = class_name.strip()
                if function.strip():
                    updated["function"] = function.strip()
                if file_path.strip():
                    updated["file"] = file_path.strip()
                section[pointer_key] = updated
                _sync_yaml_from_config()
                st.success(f"Updated {section_key}.{pointer_key}.")
            except Exception as exc:
                st.error(str(exc))


def _render_sidebar() -> None:
    st.sidebar.header("Config")
    config_files = list_config_files()
    config_options = [str(path.relative_to(REPO_ROOT)) for path in config_files]
    loaded_rel = str(Path(st.session_state.loaded_config_path).relative_to(REPO_ROOT))
    default_idx = config_options.index(loaded_rel) if loaded_rel in config_options else 0
    selected_rel = st.sidebar.selectbox("Base Config", options=config_options, index=default_idx)
    selected_path = REPO_ROOT / selected_rel
    if st.sidebar.button("Load Selected Config", use_container_width=True):
        _load_into_session(selected_path)

    st.sidebar.text_input("Save Path", key="save_path")
    if st.sidebar.button("Save Current Config", use_container_width=True):
        try:
            cfg = validate_config_dict(ensure_sections(st.session_state.editor_config))
            save_path = save_config_dict(st.session_state.save_path, cfg.to_dict())
            st.sidebar.success(f"Saved {save_path}")
        except Exception as exc:
            st.sidebar.error(str(exc))

    if st.sidebar.button("Run Current Config", type="primary", use_container_width=True):
        try:
            cfg = validate_config_dict(ensure_sections(st.session_state.editor_config))
            save_path = save_config_dict(st.session_state.save_path, cfg.to_dict())
            with st.spinner("Running simulation..."):
                st.session_state.run_result = run_simulation_cli(save_path)
            st.sidebar.success(f"Run finished with code {st.session_state.run_result['returncode']}.")
        except Exception as exc:
            st.session_state.run_result = {
                "command": build_run_command(st.session_state.save_path),
                "returncode": -1,
                "stdout": "",
                "stderr": f"{exc}\n\n{traceback.format_exc()}",
                "elapsed_s": 0.0,
            }
            st.sidebar.error(str(exc))

    st.sidebar.caption(f"Repo root: {REPO_ROOT}")
    st.sidebar.caption("Existing CLI usage remains unchanged: `python run_simulation.py --config ...`")


def _render_summary() -> None:
    try:
        cfg = validate_config_dict(ensure_sections(st.session_state.editor_config))
        summary = summarize_config(cfg)
        st.caption(
            f"Scenario `{summary['scenario_name']}` | type `{summary['scenario_type']}` | "
            f"dt `{summary['dt_s']}` s | duration `{summary['duration_s']}` s | "
            f"objects `{', '.join(summary['objects']) if summary['objects'] else 'none'}`"
        )
    except Exception as exc:
        st.warning(f"Current form state is not valid yet: {exc}")


def _render_scenario_tab() -> None:
    cfg = st.session_state.editor_config
    sim = cfg.setdefault("simulator", {})
    outputs = cfg.setdefault("outputs", {})
    mc = cfg.setdefault("monte_carlo", {})
    with st.form("scenario_form"):
        scenario_name = st.text_input("Scenario Name", value=str(cfg.get("scenario_name", "")))
        scenario_type = st.selectbox(
            "Scenario Type",
            options=SCENARIO_TYPES,
            index=SCENARIO_TYPES.index(str(sim.get("scenario_type", "auto")))
            if str(sim.get("scenario_type", "auto")) in SCENARIO_TYPES
            else 0,
        )
        col1, col2 = st.columns(2)
        duration_s = col1.number_input("Duration (s)", min_value=0.001, value=float(sim.get("duration_s", 3600.0)))
        dt_s = col2.number_input("dt (s)", min_value=0.000001, value=float(sim.get("dt_s", 1.0)), format="%.6f")
        output_mode = st.selectbox(
            "Output Mode",
            options=OUTPUT_MODES,
            index=OUTPUT_MODES.index(str(outputs.get("mode", "interactive")))
            if str(outputs.get("mode", "interactive")) in OUTPUT_MODES
            else 0,
        )
        output_dir = st.text_input("Output Directory", value=str(outputs.get("output_dir", "outputs/gui_run")))
        st.markdown("**Monte Carlo**")
        mc_enabled = st.checkbox("Enable Monte Carlo", value=bool(mc.get("enabled", False)))
        col3, col4, col5 = st.columns(3)
        iterations = col3.number_input("Iterations", min_value=1, value=int(mc.get("iterations", 1)), step=1)
        parallel_enabled = col4.checkbox("Parallel", value=bool(mc.get("parallel_enabled", False)))
        parallel_workers = col5.number_input("Workers (0=auto)", min_value=0, value=int(mc.get("parallel_workers", 0)), step=1)
        submitted = st.form_submit_button("Apply Scenario Settings")
        if submitted:
            cfg["scenario_name"] = scenario_name
            sim["scenario_type"] = scenario_type
            sim["duration_s"] = float(duration_s)
            sim["dt_s"] = float(dt_s)
            outputs["mode"] = output_mode
            outputs["output_dir"] = output_dir
            mc["enabled"] = bool(mc_enabled)
            mc["iterations"] = int(iterations)
            mc["parallel_enabled"] = bool(parallel_enabled)
            mc["parallel_workers"] = int(parallel_workers)
            _sync_yaml_from_config()
            st.success("Scenario settings updated.")


def _render_objects_tab() -> None:
    cfg = st.session_state.editor_config
    rocket = cfg.setdefault("rocket", {})
    chaser = cfg.setdefault("chaser", {})
    target = cfg.setdefault("target", {})
    with st.form("objects_form"):
        st.markdown("**Target**")
        tcol1, tcol2, tcol3 = st.columns(3)
        target_enabled = tcol1.checkbox("Target Enabled", value=bool(target.get("enabled", True)))
        target_preset = tcol2.text_input("Target Preset", value=str(target.get("specs", {}).get("preset_satellite", "") or ""))
        target_mass = tcol3.number_input("Target Mass (kg)", min_value=0.0, value=float(target.get("specs", {}).get("mass_kg", 400.0)))
        coes = dict(target.get("initial_state", {}).get("coes", {}) or {})
        c1, c2, c3 = st.columns(3)
        a_km = c1.number_input("a_km", value=float(coes.get("a_km", 7000.0)))
        ecc = c2.number_input("ecc", min_value=0.0, value=float(coes.get("ecc", 0.001)), format="%.6f")
        inc_deg = c3.number_input("inc_deg", value=float(coes.get("inc_deg", 45.0)))
        c4, c5, c6 = st.columns(3)
        raan_deg = c4.number_input("raan_deg", value=float(coes.get("raan_deg", 0.0)))
        argp_deg = c5.number_input("argp_deg", value=float(coes.get("argp_deg", 0.0)))
        ta_deg = c6.number_input("true_anomaly_deg", value=float(coes.get("true_anomaly_deg", 0.0)))

        st.markdown("**Chaser**")
        chcol1, chcol2, chcol3 = st.columns(3)
        chaser_enabled = chcol1.checkbox("Chaser Enabled", value=bool(chaser.get("enabled", False)))
        chaser_preset = chcol2.text_input("Chaser Preset", value=str(chaser.get("specs", {}).get("preset_satellite", "") or ""))
        chaser_mass = chcol3.number_input("Chaser Mass (kg)", min_value=0.0, value=float(chaser.get("specs", {}).get("mass_kg", 200.0)))
        source = st.selectbox(
            "Chaser Init Mode",
            options=["rocket_deployment", "relative_ric_rect", "relative_ric_curv"],
            index=0
            if str(chaser.get("initial_state", {}).get("source", "rocket_deployment")) == "rocket_deployment"
            else (1 if "relative_ric_rect" in chaser.get("initial_state", {}) else 2),
        )
        if source == "rocket_deployment":
            d1, d2 = st.columns(2)
            deploy_time_s = d1.number_input(
                "Deploy Time (s)",
                value=float(chaser.get("initial_state", {}).get("deploy_time_s", 900.0)),
            )
            deploy_dv = list(chaser.get("initial_state", {}).get("deploy_dv_body_m_s", [10.0, 0.0, 0.0]) or [10.0, 0.0, 0.0])
            dv_cols = st.columns(3)
            deploy_dv_vals = [
                dv_cols[i].number_input(f"Deploy dV body [{i}] (m/s)", value=float(deploy_dv[i]))
                for i in range(3)
            ]
        else:
            ric_key = "relative_ric_rect" if source == "relative_ric_rect" else "relative_ric_curv"
            ric_default = list(chaser.get("initial_state", {}).get(ric_key, [2.0, -8.0, 1.2, 0.0008, -0.0012, 0.0004]))
            ric_cols = st.columns(3)
            ric_vals = []
            for i in range(6):
                ric_vals.append(ric_cols[i % 3].number_input(f"{ric_key}[{i}]", value=float(ric_default[i]), format="%.6f"))
            deploy_time_s = 0.0
            deploy_dv_vals = [0.0, 0.0, 0.0]

        st.markdown("**Rocket**")
        rcol1, rcol2, rcol3 = st.columns(3)
        rocket_enabled = rcol1.checkbox("Rocket Enabled", value=bool(rocket.get("enabled", False)))
        rocket_preset = rcol2.text_input("Rocket Stack Preset", value=str(rocket.get("specs", {}).get("preset_stack", "") or ""))
        payload_mass = rcol3.number_input("Payload Mass (kg)", min_value=0.0, value=float(rocket.get("specs", {}).get("payload_mass_kg", 150.0)))
        l1, l2, l3, l4 = st.columns(4)
        launch_lat = l1.number_input("Launch Lat (deg)", value=float(rocket.get("initial_state", {}).get("launch_lat_deg", 28.5)))
        launch_lon = l2.number_input("Launch Lon (deg)", value=float(rocket.get("initial_state", {}).get("launch_lon_deg", -80.6)))
        launch_alt = l3.number_input("Launch Alt (km)", value=float(rocket.get("initial_state", {}).get("launch_alt_km", 0.0)))
        launch_az = l4.number_input("Launch Azimuth (deg)", value=float(rocket.get("initial_state", {}).get("launch_azimuth_deg", 90.0)))
        submitted = st.form_submit_button("Apply Object Settings")
        if submitted:
            target["enabled"] = bool(target_enabled)
            target.setdefault("specs", {})["preset_satellite"] = target_preset
            target["specs"]["mass_kg"] = float(target_mass)
            target.setdefault("initial_state", {})["coes"] = {
                "a_km": float(a_km),
                "ecc": float(ecc),
                "inc_deg": float(inc_deg),
                "raan_deg": float(raan_deg),
                "argp_deg": float(argp_deg),
                "true_anomaly_deg": float(ta_deg),
            }

            chaser["enabled"] = bool(chaser_enabled)
            chaser.setdefault("specs", {})["preset_satellite"] = chaser_preset
            chaser["specs"]["mass_kg"] = float(chaser_mass)
            chaser.setdefault("initial_state", {})
            if source == "rocket_deployment":
                chaser["initial_state"] = {
                    "source": "rocket_deployment",
                    "deploy_time_s": float(deploy_time_s),
                    "deploy_dv_body_m_s": [float(v) for v in deploy_dv_vals],
                }
            else:
                chaser["initial_state"] = {source: [float(v) for v in ric_vals]}

            rocket["enabled"] = bool(rocket_enabled)
            rocket.setdefault("specs", {})["preset_stack"] = rocket_preset
            rocket["specs"]["payload_mass_kg"] = float(payload_mass)
            rocket["initial_state"] = {
                "launch_lat_deg": float(launch_lat),
                "launch_lon_deg": float(launch_lon),
                "launch_alt_km": float(launch_alt),
                "launch_azimuth_deg": float(launch_az),
            }
            _sync_yaml_from_config()
            st.success("Object settings updated.")


def _render_controllers_tab() -> None:
    st.write("These fields stay generic on purpose. For complex mission-objective wiring, use the Advanced YAML tab.")
    object_key = st.selectbox("Object", options=["rocket", "chaser", "target"], key="controller_object_select")
    _edit_pointer_form(object_key, "guidance", "Guidance")
    _edit_pointer_form(object_key, "orbit_control", "Orbit Control")
    _edit_pointer_form(object_key, "attitude_control", "Attitude Control")


def _render_outputs_tab() -> None:
    cfg = st.session_state.editor_config
    outputs = cfg.setdefault("outputs", {})
    stats = outputs.setdefault("stats", {})
    plots = outputs.setdefault("plots", {})
    animations = outputs.setdefault("animations", {})
    with st.form("outputs_form"):
        st.markdown("**Stats**")
        scol1, scol2, scol3, scol4 = st.columns(4)
        stats_enabled = scol1.checkbox("Enabled", value=bool(stats.get("enabled", True)))
        print_summary = scol2.checkbox("Print Summary", value=bool(stats.get("print_summary", True)))
        save_json = scol3.checkbox("Save JSON", value=bool(stats.get("save_json", True)))
        save_csv = scol4.checkbox("Save CSV", value=bool(stats.get("save_csv", False)))

        st.markdown("**Plots**")
        pcol1, pcol2 = st.columns(2)
        plots_enabled = pcol1.checkbox("Plots Enabled", value=bool(plots.get("enabled", True)))
        dpi = pcol2.number_input("DPI", min_value=50, value=int(plots.get("dpi", 150)), step=10)
        figure_ids = st.multiselect(
            "Figure IDs",
            options=SUPPORTED_FIGURES,
            default=list(plots.get("figure_ids", []) or []),
        )
        reference_object_id = st.text_input(
            "RIC Reference Object ID",
            value=str(plots.get("reference_object_id", "") or ""),
        )

        st.markdown("**Animations**")
        acol1, acol2, acol3 = st.columns(3)
        animations_enabled = acol1.checkbox("Animations Enabled", value=bool(animations.get("enabled", False)))
        fps = acol2.number_input("FPS", min_value=1, value=int(animations.get("fps", 20)), step=1)
        speed_multiple = acol3.number_input(
            "Speed Multiple",
            min_value=0.1,
            value=float(animations.get("speed_multiple", 50.0)),
            format="%.2f",
        )
        animation_types = st.multiselect(
            "Animation Types",
            options=SUPPORTED_ANIMATIONS,
            default=list(animations.get("types", []) or []),
        )
        submitted = st.form_submit_button("Apply Output Settings")
        if submitted:
            stats.update(
                {
                    "enabled": bool(stats_enabled),
                    "print_summary": bool(print_summary),
                    "save_json": bool(save_json),
                    "save_csv": bool(save_csv),
                }
            )
            plots.update(
                {
                    "enabled": bool(plots_enabled),
                    "figure_ids": list(figure_ids),
                    "dpi": int(dpi),
                }
            )
            if reference_object_id.strip():
                plots["reference_object_id"] = reference_object_id.strip()
            else:
                plots.pop("reference_object_id", None)
            animations.update(
                {
                    "enabled": bool(animations_enabled),
                    "types": list(animation_types),
                    "fps": int(fps),
                    "speed_multiple": float(speed_multiple),
                }
            )
            _sync_yaml_from_config()
            st.success("Output settings updated.")


def _render_yaml_tab() -> None:
    st.text_area("YAML", key="yaml_text", height=620)
    col1, col2 = st.columns(2)
    if col1.button("Validate YAML", use_container_width=True):
        try:
            validate_yaml_text(st.session_state.yaml_text)
            st.success("YAML is valid.")
        except Exception as exc:
            st.error(str(exc))
    if col2.button("Apply YAML To Forms", use_container_width=True):
        try:
            _load_yaml_text_into_config(st.session_state.yaml_text)
            st.success("Form state updated from YAML.")
        except Exception as exc:
            st.error(str(exc))


def _render_results_tab() -> None:
    result = st.session_state.run_result
    if not result:
        st.info("No run has been launched from the GUI yet.")
        return
    st.markdown("**Run Result**")
    st.code(" ".join(result["command"]))
    st.write(f"Return code: `{result['returncode']}`")
    st.write(f"Elapsed: `{result['elapsed_s']:.2f} s`")
    if result["stderr"]:
        st.markdown("**stderr**")
        st.code(result["stderr"])
    st.markdown("**stdout**")
    st.code(result["stdout"] or "(no stdout)")

    try:
        cfg = validate_config_dict(ensure_sections(st.session_state.editor_config))
        files = list_output_files(cfg.outputs.output_dir)
        st.markdown("**Generated Files**")
        if files:
            for path in files:
                st.write(str(path.relative_to(REPO_ROOT)))
        else:
            st.caption("No files found in the configured output directory yet.")
    except Exception as exc:
        st.warning(f"Could not inspect output files: {exc}")


def main() -> None:
    st.set_page_config(page_title="NonCooperativeRPO GUI", layout="wide")
    _init_state()
    _render_sidebar()
    st.title("NonCooperativeRPO Streamlit GUI")
    _render_summary()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Scenario", "Objects", "Controllers", "Outputs", "Advanced YAML"])
    with tab1:
        _render_scenario_tab()
    with tab2:
        _render_objects_tab()
    with tab3:
        _render_controllers_tab()
    with tab4:
        _render_outputs_tab()
    with tab5:
        _render_yaml_tab()

    st.divider()
    _render_results_tab()


if __name__ == "__main__":
    main()

