from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
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
    QSpinBox,
    QDoubleSpinBox,
    QStatusBar,
    QTabWidget,
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
    summarize_config,
    validate_config,
)


SCENARIO_TYPES = [
    "auto",
    "asat_phased_engagement",
    "rocket_ascent",
    "free_tumble_one_orbit",
    "free_tumble_one_orbit_ric",
]
OUTPUT_MODES = ["interactive", "save", "both"]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.repo_root = get_repo_root()
        self.loaded_config_path = get_default_config_path()
        self.current_config = load_config(self.loaded_config_path)
        self.process: QProcess | None = None
        self._build_ui()
        self._load_config_into_widgets(self.current_config)
        self._refresh_yaml()
        self._refresh_summary()
        self._refresh_output_files()

    def _build_ui(self) -> None:
        self.setWindowTitle("NonCooperativeRPO Operator Console")
        self.resize(1360, 900)

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        self.config_selector = QComboBox()
        for path in list_available_configs():
            self.config_selector.addItem(str(path.relative_to(self.repo_root)), str(path))
        current_rel = str(self.loaded_config_path.relative_to(self.repo_root))
        idx = self.config_selector.findText(current_rel)
        if idx >= 0:
            self.config_selector.setCurrentIndex(idx)
        top_bar.addWidget(QLabel("Base Config"))
        top_bar.addWidget(self.config_selector, 1)

        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self._on_load_selected)
        top_bar.addWidget(self.load_button)

        self.open_button = QPushButton("Open...")
        self.open_button.clicked.connect(self._on_open_file)
        top_bar.addWidget(self.open_button)

        self.save_path_edit = QLineEdit(str(self.repo_root / "configs" / "gui_working.yaml"))
        top_bar.addWidget(QLabel("Save Path"))
        top_bar.addWidget(self.save_path_edit, 1)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._on_save)
        top_bar.addWidget(self.save_button)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run)
        top_bar.addWidget(self.run_button)

        root.addLayout(top_bar)

        self.summary_label = QLabel("")
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.summary_label)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self.tabs.addTab(self._build_scenario_tab(), "Scenario")
        self.tabs.addTab(self._build_objects_tab(), "Objects")
        self.tabs.addTab(self._build_outputs_tab(), "Outputs")
        self.tabs.addTab(self._build_yaml_tab(), "Advanced YAML")
        self.tabs.addTab(self._build_results_tab(), "Results")

        status = QStatusBar(self)
        self.setStatusBar(status)
        status.showMessage("Ready.")

    def _build_scenario_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        self.scenario_name_edit = QLineEdit()
        self.scenario_type_combo = QComboBox()
        self.scenario_type_combo.addItems(SCENARIO_TYPES)
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
        self.mc_enabled_check = QCheckBox("Enable Monte Carlo")
        self.mc_iterations_spin = QSpinBox()
        self.mc_iterations_spin.setRange(1, 1000000)
        self.mc_parallel_check = QCheckBox("Parallel")
        self.mc_workers_spin = QSpinBox()
        self.mc_workers_spin.setRange(0, 1024)

        layout.addRow("Scenario Name", self.scenario_name_edit)
        layout.addRow("Scenario Type", self.scenario_type_combo)
        layout.addRow("Duration (s)", self.duration_spin)
        layout.addRow("dt (s)", self.dt_spin)
        layout.addRow("Output Mode", self.output_mode_combo)
        layout.addRow("Output Directory", self.output_dir_edit)
        layout.addRow(self.mc_enabled_check)
        layout.addRow("MC Iterations", self.mc_iterations_spin)
        layout.addRow(self.mc_parallel_check)
        layout.addRow("MC Workers (0=auto)", self.mc_workers_spin)
        return tab

    def _build_objects_tab(self) -> QWidget:
        tab = QWidget()
        layout = QGridLayout(tab)

        target_box = QGroupBox("Target")
        target_form = QFormLayout(target_box)
        self.target_enabled = QCheckBox("Enabled")
        self.target_preset = QLineEdit()
        self.target_mass = QDoubleSpinBox()
        self.target_mass.setRange(0.0, 1.0e9)
        self.target_mass.setDecimals(3)
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
        target_form.addRow("Mass (kg)", self.target_mass)
        target_form.addRow("a_km", self.target_a)
        target_form.addRow("ecc", self.target_ecc)
        target_form.addRow("inc_deg", self.target_inc)
        target_form.addRow("raan_deg", self.target_raan)
        target_form.addRow("argp_deg", self.target_argp)
        target_form.addRow("true_anomaly_deg", self.target_ta)
        layout.addWidget(target_box, 0, 0)

        chaser_box = QGroupBox("Chaser")
        chaser_form = QFormLayout(chaser_box)
        self.chaser_enabled = QCheckBox("Enabled")
        self.chaser_preset = QLineEdit()
        self.chaser_mass = QDoubleSpinBox()
        self.chaser_mass.setRange(0.0, 1.0e9)
        self.chaser_mass.setDecimals(3)
        self.chaser_init_mode = QComboBox()
        self.chaser_init_mode.addItems(["rocket_deployment", "relative_ric_rect", "relative_ric_curv"])
        self.chaser_init_values = [self._make_free_spinbox() for _ in range(6)]
        self.chaser_deploy_time = self._make_free_spinbox()
        chaser_form.addRow(self.chaser_enabled)
        chaser_form.addRow("Preset", self.chaser_preset)
        chaser_form.addRow("Mass (kg)", self.chaser_mass)
        chaser_form.addRow("Init Mode", self.chaser_init_mode)
        chaser_form.addRow("Deploy Time (s)", self.chaser_deploy_time)
        for i, widget in enumerate(self.chaser_init_values):
            chaser_form.addRow(f"Init[{i}]", widget)
        layout.addWidget(chaser_box, 0, 1)

        rocket_box = QGroupBox("Rocket")
        rocket_form = QFormLayout(rocket_box)
        self.rocket_enabled = QCheckBox("Enabled")
        self.rocket_preset = QLineEdit()
        self.rocket_payload = QDoubleSpinBox()
        self.rocket_payload.setRange(0.0, 1.0e9)
        self.rocket_payload.setDecimals(3)
        self.rocket_launch_lat = self._make_free_spinbox()
        self.rocket_launch_lon = self._make_free_spinbox()
        self.rocket_launch_alt = self._make_free_spinbox()
        self.rocket_launch_az = self._make_free_spinbox()
        rocket_form.addRow(self.rocket_enabled)
        rocket_form.addRow("Stack Preset", self.rocket_preset)
        rocket_form.addRow("Payload Mass (kg)", self.rocket_payload)
        rocket_form.addRow("Launch Lat (deg)", self.rocket_launch_lat)
        rocket_form.addRow("Launch Lon (deg)", self.rocket_launch_lon)
        rocket_form.addRow("Launch Alt (km)", self.rocket_launch_alt)
        rocket_form.addRow("Launch Azimuth (deg)", self.rocket_launch_az)
        layout.addWidget(rocket_box, 0, 2)
        return tab

    def _build_outputs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        self.stats_enabled = QCheckBox("Enable Stats")
        self.stats_print_summary = QCheckBox("Print Summary")
        self.stats_save_json = QCheckBox("Save JSON")
        self.stats_save_csv = QCheckBox("Save CSV")
        self.plots_enabled = QCheckBox("Enable Plots")
        self.plots_dpi = QSpinBox()
        self.plots_dpi.setRange(50, 2000)
        self.reference_object_edit = QLineEdit()
        self.figure_ids_edit = QLineEdit()
        self.animations_enabled = QCheckBox("Enable Animations")
        self.animation_types_edit = QLineEdit()
        layout.addRow(self.stats_enabled)
        layout.addRow(self.stats_print_summary)
        layout.addRow(self.stats_save_json)
        layout.addRow(self.stats_save_csv)
        layout.addRow(self.plots_enabled)
        layout.addRow("Plot DPI", self.plots_dpi)
        layout.addRow("Figure IDs (comma-separated)", self.figure_ids_edit)
        layout.addRow("RIC Reference Object", self.reference_object_edit)
        layout.addRow(self.animations_enabled)
        layout.addRow("Animation Types (comma-separated)", self.animation_types_edit)
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
        layout = QGridLayout(tab)
        self.command_label = QLabel("")
        self.command_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(QLabel("Command"), 0, 0)
        layout.addWidget(self.command_label, 0, 1)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(QLabel("Console"), 1, 0)
        layout.addWidget(self.console, 1, 1)

        self.output_files = QListWidget()
        layout.addWidget(QLabel("Output Files"), 1, 2)
        layout.addWidget(self.output_files, 1, 3)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(3, 2)
        return tab

    def _make_free_spinbox(self) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(-1.0e12, 1.0e12)
        widget.setDecimals(6)
        return widget

    def _load_config_into_widgets(self, cfg: dict) -> None:
        sim = cfg.get("simulator", {})
        outputs = cfg.get("outputs", {})
        mc = cfg.get("monte_carlo", {})
        target = cfg.get("target", {})
        chaser = cfg.get("chaser", {})
        rocket = cfg.get("rocket", {})

        self.scenario_name_edit.setText(str(cfg.get("scenario_name", "")))
        self.scenario_type_combo.setCurrentText(str(sim.get("scenario_type", "auto")))
        self.duration_spin.setValue(float(sim.get("duration_s", 3600.0)))
        self.dt_spin.setValue(float(sim.get("dt_s", 1.0)))
        self.output_mode_combo.setCurrentText(str(outputs.get("mode", "interactive")))
        self.output_dir_edit.setText(str(outputs.get("output_dir", "outputs/gui_run")))
        self.mc_enabled_check.setChecked(bool(mc.get("enabled", False)))
        self.mc_iterations_spin.setValue(int(mc.get("iterations", 1)))
        self.mc_parallel_check.setChecked(bool(mc.get("parallel_enabled", False)))
        self.mc_workers_spin.setValue(int(mc.get("parallel_workers", 0)))

        target_specs = dict(target.get("specs", {}) or {})
        target_coes = dict(target.get("initial_state", {}).get("coes", {}) or {})
        self.target_enabled.setChecked(bool(target.get("enabled", True)))
        self.target_preset.setText(str(target_specs.get("preset_satellite", "") or ""))
        self.target_mass.setValue(float(target_specs.get("mass_kg", 400.0) or 0.0))
        self.target_a.setValue(float(target_coes.get("a_km", 7000.0) or 7000.0))
        self.target_ecc.setValue(float(target_coes.get("ecc", 0.001) or 0.0))
        self.target_inc.setValue(float(target_coes.get("inc_deg", 45.0) or 0.0))
        self.target_raan.setValue(float(target_coes.get("raan_deg", 0.0) or 0.0))
        self.target_argp.setValue(float(target_coes.get("argp_deg", 0.0) or 0.0))
        self.target_ta.setValue(float(target_coes.get("true_anomaly_deg", 0.0) or 0.0))

        chaser_specs = dict(chaser.get("specs", {}) or {})
        chaser_init = dict(chaser.get("initial_state", {}) or {})
        self.chaser_enabled.setChecked(bool(chaser.get("enabled", False)))
        self.chaser_preset.setText(str(chaser_specs.get("preset_satellite", "") or ""))
        self.chaser_mass.setValue(float(chaser_specs.get("mass_kg", 200.0) or 0.0))
        if "relative_ric_rect" in chaser_init:
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

        rocket_specs = dict(rocket.get("specs", {}) or {})
        rocket_init = dict(rocket.get("initial_state", {}) or {})
        self.rocket_enabled.setChecked(bool(rocket.get("enabled", False)))
        self.rocket_preset.setText(str(rocket_specs.get("preset_stack", "") or ""))
        self.rocket_payload.setValue(float(rocket_specs.get("payload_mass_kg", 150.0) or 0.0))
        self.rocket_launch_lat.setValue(float(rocket_init.get("launch_lat_deg", 28.5) or 0.0))
        self.rocket_launch_lon.setValue(float(rocket_init.get("launch_lon_deg", -80.6) or 0.0))
        self.rocket_launch_alt.setValue(float(rocket_init.get("launch_alt_km", 0.0) or 0.0))
        self.rocket_launch_az.setValue(float(rocket_init.get("launch_azimuth_deg", 90.0) or 0.0))

        stats = dict(outputs.get("stats", {}) or {})
        plots = dict(outputs.get("plots", {}) or {})
        animations = dict(outputs.get("animations", {}) or {})
        self.stats_enabled.setChecked(bool(stats.get("enabled", True)))
        self.stats_print_summary.setChecked(bool(stats.get("print_summary", True)))
        self.stats_save_json.setChecked(bool(stats.get("save_json", True)))
        self.stats_save_csv.setChecked(bool(stats.get("save_csv", False)))
        self.plots_enabled.setChecked(bool(plots.get("enabled", True)))
        self.plots_dpi.setValue(int(plots.get("dpi", 150) or 150))
        self.figure_ids_edit.setText(", ".join(str(x) for x in (plots.get("figure_ids", []) or [])))
        self.reference_object_edit.setText(str(plots.get("reference_object_id", "") or ""))
        self.animations_enabled.setChecked(bool(animations.get("enabled", False)))
        self.animation_types_edit.setText(", ".join(str(x) for x in (animations.get("types", []) or [])))

    def _collect_config_from_widgets(self) -> dict:
        cfg = dict(self.current_config)
        cfg["scenario_name"] = self.scenario_name_edit.text().strip()
        sim = cfg.setdefault("simulator", {})
        outputs = cfg.setdefault("outputs", {})
        mc = cfg.setdefault("monte_carlo", {})
        target = cfg.setdefault("target", {})
        chaser = cfg.setdefault("chaser", {})
        rocket = cfg.setdefault("rocket", {})

        sim["scenario_type"] = self.scenario_type_combo.currentText()
        sim["duration_s"] = float(self.duration_spin.value())
        sim["dt_s"] = float(self.dt_spin.value())

        outputs["mode"] = self.output_mode_combo.currentText()
        outputs["output_dir"] = self.output_dir_edit.text().strip()
        mc["enabled"] = bool(self.mc_enabled_check.isChecked())
        mc["iterations"] = int(self.mc_iterations_spin.value())
        mc["parallel_enabled"] = bool(self.mc_parallel_check.isChecked())
        mc["parallel_workers"] = int(self.mc_workers_spin.value())

        target["enabled"] = bool(self.target_enabled.isChecked())
        target.setdefault("specs", {})["preset_satellite"] = self.target_preset.text().strip()
        target["specs"]["mass_kg"] = float(self.target_mass.value())
        target.setdefault("initial_state", {})["coes"] = {
            "a_km": float(self.target_a.value()),
            "ecc": float(self.target_ecc.value()),
            "inc_deg": float(self.target_inc.value()),
            "raan_deg": float(self.target_raan.value()),
            "argp_deg": float(self.target_argp.value()),
            "true_anomaly_deg": float(self.target_ta.value()),
        }

        chaser["enabled"] = bool(self.chaser_enabled.isChecked())
        chaser.setdefault("specs", {})["preset_satellite"] = self.chaser_preset.text().strip()
        chaser["specs"]["mass_kg"] = float(self.chaser_mass.value())
        init_mode = self.chaser_init_mode.currentText()
        if init_mode == "rocket_deployment":
            chaser["initial_state"] = {
                "source": "rocket_deployment",
                "deploy_time_s": float(self.chaser_deploy_time.value()),
                "deploy_dv_body_m_s": [float(self.chaser_init_values[i].value()) for i in range(3)],
            }
        else:
            chaser["initial_state"] = {
                init_mode: [float(widget.value()) for widget in self.chaser_init_values],
            }

        rocket["enabled"] = bool(self.rocket_enabled.isChecked())
        rocket.setdefault("specs", {})["preset_stack"] = self.rocket_preset.text().strip()
        rocket["specs"]["payload_mass_kg"] = float(self.rocket_payload.value())
        rocket["initial_state"] = {
            "launch_lat_deg": float(self.rocket_launch_lat.value()),
            "launch_lon_deg": float(self.rocket_launch_lon.value()),
            "launch_alt_km": float(self.rocket_launch_alt.value()),
            "launch_azimuth_deg": float(self.rocket_launch_az.value()),
        }

        stats = outputs.setdefault("stats", {})
        plots = outputs.setdefault("plots", {})
        animations = outputs.setdefault("animations", {})
        stats["enabled"] = bool(self.stats_enabled.isChecked())
        stats["print_summary"] = bool(self.stats_print_summary.isChecked())
        stats["save_json"] = bool(self.stats_save_json.isChecked())
        stats["save_csv"] = bool(self.stats_save_csv.isChecked())
        plots["enabled"] = bool(self.plots_enabled.isChecked())
        plots["dpi"] = int(self.plots_dpi.value())
        figure_ids = [x.strip() for x in self.figure_ids_edit.text().split(",") if x.strip()]
        plots["figure_ids"] = figure_ids
        ref_obj = self.reference_object_edit.text().strip()
        if ref_obj:
            plots["reference_object_id"] = ref_obj
        else:
            plots.pop("reference_object_id", None)
        animations["enabled"] = bool(self.animations_enabled.isChecked())
        animations["types"] = [x.strip() for x in self.animation_types_edit.text().split(",") if x.strip()]
        return cfg

    def _refresh_yaml(self) -> None:
        try:
            self.current_config = self._collect_config_from_widgets()
        except Exception:
            pass
        self.yaml_editor.setPlainText(dump_config_text(self.current_config))

    def _apply_yaml_to_form(self) -> None:
        try:
            cfg = parse_config_text(self.yaml_editor.toPlainText())
            validate_config(cfg)
            self.current_config = cfg
            self._load_config_into_widgets(cfg)
            self._refresh_summary()
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

    def _refresh_summary(self) -> None:
        try:
            cfg = validate_config(self._collect_config_from_widgets())
            self.current_config = cfg.to_dict()
            summary = summarize_config(cfg)
            obj_txt = ", ".join(summary.objects) if summary.objects else "none"
            mc_txt = f"MC x{summary.mc_iterations}" if summary.monte_carlo_enabled else "single run"
            self.summary_label.setText(
                f"Scenario: {summary.scenario_name} | Type: {summary.scenario_type} | "
                f"dt: {summary.dt_s:.6f} s | Duration: {summary.duration_s:.1f} s | "
                f"Objects: {obj_txt} | Mode: {summary.output_mode} | {mc_txt}"
            )
        except Exception as exc:
            self.summary_label.setText(f"Current config invalid: {exc}")

    def _refresh_output_files(self) -> None:
        self.output_files.clear()
        try:
            cfg = validate_config(self._collect_config_from_widgets())
            for path in get_output_files(cfg.outputs.output_dir):
                self.output_files.addItem(str(path.relative_to(self.repo_root)))
        except Exception:
            return

    def _on_load_selected(self) -> None:
        path = Path(self.config_selector.currentData())
        self.loaded_config_path = path
        self.current_config = load_config(path)
        self._load_config_into_widgets(self.current_config)
        self._refresh_yaml()
        self._refresh_summary()
        self._refresh_output_files()
        self.statusBar().showMessage(f"Loaded {path}", 5000)

    def _on_open_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Open Config", str(self.repo_root), "YAML Files (*.yaml *.yml)")
        if not path_str:
            return
        path = Path(path_str)
        self.loaded_config_path = path
        self.current_config = load_config(path)
        self._load_config_into_widgets(self.current_config)
        self._refresh_yaml()
        self._refresh_summary()
        self._refresh_output_files()
        self.statusBar().showMessage(f"Loaded {path}", 5000)

    def _on_save(self) -> None:
        try:
            cfg_dict = self._collect_config_from_widgets()
            save_path = save_config(self.save_path_edit.text().strip(), cfg_dict)
            self.current_config = load_config(save_path)
            self._refresh_yaml()
            self._refresh_summary()
            self.statusBar().showMessage(f"Saved {save_path}", 5000)
        except Exception as exc:
            self._show_error("Save Failed", str(exc))

    def _on_run(self) -> None:
        if self.process is not None:
            self._show_error("Run In Progress", "A simulation is already running.")
            return
        try:
            cfg_dict = self._collect_config_from_widgets()
            save_path = save_config(self.save_path_edit.text().strip(), cfg_dict)
            cmd = build_cli_run_command(save_path)
            self.command_label.setText(" ".join(cmd))
            self.console.clear()
            self.output_files.clear()
            self.process = QProcess(self)
            self.process.setWorkingDirectory(str(self.repo_root))
            self.process.setProcessChannelMode(QProcess.MergedChannels)
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

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 10000)
