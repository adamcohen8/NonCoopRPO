from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import yaml

from PySide6.QtCore import QEvent, QObject, QThread, Qt, Signal
from PySide6.QtGui import QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
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
    dump_config_text,
    get_default_config_path,
    get_gui_capabilities,
    get_output_files,
    get_repo_root,
    list_available_configs,
    load_config,
    parse_config_text,
    run_config_via_api,
    save_config,
    validate_config,
)
from sim.app.models import AnalysisUiProfile
from sim.app.gui_config_adapter import GUI_CONFIG_ADAPTER
from sim.app.pointer_utils import (
    default_params_for_pointer,
    format_vector_text,
    format_yaml_text,
    normalize_form_value,
    parse_vector_text,
    parse_yaml_text,
    pointer_display_name,
    pointer_form_schema,
)
from sim.gui.sections import (
    build_monte_carlo_tab,
    build_objects_tab,
    build_outputs_tab,
    build_results_tab,
    build_scenario_tab,
    build_yaml_tab,
)


GUI_CAPABILITIES = get_gui_capabilities()
OUTPUT_MODES = GUI_CAPABILITIES.output_modes
ORBIT_INTEGRATOR_OPTIONS = GUI_CAPABILITIES.orbit_integrators
ANALYSIS_STUDY_TYPES = GUI_CAPABILITIES.analysis_study_types
SENSITIVITY_METHODS = GUI_CAPABILITIES.sensitivity_methods
MC_MODE_OPTIONS = GUI_CAPABILITIES.monte_carlo_modes
MC_LHS_MODE_OPTIONS = GUI_CAPABILITIES.monte_carlo_lhs_modes
CHASER_INIT_MODE_OPTIONS = GUI_CAPABILITIES.chaser_init_modes
BASE_GUIDANCE_OPTIONS = GUI_CAPABILITIES.base_guidance_options
GUIDANCE_MODIFIER_OPTIONS = GUI_CAPABILITIES.guidance_modifier_options
ORBIT_CONTROL_OPTIONS = GUI_CAPABILITIES.orbit_control_options
ATTITUDE_CONTROL_OPTIONS = GUI_CAPABILITIES.attitude_control_options
MISSION_STRATEGY_OPTIONS = GUI_CAPABILITIES.mission_strategy_options
MISSION_EXECUTION_OPTIONS = GUI_CAPABILITIES.mission_execution_options
SATELLITE_PRESET_OPTIONS = GUI_CAPABILITIES.satellite_presets
ROCKET_PRESET_OPTIONS = GUI_CAPABILITIES.rocket_preset_stacks
FIGURE_ID_OPTIONS = GUI_CAPABILITIES.figure_ids
ANIMATION_TYPE_OPTIONS = GUI_CAPABILITIES.animation_types
MC_PARAMETER_CATEGORIES = GUI_CAPABILITIES.monte_carlo_parameter_categories

PARAMETER_FORM_SCHEMAS = GUI_CAPABILITIES.parameter_form_schemas
ANALYSIS_UI_PROFILES = GUI_CAPABILITIES.analysis_ui_profiles


class _ApiRunWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config_path: Path) -> None:
        super().__init__()
        self.config_path = Path(config_path)

    def run(self) -> None:
        try:
            self.progress.emit(f"Running via API: {self.config_path}\n")
            last_emit = -1
            emit_stride = 1

            def _step_callback(step: int, total: int) -> None:
                nonlocal last_emit, emit_stride
                total_i = max(int(total), 0)
                step_i = max(int(step), 0)
                emit_stride = max(1, total_i // 20) if total_i > 0 else 1
                if step_i not in (0, total_i) and (step_i - last_emit) < emit_stride:
                    return
                last_emit = step_i
                self.progress.emit(f"[progress] step {step_i}/{total_i}\n")

            result = run_config_via_api(self.config_path, step_callback=_step_callback)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.repo_root = get_repo_root()
        self.loaded_config_path = get_default_config_path()
        self.current_config = load_config(self.loaded_config_path)
        self.mc_variations: list[dict] = []
        self._collapse_validation_on_startup = True
        self.run_thread: QThread | None = None
        self.run_worker: _ApiRunWorker | None = None
        self.preview_image_path: Path | None = None
        self.preview_zoom_factor = 1.0
        self.preview_fit_to_window = True
        self.preview_drag_active = False
        self.preview_drag_last_pos = None
        self.results_output_dir: Path | None = None
        self.preview_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.rocket_guidance_modifiers_config: list[dict] = []
        self.output_modes = OUTPUT_MODES
        self.orbit_integrator_options = ORBIT_INTEGRATOR_OPTIONS
        self.analysis_study_types = ANALYSIS_STUDY_TYPES
        self.sensitivity_methods = SENSITIVITY_METHODS
        self.mc_mode_options = MC_MODE_OPTIONS
        self.mc_lhs_mode_options = MC_LHS_MODE_OPTIONS
        self.chaser_init_mode_options = CHASER_INIT_MODE_OPTIONS
        self.base_guidance_options = BASE_GUIDANCE_OPTIONS
        self.orbit_control_options = ORBIT_CONTROL_OPTIONS
        self.attitude_control_options = ATTITUDE_CONTROL_OPTIONS
        self.mission_strategy_options = MISSION_STRATEGY_OPTIONS
        self.mission_execution_options = MISSION_EXECUTION_OPTIONS
        self.satellite_preset_options = SATELLITE_PRESET_OPTIONS
        self.rocket_preset_options = ROCKET_PRESET_OPTIONS
        self.figure_id_options = FIGURE_ID_OPTIONS
        self.animation_type_options = ANIMATION_TYPE_OPTIONS
        self.is_dirty = False
        self._suppress_dirty_tracking = False
        self._suppress_config_selector_load = False
        self._build_ui()
        self._connect_dirty_tracking()
        self.mc_enabled_check.toggled.connect(self._refresh_outputs_mode_ui)
        self.analysis_study_type_combo.currentTextChanged.connect(self._refresh_outputs_mode_ui)
        self.analysis_study_type_combo.currentTextChanged.connect(self._refresh_analysis_editor_ui)
        self.sensitivity_method_combo.currentTextChanged.connect(self._refresh_analysis_editor_ui)
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
        self.setWindowTitle("Orbital Engagement Lab Operator Console")
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
        self.tabs.addTab(self._build_monte_carlo_tab(), "Analysis")
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

        mc_item = QTreeWidgetItem(["Analysis"])
        mc_item.setData(0, Qt.UserRole, 2)
        mc_item.addChild(self._nav_item("Execution", 2))
        mc_item.addChild(self._nav_item("Study Inputs", 2))

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
            self.scenario_description_edit,
            self.output_dir_edit,
            self.reference_object_edit,
            self.mc_baseline_summary_json,
            self.analysis_baseline_path_edit,
            self.save_path_edit,
            self.yaml_editor,
            self.target_knowledge_targets_edit,
            self.chaser_knowledge_targets_edit,
            self.rocket_knowledge_targets_edit,
        ]
        for widget in line_edits:
            signal = getattr(widget, "textChanged", None)
            if signal is not None:
                signal.connect(self._mark_dirty)
        for widget in (
            self.analysis_metrics_edit,
        ):
            widget.textChanged.connect(self._mark_dirty)
        combo_boxes = [
            self.orbit_integrator_combo,
            self.output_mode_combo,
            self.analysis_study_type_combo,
            self.sensitivity_method_combo,
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
            self.analysis_baseline_enable_check,
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
            self.orbit_adaptive_atol_spin,
            self.orbit_adaptive_rtol_spin,
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
            self.analysis_lhs_samples_spin,
            self.target_knowledge_refresh_rate,
            self.target_knowledge_max_range,
            self.target_knowledge_dropout_prob,
            self.target_knowledge_solid_angle,
            self.target_knowledge_sensor_pos_x,
            self.target_knowledge_sensor_pos_y,
            self.target_knowledge_sensor_pos_z,
            self.target_knowledge_sensor_bore_x,
            self.target_knowledge_sensor_bore_y,
            self.target_knowledge_sensor_bore_z,
            self.chaser_knowledge_refresh_rate,
            self.chaser_knowledge_max_range,
            self.chaser_knowledge_dropout_prob,
            self.chaser_knowledge_solid_angle,
            self.chaser_knowledge_sensor_pos_x,
            self.chaser_knowledge_sensor_pos_y,
            self.chaser_knowledge_sensor_pos_z,
            self.chaser_knowledge_sensor_bore_x,
            self.chaser_knowledge_sensor_bore_y,
            self.chaser_knowledge_sensor_bore_z,
            self.rocket_knowledge_refresh_rate,
            self.rocket_knowledge_max_range,
            self.rocket_knowledge_dropout_prob,
            self.rocket_knowledge_solid_angle,
            self.rocket_knowledge_sensor_pos_x,
            self.rocket_knowledge_sensor_pos_y,
            self.rocket_knowledge_sensor_pos_z,
            self.rocket_knowledge_sensor_bore_x,
            self.rocket_knowledge_sensor_bore_y,
            self.rocket_knowledge_sensor_bore_z,
        ]
        spin_boxes.extend(self.chaser_init_values)
        for widget in spin_boxes:
            widget.valueChanged.connect(self._mark_dirty)
        for widget in (
            self.target_knowledge_require_los,
            self.chaser_knowledge_require_los,
            self.rocket_knowledge_require_los,
        ):
            widget.toggled.connect(self._mark_dirty)
        self.orbit_integrator_combo.currentIndexChanged.connect(self._refresh_integrator_visibility)

    def _build_scenario_tab(self) -> QWidget:
        return build_scenario_tab(self)

    def _browse_output_directory(self) -> None:
        current_text = self.output_dir_edit.text().strip()
        start_dir = self.repo_root
        if current_text:
            candidate = Path(current_text).expanduser()
            if not candidate.is_absolute():
                candidate = (self.repo_root / candidate).resolve()
            if candidate.exists():
                start_dir = candidate if candidate.is_dir() else candidate.parent
            else:
                parent = candidate.parent
                if parent.exists():
                    start_dir = parent
        selected = QFileDialog.getExistingDirectory(self, "Select Output Directory", str(start_dir))
        if selected:
            self.output_dir_edit.setText(selected)

    def _build_monte_carlo_tab(self) -> QWidget:
        return build_monte_carlo_tab(self)

    def _build_objects_tab(self) -> QWidget:
        return build_objects_tab(self)

    def _build_outputs_tab(self) -> QWidget:
        return build_outputs_tab(self)

    def _build_yaml_tab(self) -> QWidget:
        return build_yaml_tab(self)

    def _build_results_tab(self) -> QWidget:
        return build_results_tab(self)

    def _make_free_spinbox(self, decimals: int = 6) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(-1.0e12, 1.0e12)
        widget.setDecimals(int(decimals))
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

    def _populate_value_combo(self, combo: QComboBox, options: list[tuple[str, str]]) -> None:
        combo.clear()
        for value, label in options:
            combo.addItem(label, value)

    def _set_combo_data_or_text(self, combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
            return
        text_idx = combo.findText(value)
        if text_idx >= 0:
            combo.setCurrentIndex(text_idx)
            return
        combo.addItem(value, value)
        combo.setCurrentIndex(combo.count() - 1)

    def _make_section_toggle_button(self, object_key: str, section_key: str = "initial_state") -> QPushButton:
        button = QPushButton("+")
        button.setFixedWidth(28)
        button.clicked.connect(lambda: self._toggle_object_section(object_key, section_key))
        return button

    def _wrap_object_panel(self, box: QWidget, title: str) -> QWidget:
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 600;")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(box)
        scroll.setFixedHeight(520)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(title_label)
        layout.addWidget(scroll)
        layout.addStretch(1)
        return container

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

    def _build_knowledge_editor(self, object_key: str) -> QWidget:
        targets_edit = QLineEdit()
        targets_edit.setPlaceholderText("target, chaser")
        targets_edit.textChanged.connect(self._mark_dirty)
        setattr(self, f"{object_key}_knowledge_targets_edit", targets_edit)

        refresh_spin = self._make_free_spinbox()
        max_range_spin = self._make_free_spinbox()
        dropout_spin = self._make_free_spinbox()
        solid_angle_spin = self._make_free_spinbox()
        for widget in (refresh_spin, max_range_spin, dropout_spin, solid_angle_spin):
            self._configure_compact_spinbox(widget)
        require_los = QCheckBox("Require Line Of Sight")
        setattr(self, f"{object_key}_knowledge_refresh_rate", refresh_spin)
        setattr(self, f"{object_key}_knowledge_max_range", max_range_spin)
        setattr(self, f"{object_key}_knowledge_dropout_prob", dropout_spin)
        setattr(self, f"{object_key}_knowledge_solid_angle", solid_angle_spin)
        setattr(self, f"{object_key}_knowledge_require_los", require_los)

        sensor_pos_widgets = [self._make_free_spinbox() for _ in range(3)]
        sensor_bore_widgets = [self._make_free_spinbox() for _ in range(3)]
        for widget in sensor_pos_widgets + sensor_bore_widgets:
            self._configure_compact_spinbox(widget, width=72)
        for axis, widget in zip(("x", "y", "z"), sensor_pos_widgets):
            setattr(self, f"{object_key}_knowledge_sensor_pos_{axis}", widget)
        for axis, widget in zip(("x", "y", "z"), sensor_bore_widgets):
            setattr(self, f"{object_key}_knowledge_sensor_bore_{axis}", widget)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)

        sensor_pos_column = QWidget()
        sensor_pos_layout = QFormLayout(sensor_pos_column)
        sensor_pos_layout.setContentsMargins(0, 0, 0, 0)
        sensor_pos_layout.setSpacing(4)
        for axis_label, widget in zip(("X", "Y", "Z"), sensor_pos_widgets):
            sensor_pos_layout.addRow(axis_label, widget)

        sensor_bore_column = QWidget()
        sensor_bore_layout = QFormLayout(sensor_bore_column)
        sensor_bore_layout.setContentsMargins(0, 0, 0, 0)
        sensor_bore_layout.setSpacing(4)
        for axis_label, widget in zip(("X", "Y", "Z"), sensor_bore_widgets):
            sensor_bore_layout.addRow(axis_label, widget)

        form.addRow("Targets", targets_edit)
        form.addRow("Refresh Rate (s)", refresh_spin)
        form.addRow("Max Range (km)", max_range_spin)
        form.addRow("Dropout Prob", dropout_spin)
        form.addRow("Solid Angle (sr)", solid_angle_spin)
        form.addRow("", require_los)
        form.addRow("Sensor Position Body (m)", sensor_pos_column)
        form.addRow("Sensor Boresight Body", sensor_bore_column)
        return form_widget

    def _edit_knowledge_settings(self, object_key: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit {object_key.title()} Knowledge / Sensor")
        dialog.resize(460, 420)
        layout = QVBoxLayout(dialog)
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(0, 0, 0, 0)

        targets_edit = QLineEdit(getattr(self, f"{object_key}_knowledge_targets_edit").text())
        refresh_spin = self._make_free_spinbox()
        refresh_spin.setValue(float(getattr(self, f"{object_key}_knowledge_refresh_rate").value()))
        max_range_spin = self._make_free_spinbox()
        max_range_spin.setValue(float(getattr(self, f"{object_key}_knowledge_max_range").value()))
        dropout_spin = self._make_free_spinbox()
        dropout_spin.setValue(float(getattr(self, f"{object_key}_knowledge_dropout_prob").value()))
        solid_angle_spin = self._make_free_spinbox()
        solid_angle_spin.setValue(float(getattr(self, f"{object_key}_knowledge_solid_angle").value()))
        require_los = QCheckBox("Require Line Of Sight")
        require_los.setChecked(bool(getattr(self, f"{object_key}_knowledge_require_los").isChecked()))

        sensor_pos_widgets = [self._make_free_spinbox() for _ in range(3)]
        sensor_bore_widgets = [self._make_free_spinbox() for _ in range(3)]
        for axis, widget in zip(("x", "y", "z"), sensor_pos_widgets):
            widget.setValue(float(getattr(self, f"{object_key}_knowledge_sensor_pos_{axis}").value()))
        for axis, widget in zip(("x", "y", "z"), sensor_bore_widgets):
            widget.setValue(float(getattr(self, f"{object_key}_knowledge_sensor_bore_{axis}").value()))

        sensor_pos_row = QWidget()
        sensor_pos_layout = QHBoxLayout(sensor_pos_row)
        sensor_pos_layout.setContentsMargins(0, 0, 0, 0)
        sensor_pos_layout.setSpacing(4)
        for axis_label, widget in zip(("X", "Y", "Z"), sensor_pos_widgets):
            widget.setPrefix(f"{axis_label}:")
            sensor_pos_layout.addWidget(widget)

        sensor_bore_row = QWidget()
        sensor_bore_layout = QHBoxLayout(sensor_bore_row)
        sensor_bore_layout.setContentsMargins(0, 0, 0, 0)
        sensor_bore_layout.setSpacing(4)
        for axis_label, widget in zip(("X", "Y", "Z"), sensor_bore_widgets):
            widget.setPrefix(f"{axis_label}:")
            sensor_bore_layout.addWidget(widget)

        form.addRow("Targets", targets_edit)
        form.addRow("Refresh Rate (s)", refresh_spin)
        form.addRow("Max Range (km)", max_range_spin)
        form.addRow("Dropout Prob", dropout_spin)
        form.addRow("Solid Angle (sr)", solid_angle_spin)
        form.addRow(require_los)
        form.addRow("Sensor Position Body (m)", sensor_pos_row)
        form.addRow("Sensor Boresight Body", sensor_bore_row)
        layout.addWidget(form_widget, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return

        getattr(self, f"{object_key}_knowledge_targets_edit").setText(targets_edit.text())
        getattr(self, f"{object_key}_knowledge_refresh_rate").setValue(float(refresh_spin.value()))
        getattr(self, f"{object_key}_knowledge_max_range").setValue(float(max_range_spin.value()))
        getattr(self, f"{object_key}_knowledge_dropout_prob").setValue(float(dropout_spin.value()))
        getattr(self, f"{object_key}_knowledge_solid_angle").setValue(float(solid_angle_spin.value()))
        getattr(self, f"{object_key}_knowledge_require_los").setChecked(bool(require_los.isChecked()))
        for axis, widget in zip(("x", "y", "z"), sensor_pos_widgets):
            getattr(self, f"{object_key}_knowledge_sensor_pos_{axis}").setValue(float(widget.value()))
        for axis, widget in zip(("x", "y", "z"), sensor_bore_widgets):
            getattr(self, f"{object_key}_knowledge_sensor_bore_{axis}").setValue(float(widget.value()))
        self._refresh_knowledge_summary_label(object_key)
        self._mark_dirty()

    def _refresh_knowledge_summary_label(self, object_key: str) -> None:
        if not hasattr(self, f"{object_key}_knowledge_summary_label"):
            return
        targets = getattr(self, f"{object_key}_knowledge_targets_edit").text().strip() or "none"
        refresh = float(getattr(self, f"{object_key}_knowledge_refresh_rate").value())
        max_range = float(getattr(self, f"{object_key}_knowledge_max_range").value())
        solid_angle = float(getattr(self, f"{object_key}_knowledge_solid_angle").value())
        los = bool(getattr(self, f"{object_key}_knowledge_require_los").isChecked())
        summary = (
            f"Targets: {targets} | dt={refresh:g}s | "
            f"range={'none' if max_range <= 0.0 else f'{max_range:g} km'} | "
            f"solid={'4pi' if solid_angle >= 4.0 * 3.141592653589793 else f'{solid_angle:g} sr'} | "
            f"LOS={'on' if los else 'off'}"
        )
        getattr(self, f"{object_key}_knowledge_summary_label").setText(summary)

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
        return pointer_form_schema(pointer, PARAMETER_FORM_SCHEMAS)

    def _pointer_display_name(self, pointer: dict) -> str:
        return pointer_display_name(pointer)

    def _default_params_for_pointer(self, pointer: dict) -> dict:
        return default_params_for_pointer(pointer)

    def _normalize_form_value(self, field_spec: dict, params: dict, defaults: dict) -> object:
        return normalize_form_value(field_spec, params, defaults)

    def _format_vector_text(self, value: object, length: int | None = None) -> str:
        return format_vector_text(value, length)

    def _parse_vector_text(self, text: str, length: int | None = None) -> list[float]:
        return parse_vector_text(text, length)

    def _format_yaml_text(self, value: object) -> str:
        return format_yaml_text(value)

    def _parse_yaml_text(self, text: str) -> object:
        return parse_yaml_text(text)

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

    def _set_object_section_visible(self, object_key: str, section_key: str, visible: bool) -> None:
        container = getattr(self, f"{object_key}_{section_key}_container")
        button = getattr(self, f"{object_key}_{section_key}_button")
        container.setVisible(bool(visible))
        button.setText("-" if visible else "+")

    def _toggle_object_section(self, object_key: str, section_key: str) -> None:
        container = getattr(self, f"{object_key}_{section_key}_container")
        self._set_object_section_visible(object_key, section_key, not container.isVisible())

    def _set_initial_state_section_visible(self, object_key: str, visible: bool) -> None:
        self._set_object_section_visible(object_key, "initial_state", visible)

    def _toggle_initial_state_section(self, object_key: str) -> None:
        self._toggle_object_section(object_key, "initial_state")

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

    def _selected_sensitivity_method(self) -> str:
        value = self.sensitivity_method_combo.currentData()
        if isinstance(value, str) and value.strip():
            return value.strip()
        raw = self.sensitivity_method_combo.currentText().strip().lower()
        return "lhs" if "latin" in raw else "one_at_a_time"

    def _current_analysis_ui_profile(self) -> AnalysisUiProfile:
        study_type = self._selected_analysis_study_type()
        sensitivity_method = self._selected_sensitivity_method()
        if study_type == "monte_carlo":
            profile_key = "monte_carlo"
        elif sensitivity_method == "lhs":
            profile_key = "sensitivity_lhs"
        else:
            profile_key = "sensitivity_one_at_a_time"
        return ANALYSIS_UI_PROFILES.get(
            profile_key,
            AnalysisUiProfile(
                count_label="Iterations",
                seed_label="Base Seed",
                inputs_title="Study Inputs",
                editor_title="Input Editor",
                help_text="",
                mode_label="Mode",
            ),
        )

    def _format_analysis_metrics_text(self, metrics: list[object] | tuple[object, ...]) -> str:
        return "\n".join(str(metric).strip() for metric in metrics if str(metric).strip())

    def _parse_analysis_metrics_text(self) -> list[str]:
        raw = self.analysis_metrics_edit.toPlainText().strip()
        if not raw:
            return []
        metrics: list[str] = []
        for line in raw.splitlines():
            for token in line.split(","):
                metric = token.strip()
                if metric:
                    metrics.append(metric)
        deduped: list[str] = []
        seen: set[str] = set()
        for metric in metrics:
            if metric not in seen:
                deduped.append(metric)
                seen.add(metric)
        return deduped

    def _browse_analysis_baseline_summary(self) -> None:
        start_dir = self.repo_root
        current_text = self.analysis_baseline_path_edit.text().strip()
        if current_text:
            candidate = Path(current_text).expanduser()
            if not candidate.is_absolute():
                candidate = (self.repo_root / candidate).resolve()
            if candidate.exists():
                start_dir = candidate.parent if candidate.is_file() else candidate
            elif candidate.parent.exists():
                start_dir = candidate.parent
        path_str, _ = QFileDialog.getOpenFileName(self, "Select Baseline Summary", str(start_dir), "JSON Files (*.json)")
        if path_str:
            self.analysis_baseline_path_edit.setText(path_str)

    def _refresh_analysis_editor_ui(self) -> None:
        study_type = self._selected_analysis_study_type()
        sensitivity_active = study_type == "sensitivity"
        lhs_active = sensitivity_active and self._selected_sensitivity_method() == "lhs"
        if sensitivity_active and not lhs_active:
            auto_runs = 0
            for variation in self.mc_variations:
                mode = str(dict(variation or {}).get("mode", "choice") or "choice").strip().lower()
                if mode == "choice":
                    auto_runs += len(list(dict(variation or {}).get("options", []) or []))
                elif mode in {"uniform", "normal"}:
                    auto_runs += 2 if mode == "uniform" else 3
            self.mc_iterations_spin.setValue(max(auto_runs, 1))
        self.analysis_lhs_samples_spin.setValue(int(self.mc_iterations_spin.value()))

        self.analysis_settings_box.setVisible(sensitivity_active)
        analysis_ui = self._current_analysis_ui_profile()
        self.analysis_count_label.setText(analysis_ui.count_label)
        self.analysis_seed_label.setText(analysis_ui.seed_label)
        self.mc_iterations_spin.setEnabled(study_type == "monte_carlo" or lhs_active)
        self.mc_base_seed_spin.setEnabled(study_type == "monte_carlo" or lhs_active)
        self.analysis_lhs_samples_spin.setEnabled(False)
        self.analysis_lhs_samples_spin.setVisible(False)
        lhs_label = self.analysis_settings_box.layout().labelForField(self.analysis_lhs_samples_spin)
        if lhs_label is not None:
            lhs_label.setVisible(False)
        self.sensitivity_method_combo.setEnabled(sensitivity_active)

        self.analysis_inputs_box.setTitle(analysis_ui.inputs_title)
        self.analysis_editor_box.setTitle(analysis_ui.editor_title)
        self.analysis_help_label.setText(analysis_ui.help_text)

        allowed_modes = MC_LHS_MODE_OPTIONS if lhs_active else MC_MODE_OPTIONS
        current_mode = str(self.mc_mode_combo.currentData() or self.mc_mode_combo.currentText()).strip().lower()
        if [str(self.mc_mode_combo.itemData(i) or self.mc_mode_combo.itemText(i)) for i in range(self.mc_mode_combo.count())] != allowed_modes:
            self.mc_mode_combo.blockSignals(True)
            self._populate_value_combo(self.mc_mode_combo, [(mode, mode) for mode in allowed_modes])
            self.mc_mode_combo.blockSignals(False)
        if current_mode not in allowed_modes:
            current_mode = allowed_modes[0]
        self._set_combo_data_or_text(self.mc_mode_combo, current_mode)
        mode_label = self.mc_variation_form.labelForField(self.mc_mode_combo)
        if mode_label is not None:
            mode_label.setText(analysis_ui.mode_label)
        self._refresh_mc_mode_ui()

    def _refresh_mc_mode_ui(self) -> None:
        mode = str(self.mc_mode_combo.currentData() or self.mc_mode_combo.currentText()).strip().lower()
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
        lhs_active = self._selected_analysis_study_type() == "sensitivity" and self._selected_sensitivity_method() == "lhs"
        if mode == "choice":
            options = list(variation.get("options", []) or [])
            return f"{label} | choice | {len(options)} option(s)"
        if mode == "uniform":
            prefix = "distribution" if lhs_active else "uniform"
            return f"{label} | {prefix} | {variation.get('low')} to {variation.get('high')}"
        if mode == "normal":
            prefix = "distribution" if lhs_active else "normal"
            return f"{label} | {prefix} | mean {variation.get('mean')} std {variation.get('std')}"
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
        self._refresh_analysis_editor_ui()

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
        allowed_modes = [str(self.mc_mode_combo.itemData(i) or self.mc_mode_combo.itemText(i)) for i in range(self.mc_mode_combo.count())]
        fallback_mode = allowed_modes[0] if allowed_modes else "choice"
        self._set_combo_data_or_text(self.mc_mode_combo, mode if mode in set(allowed_modes) else fallback_mode)
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
        default_mode = "uniform" if (self._selected_analysis_study_type() == "sensitivity" and self._selected_sensitivity_method() == "lhs") else "choice"
        self._set_combo_data_or_text(self.mc_mode_combo, default_mode)
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
            raise ValueError("Select a parameter or enter a custom path.")
        mode = str(self.mc_mode_combo.currentData() or self.mc_mode_combo.currentText()).strip().lower()
        lhs_active = self._selected_analysis_study_type() == "sensitivity" and self._selected_sensitivity_method() == "lhs"
        variation: dict[str, object] = {"parameter_path": parameter_path, "mode": mode}
        if mode == "choice":
            if lhs_active:
                raise ValueError("LHS parameters must use uniform or normal distributions.")
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
            noun = "analysis input" if self._selected_analysis_study_type() == "monte_carlo" else "analysis parameter"
            self.statusBar().showMessage(f"{action} {noun}.", 5000)
        except Exception as exc:
            self._show_error("Invalid Analysis Input", str(exc))

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
        self.statusBar().showMessage("Removed analysis input.", 5000)

    def _load_config_into_widgets(self, cfg: dict) -> None:
        GUI_CONFIG_ADAPTER.load_into_window(self, cfg)

    def _collect_config_from_widgets(self) -> dict:
        return GUI_CONFIG_ADAPTER.collect_from_window(self, self.current_config)

    def _load_knowledge_into_widgets(self, object_key: str, knowledge: dict) -> None:
        GUI_CONFIG_ADAPTER.load_knowledge_into_window(self, object_key, knowledge)

    def _collect_knowledge_from_widgets(self, object_key: str, existing: dict | None = None) -> dict:
        return GUI_CONFIG_ADAPTER.collect_knowledge_from_window(self, object_key, existing)

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
        analysis_summary_path = output_dir / "master_analysis_sensitivity_summary.json"
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
        if analysis_summary_path.exists():
            try:
                data = json.loads(analysis_summary_path.read_text(encoding="utf-8"))
                text.append(self._format_json_summary("Analysis Summary", data))
            except Exception as exc:
                text.append(f"Failed to read {analysis_summary_path.name}: {exc}")
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
        if self.run_thread is not None:
            self._show_error("Run In Progress", "A simulation is already running.")
            return
        try:
            cfg_dict = self._collect_config_from_widgets()
            save_path = save_config(self.save_path_edit.text().strip(), cfg_dict)
            run_config_path = save_path
            self.results_output_dir = None
            if str(cfg_dict.get("outputs", {}).get("mode", "")).strip().lower() == "interactive":
                run_config_path = self._build_preview_run_config(cfg_dict, save_path)
            self.console.clear()
            self.output_files.clear()
            self.run_thread = QThread(self)
            self.run_worker = _ApiRunWorker(run_config_path)
            self.run_worker.moveToThread(self.run_thread)
            self.run_thread.started.connect(self.run_worker.run)
            self.run_worker.progress.connect(self._append_console_text)
            self.run_worker.finished.connect(self._on_run_finished)
            self.run_worker.failed.connect(self._on_run_failed)
            self.run_worker.finished.connect(self.run_thread.quit)
            self.run_worker.failed.connect(self.run_thread.quit)
            self.run_thread.finished.connect(self._cleanup_run_worker)
            self.run_thread.start()
            self.run_button.setEnabled(False)
            self.statusBar().showMessage("Simulation running...")
            self.tabs.setCurrentIndex(5)
        except Exception as exc:
            self._show_error("Run Failed", str(exc))

    def _append_console_text(self, txt: str) -> None:
        if txt:
            self.console.moveCursor(QTextCursor.End)
            self.console.insertPlainText(txt)
            self.console.ensureCursorVisible()

    def _on_run_finished(self, result) -> None:
        self.run_button.setEnabled(True)
        if getattr(result, "output_dir", None) and self.results_output_dir is None:
            self.results_output_dir = self._resolve_output_dir(str(result.output_dir))
        self._append_console_text(str(getattr(result, "stdout", "")))
        self.statusBar().showMessage("Simulation finished.", 10000)
        self._refresh_output_files()

    def _on_run_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self._append_console_text(f"\nRun failed: {message}\n")
        self._show_error("Run Failed", message)

    def _cleanup_run_worker(self) -> None:
        if self.run_worker is not None:
            self.run_worker.deleteLater()
        if self.run_thread is not None:
            self.run_thread.deleteLater()
        self.run_worker = None
        self.run_thread = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.preview_image_path is not None:
            self._update_image_preview()

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 10000)

    def _build_preview_run_config(self, cfg_dict: dict, saved_config_path: Path) -> Path:
        self.preview_temp_dir = tempfile.TemporaryDirectory(prefix="orbital_engagement_lab_preview_")
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
        analysis = dict(cfg_dict.get("analysis", {}) or {})
        if bool(analysis.get("enabled", False)) and str(analysis.get("study_type", "")).strip().lower() == "sensitivity":
            sensitivity = dict(analysis.get("sensitivity", {}) or {})
            for param in list(sensitivity.get("parameters", []) or []):
                path = str(dict(param or {}).get("parameter_path", "") or "").strip()
                if path and not self._path_exists(cfg_dict, path):
                    return f"Warning: analysis parameter path missing: {path}"
            return ""
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
        self.setWindowTitle(f"{marker}{self.loaded_config_path.name} - Orbital Engagement Lab Operator Console")

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

    def _selected_analysis_study_type(self) -> str:
        value = self.analysis_study_type_combo.currentData()
        if isinstance(value, str) and value.strip():
            return value.strip()
        raw = self.analysis_study_type_combo.currentText().strip().lower()
        return "sensitivity" if raw == "sensitivity" else "monte_carlo"

    def _toggle_validation_panel(self) -> None:
        if self.validation_panel.isVisible():
            self.validation_panel.hide()
            self.validation_toggle.setText("Show Details")
        else:
            self.validation_panel.show()
            self.validation_toggle.setText("Hide Details")

    def _refresh_outputs_mode_ui(self) -> None:
        analysis_enabled = bool(self.mc_enabled_check.isChecked())
        study_type = self._selected_analysis_study_type()
        if analysis_enabled:
            self.outputs_stack.setCurrentIndex(1)
            if study_type == "sensitivity":
                self.outputs_mode_label.setText(
                    "Sensitivity analysis is enabled. Configure campaign outputs here; tracked metrics, method, and baseline options live in the Analysis tab."
                )
            else:
                self.outputs_mode_label.setText(
                    "Monte Carlo analysis is enabled. Configure campaign outputs, dashboards, and pass/fail gates here."
                )
        else:
            self.outputs_stack.setCurrentIndex(0)
            self.outputs_mode_label.setText(
                "Analysis is disabled. Configure single-run plots, stats, and animations here."
            )

    def _refresh_substep_visibility(self) -> None:
        self.orbit_substep_spin.setVisible(bool(self.orbit_substep_enabled_check.isChecked()))
        self.attitude_substep_spin.setVisible(bool(self.attitude_substep_enabled_check.isChecked()))

    def _refresh_integrator_visibility(self) -> None:
        adaptive_integrator = self.orbit_integrator_combo.currentText() in ("rkf78", "dopri5", "adaptive")
        layout = getattr(self, "scenario_form_layout", None)
        if layout is not None and hasattr(layout, "setRowVisible"):
            layout.setRowVisible(self.orbit_adaptive_atol_spin, adaptive_integrator)
            layout.setRowVisible(self.orbit_adaptive_rtol_spin, adaptive_integrator)
        else:
            self.orbit_adaptive_atol_spin.setVisible(adaptive_integrator)
            self.orbit_adaptive_rtol_spin.setVisible(adaptive_integrator)
