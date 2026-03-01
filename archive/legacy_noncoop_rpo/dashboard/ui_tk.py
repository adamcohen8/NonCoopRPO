from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

from ..utils import master_log_summary_text
from .controller import run_master_sim_from_config
from .schema import DashboardConfig


class MasterSimDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Master Simulator Dashboard (Scaffold)")
        self.root.geometry("920x760")

        self.status_var = tk.StringVar(value="Ready")
        self._build_vars()
        self._build_layout()

    def _build_vars(self) -> None:
        defaults = DashboardConfig()
        self.vars = {
            "target_alt_km": tk.StringVar(value=str(defaults.target_alt_km)),
            "target_incl_deg": tk.StringVar(value=str(defaults.target_incl_deg)),
            "target_raan_deg": tk.StringVar(value=str(defaults.target_raan_deg)),
            "insertion_delta_alt_km": tk.StringVar(value=str(defaults.insertion_delta_alt_km)),
            "launch_site_lat_deg": tk.StringVar(value=str(defaults.launch_site_lat_deg)),
            "launch_site_lon_deg": tk.StringVar(value=str(defaults.launch_site_lon_deg)),
            "launch_site_alt_km": tk.StringVar(value=str(defaults.launch_site_alt_km)),
            "launch_timing_mode": tk.StringVar(value=defaults.launch_timing_mode),
            "rocket_preset": tk.StringVar(value=defaults.rocket_preset),
            "dt_s": tk.StringVar(value=str(defaults.dt_s)),
            "pre_sim_duration_s": tk.StringVar(value=str(defaults.pre_sim_duration_s)),
            "rendezvous_duration_s": tk.StringVar(value=str(defaults.rendezvous_duration_s)),
            "inserted_lqr_a_max_km_s2": tk.StringVar(value=str(defaults.inserted_lqr_a_max_km_s2)),
            "output_log_path": tk.StringVar(value=defaults.output_log_path),
        }

    def _build_layout(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(frame, text="Master Simulation Dashboard", font=("Helvetica", 16, "bold"))
        title.pack(anchor=tk.W, pady=(0, 8))

        form = ttk.Frame(frame)
        form.pack(fill=tk.X)

        row = 0
        for label, key in [
            ("Target Altitude (km)", "target_alt_km"),
            ("Target Inclination (deg)", "target_incl_deg"),
            ("Target RAAN (deg)", "target_raan_deg"),
            ("Insertion Delta Altitude (km, below target)", "insertion_delta_alt_km"),
            ("Launch Site Latitude (deg)", "launch_site_lat_deg"),
            ("Launch Site Longitude (deg)", "launch_site_lon_deg"),
            ("Launch Site Altitude (km)", "launch_site_alt_km"),
            ("Launch Timing Mode (go_now/when_feasible/optimal)", "launch_timing_mode"),
            ("Rocket Preset", "rocket_preset"),
            ("Simulation dt (s)", "dt_s"),
            ("Pre-sim Duration (s)", "pre_sim_duration_s"),
            ("Rendezvous Duration (s)", "rendezvous_duration_s"),
            ("Inserted LQR a_max (km/s^2)", "inserted_lqr_a_max_km_s2"),
            ("Output Log Path", "output_log_path"),
        ]:
            ttk.Label(form, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=3)
            ttk.Entry(form, textvariable=self.vars[key], width=42).grid(row=row, column=1, sticky=tk.EW, pady=3)
            row += 1

        form.columnconfigure(1, weight=1)

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X, pady=(10, 0))

        run_btn = ttk.Button(controls, text="Run Master Sim", command=self._on_run)
        run_btn.pack(side=tk.LEFT)

        status = ttk.Label(controls, textvariable=self.status_var)
        status.pack(side=tk.LEFT, padx=(12, 0))

        out_label = ttk.Label(frame, text="Run Output")
        out_label.pack(anchor=tk.W, pady=(12, 4))

        self.output_text = tk.Text(frame, height=20, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.output_text.insert(
            tk.END,
            "Scaffold ready. Adjust parameters and click 'Run Master Sim'.\n"
            "This scaffold currently runs one mission profile and saves a master log.\n",
        )

    def _read_config(self) -> DashboardConfig:
        return DashboardConfig(
            target_alt_km=float(self.vars["target_alt_km"].get()),
            target_incl_deg=float(self.vars["target_incl_deg"].get()),
            target_raan_deg=float(self.vars["target_raan_deg"].get()),
            insertion_delta_alt_km=float(self.vars["insertion_delta_alt_km"].get()),
            launch_site_lat_deg=float(self.vars["launch_site_lat_deg"].get()),
            launch_site_lon_deg=float(self.vars["launch_site_lon_deg"].get()),
            launch_site_alt_km=float(self.vars["launch_site_alt_km"].get()),
            launch_timing_mode=self.vars["launch_timing_mode"].get().strip(),
            rocket_preset=self.vars["rocket_preset"].get().strip(),
            dt_s=float(self.vars["dt_s"].get()),
            pre_sim_duration_s=float(self.vars["pre_sim_duration_s"].get()),
            rendezvous_duration_s=float(self.vars["rendezvous_duration_s"].get()),
            inserted_lqr_a_max_km_s2=float(self.vars["inserted_lqr_a_max_km_s2"].get()),
            output_log_path=self.vars["output_log_path"].get().strip(),
        )

    def _append(self, msg: str) -> None:
        self.output_text.insert(tk.END, msg + "\n")
        self.output_text.see(tk.END)

    def _on_run(self) -> None:
        self.status_var.set("Running...")
        self._append("Starting simulation...")

        def work() -> None:
            try:
                cfg = self._read_config()
                log = run_master_sim_from_config(cfg)
                summary = master_log_summary_text(log)
                self.root.after(0, lambda: self._on_success(cfg.output_log_path_abs.as_posix(), summary))
            except Exception as exc:
                self.root.after(0, lambda: self._on_error(exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_success(self, log_path: str, summary: str) -> None:
        self.status_var.set("Completed")
        self._append(f"Simulation complete. Log saved to: {log_path}")
        self._append(summary)

    def _on_error(self, exc: Exception) -> None:
        self.status_var.set("Failed")
        self._append(f"Error: {type(exc).__name__}: {exc}")


def launch_dashboard() -> None:
    root = tk.Tk()
    MasterSimDashboard(root)
    root.mainloop()
