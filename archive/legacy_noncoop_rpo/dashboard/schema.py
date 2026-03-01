from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DashboardConfig:
    target_alt_km: float = 350.0
    target_incl_deg: float = 25.0
    target_raan_deg: float = 20.0
    insertion_delta_alt_km: float = 10.0
    launch_site_lat_deg: float = 5.236
    launch_site_lon_deg: float = -52.768
    launch_site_alt_km: float = 0.0
    launch_timing_mode: str = "optimal"
    rocket_preset: str = "demo_heavy_lift"
    dt_s: float = 1.0
    pre_sim_duration_s: float = 6.0 * 3600.0
    rendezvous_duration_s: float = 2.0 * 3600.0
    inserted_lqr_a_max_km_s2: float = 3.0e-5
    output_log_path: str = "outputs/master_sim_log.npz"

    def validate(self) -> None:
        if self.target_alt_km <= 0.0:
            raise ValueError("target_alt_km must be positive.")
        if self.insertion_delta_alt_km < 0.0:
            raise ValueError("insertion_delta_alt_km must be non-negative.")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive.")
        if self.pre_sim_duration_s < 0.0:
            raise ValueError("pre_sim_duration_s must be non-negative.")
        if self.rendezvous_duration_s <= 0.0:
            raise ValueError("rendezvous_duration_s must be positive.")
        if self.inserted_lqr_a_max_km_s2 < 0.0:
            raise ValueError("inserted_lqr_a_max_km_s2 must be non-negative.")

    @property
    def output_log_path_abs(self) -> Path:
        return Path(self.output_log_path).expanduser().resolve()
