from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.models import SimLog


@dataclass(frozen=True)
class EngagementMetrics:
    min_separation_km: float
    time_inside_keepout_s: float
    fuel_used_kg_by_object: dict[str, float]
    compute_overruns_by_object: dict[str, int]
    jitter_ms_by_object: dict[str, float]


def compute_engagement_metrics(log: SimLog, keepout_radius_km: float | None = None) -> EngagementMetrics:
    object_ids = sorted(log.truth_by_object.keys())
    t = log.t_s
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0

    min_sep = np.inf
    time_inside = 0.0
    for i, oid_i in enumerate(object_ids):
        ri = log.truth_by_object[oid_i][:, :3]
        if keepout_radius_km is not None:
            r_norm = np.linalg.norm(ri, axis=1)
            time_inside += float(np.sum(r_norm < keepout_radius_km) * dt)
        for oid_j in object_ids[i + 1 :]:
            rj = log.truth_by_object[oid_j][:, :3]
            d = np.linalg.norm(ri - rj, axis=1)
            min_sep = min(min_sep, float(np.min(d)))
    if np.isinf(min_sep):
        min_sep = 0.0

    fuel_used = {}
    overruns = {}
    jitter = {}
    for oid in object_ids:
        mass = log.truth_by_object[oid][:, 13]
        fuel_used[oid] = float(max(0.0, mass[0] - mass[-1]))
        skip = log.controller_skipped_by_object[oid]
        rt = log.controller_runtime_ms_by_object[oid]
        overruns[oid] = int(np.sum(skip))
        jitter[oid] = float(np.std(rt[1:])) if rt.size > 1 else 0.0

    return EngagementMetrics(
        min_separation_km=min_sep,
        time_inside_keepout_s=time_inside,
        fuel_used_kg_by_object=fuel_used,
        compute_overruns_by_object=overruns,
        jitter_ms_by_object=jitter,
    )
