from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.core.models import SimLog


@dataclass(frozen=True)
class ScoreSummary:
    min_separation_km: float
    controller_overruns: dict[str, int]
    mean_runtime_ms: dict[str, float]
    p95_runtime_ms: dict[str, float]
    jitter_ms: dict[str, float]


def compute_scores(log: SimLog) -> ScoreSummary:
    object_ids = sorted(log.truth_by_object.keys())

    min_sep = np.inf
    for i, oid_i in enumerate(object_ids):
        ri = log.truth_by_object[oid_i][:, :3]
        for oid_j in object_ids[i + 1 :]:
            rj = log.truth_by_object[oid_j][:, :3]
            dist = np.linalg.norm(ri - rj, axis=1)
            min_sep = min(min_sep, float(np.min(dist)))
    if np.isinf(min_sep):
        min_sep = 0.0

    overruns = {}
    mean_runtime = {}
    p95_runtime = {}
    jitter = {}
    for oid in object_ids:
        rt = log.controller_runtime_ms_by_object[oid][1:]
        skip = log.controller_skipped_by_object[oid][1:]
        overruns[oid] = int(np.sum(skip))
        mean_runtime[oid] = float(np.mean(rt))
        p95_runtime[oid] = float(np.percentile(rt, 95.0))
        jitter[oid] = float(np.std(rt))

    return ScoreSummary(
        min_separation_km=min_sep,
        controller_overruns=overruns,
        mean_runtime_ms=mean_runtime,
        p95_runtime_ms=p95_runtime,
        jitter_ms=jitter,
    )
