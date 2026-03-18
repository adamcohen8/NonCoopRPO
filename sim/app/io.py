from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from sim.config.scenario_yaml import SimulationScenarioConfig, load_simulation_yaml, scenario_config_from_dict

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for application I/O helpers.") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "simulation_template.yaml"


def list_config_files(config_dir: Path = CONFIG_DIR) -> list[Path]:
    return sorted(config_dir.glob("*.yaml"))


def read_yaml_file(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def parse_yaml_text(yaml_text: str) -> dict[str, Any]:
    raw = yaml.safe_load(yaml_text)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping/object.")
    return dict(raw)


def dump_yaml_text(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)


def validate_config_dict(data: dict[str, Any]) -> SimulationScenarioConfig:
    return scenario_config_from_dict(data)


def validate_yaml_text(yaml_text: str) -> SimulationScenarioConfig:
    return validate_config_dict(parse_yaml_text(yaml_text))


def config_to_dict(cfg: SimulationScenarioConfig) -> dict[str, Any]:
    return asdict(cfg)


def load_config_dict(path: str | Path) -> dict[str, Any]:
    return config_to_dict(load_simulation_yaml(path))


def ensure_sections(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.setdefault("metadata", {})
    out.setdefault("rocket", {})
    out.setdefault("chaser", {})
    out.setdefault("target", {})
    out.setdefault("simulator", {})
    out.setdefault("outputs", {})
    out.setdefault("monte_carlo", {})
    out["simulator"].setdefault("dynamics", {})
    out["outputs"].setdefault("stats", {})
    out["outputs"].setdefault("plots", {})
    out["outputs"].setdefault("animations", {})
    out["outputs"].setdefault("monte_carlo", {})
    return out


def save_config_dict(path: str | Path, data: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_yaml_text(data), encoding="utf-8")
    return target


def write_temp_config(data: dict[str, Any]) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        tf.write(dump_yaml_text(data))
        return Path(tf.name)


def build_run_command(config_path: str | Path) -> list[str]:
    return [sys.executable, str(REPO_ROOT / "run_simulation.py"), "--config", str(Path(config_path))]


def run_simulation_cli(config_path: str | Path) -> dict[str, Any]:
    cmd = build_run_command(config_path)
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_s = time.perf_counter() - t0
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_s": float(elapsed_s),
    }


def list_output_files(output_dir: str | Path, limit: int = 200) -> list[Path]:
    outdir = Path(output_dir)
    if not outdir.exists() or not outdir.is_dir():
        return []
    files = sorted((p for p in outdir.rglob("*") if p.is_file()), key=lambda p: str(p))
    return files[: max(int(limit), 0)]
