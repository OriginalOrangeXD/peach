"""Parameter sweep. The LLM does not pick wall thickness; this driver does.

Usage:
  python -m armpipe.sweep armpipe/params.yaml \\
      --path geometry.links.link1.wall_mm --values 2.5,3.0,3.5,4.0
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from armpipe.params import load_params
from armpipe.run import run


def _set_path(tree: dict, dotted: str, value: float) -> None:
    keys = dotted.split(".")
    cur = tree
    for key in keys[:-1]:
        cur = cur[key]
    cur[keys[-1]] = value


def sweep(params_path: Path, dotted: str, values: list[float], out_root: Path) -> Path:
    base = load_params(params_path)
    rows = []
    for value in values:
        params = copy.deepcopy(base)
        _set_path(params, dotted, float(value))
        work = out_root / f"{dotted.replace('.', '_')}={value:g}"
        work.mkdir(parents=True, exist_ok=True)
        tmp_yaml = work / "params.yaml"
        tmp_yaml.write_text(yaml.safe_dump(params, sort_keys=False))
        metrics_path = run(tmp_yaml, work)
        metrics = json.loads(metrics_path.read_text())
        rows.append({"param": dotted, "value": value, **metrics})
    table = out_root / "sweep.json"
    table.write_text(json.dumps(rows, indent=2))
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep one params.yaml key")
    parser.add_argument("params", type=Path)
    parser.add_argument("--path", required=True, help="Dotted key, e.g. geometry.links.link1.wall_mm")
    parser.add_argument("--values", required=True, help="Comma-separated floats")
    parser.add_argument("--out", type=Path, default=Path("work/sweep"))
    args = parser.parse_args(argv)
    values = [float(x) for x in args.values.split(",") if x.strip()]
    path = sweep(args.params, args.path, values, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
