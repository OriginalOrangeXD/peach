"""Load ``params.yaml``. This file is the only numeric source of truth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from armpipe.units import (
    Material,
    density_kg_m3,
    poisson,
    youngs_Pa,
)


class ParamsError(ValueError):
    """Missing or malformed key in params.yaml."""


def load_params(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ParamsError(f"{path} must contain a mapping, got {type(data).__name__}")
    return data


def require(params: dict, *keys: str) -> Any:
    cur: Any = params
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(cur, dict) or key not in cur:
            raise ParamsError(f"missing params key: {'.'.join(walked)}")
        cur = cur[key]
    return cur


def material_from_params(params: dict) -> Material:
    """Material block is stored in SI (kg/m³, Pa). Convert at the FEA boundary."""
    block = require(params, "material")
    return Material(
        name=str(block.get("name", "unnamed")),
        density=density_kg_m3(float(block["density_kg_m3"])),
        E=youngs_Pa(float(block["E_Pa"])),
        nu=poisson(float(block["nu"])),
    )
