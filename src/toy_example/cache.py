"""Save/load stage results as .npz arrays with a JSON metadata sidecar."""

import json
from pathlib import Path

import numpy as np


def save(path: Path, arrays: dict[str, np.ndarray], meta: dict, compress: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    (np.savez_compressed if compress else np.savez)(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    return path


def load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the corresponding build script first.")
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    return arrays, meta
