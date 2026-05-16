from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if "base" in data:
        base_path = (path.parent / data.pop("base")).resolve()
        base = load_yaml(base_path)
        base.update(data)
        data = base

    return data


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    field_names = {f.name for f in fields(cls)}
    known = {k: v for k, v in data.items() if k in field_names}
    unknown = set(data.keys()) - field_names
    if unknown:
        print(f"warning: ignoring unknown config keys for {cls.__name__}: {unknown}")

    return cls(**known)


def load_config(path: str | Path, model_cls: type, train_cls: type) -> tuple[Any, Any]:
    data = load_yaml(path)
    model_cfg = from_dict(model_cls, data.get("model", {}))
    train_cfg = from_dict(train_cls, data.get("train", {}))
    return model_cfg, train_cfg