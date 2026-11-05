from copy import deepcopy
from pathlib import Path


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_config(path):
    import yaml
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = cfg.pop("base", None)
    return merge(load_config(path.parent / parent), cfg) if parent else cfg

