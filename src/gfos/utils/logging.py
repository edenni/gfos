from collections.abc import MutableMapping
from typing import Any, Dict


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "/"
) -> Dict[str, Any]:
    """
    Flattens a nested dictionary by concatenating keys with a separator.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
