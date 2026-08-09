"""Cycle-safe conversion of runtime metadata to JSON-compatible values.

``dataclasses.asdict`` recursively deep-copies every nested value.  That is
unnecessarily expensive for debug metadata and, more importantly, it does not
protect against a dictionary that refers to itself.  LaMa retry diagnostics
used to create exactly such a cycle when the retry result was selected.

The helpers in this module walk the active recursion stack explicitly.  A
repeated value in two independent branches is serialized normally; only a real
cycle is replaced by a short marker.  This keeps diagnostics writable without
ever descending into model/runtime objects.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping


_RECURSIVE_REFERENCE = "<recursive-reference>"
_MAX_DEPTH_MARKER = "<maximum-metadata-depth>"


def to_json_safe(
    value: Any,
    *,
    max_depth: int = 64,
    _active: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    """Return a JSON-compatible copy without recursive deep-copy semantics."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if _depth >= max(1, int(max_depth)):
        return _MAX_DEPTH_MARKER

    # NumPy scalar values expose ``item`` but are not JSON primitives.
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (Mapping, list, tuple, set)):
        try:
            converted = item()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return to_json_safe(
                converted,
                max_depth=max_depth,
                _active=_active,
                _depth=_depth + 1,
            )

    active = _active if _active is not None else set()
    identity = id(value)
    if identity in active:
        return _RECURSIVE_REFERENCE

    active.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: to_json_safe(
                    getattr(value, field.name),
                    max_depth=max_depth,
                    _active=active,
                    _depth=_depth + 1,
                )
                for field in fields(value)
            }
        if isinstance(value, Mapping):
            return {
                str(
                    to_json_safe(
                        key,
                        max_depth=max_depth,
                        _active=active,
                        _depth=_depth + 1,
                    )
                ): to_json_safe(
                    child,
                    max_depth=max_depth,
                    _active=active,
                    _depth=_depth + 1,
                )
                for key, child in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [
                to_json_safe(
                    child,
                    max_depth=max_depth,
                    _active=active,
                    _depth=_depth + 1,
                )
                for child in value
            ]

        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return to_json_safe(
                    tolist(),
                    max_depth=max_depth,
                    _active=active,
                    _depth=_depth + 1,
                )
            except (TypeError, ValueError):
                pass
        return str(value)
    finally:
        active.discard(identity)

