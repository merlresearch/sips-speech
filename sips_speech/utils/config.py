# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Utility functions for reading configuration files."""

import importlib
import os
import types
from collections.abc import Callable
from typing import Any, Mapping, Union

_MISSING = object()


def read_config(config_path):
    if os.fspath(config_path).endswith((".yaml", ".yml")):
        import yaml

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    raise ValueError(f"Unsupported config format: {config_path}")


class EasyDict(dict):
    """Dictionary with attribute-style access.

    Examples:
        d = EasyDict(a=1)
        d.a == 1

        d.b = 2
        d["b"] == 2
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        data = dict(*args, **kwargs)
        for key, value in data.items():
            self[key] = self._convert(value)

    @staticmethod
    def _convert(value: Any) -> Any:
        """Recursively convert nested dicts into EasyDict."""
        if isinstance(value, dict) and not isinstance(value, EasyDict):
            return EasyDict(value)

        if isinstance(value, list):
            return [EasyDict._convert(v) for v in value]

        if isinstance(value, tuple):
            return tuple(EasyDict._convert(v) for v in value)

        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._convert(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute '{name}'") from exc

    def copy(self) -> "EasyDict":
        return EasyDict(super().copy())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "EasyDict":
        return cls(mapping)

    def to_dict(self) -> dict:
        """Recursively convert EasyDict back to a plain dict."""

        def convert(obj):
            if isinstance(obj, EasyDict):
                return {k: convert(v) for k, v in obj.items()}

            if isinstance(obj, list):
                return [convert(v) for v in obj]

            if isinstance(obj, tuple):
                return tuple(convert(v) for v in obj)

            return obj

        return convert(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict.__repr__(self)})"


def get_module_from_obj_name(name: str) -> tuple[types.ModuleType, str]:
    """Split a fully qualified object name into its module and object path.

    Examples:
        "package.module.ClassName" -> (package.module, "ClassName")
        "package.module.outer.inner" -> (package.module, "outer.inner")
        "math" -> (math, "")
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Object name must be a non-empty string.")

    parts = name.split(".")

    last_error: Exception | None = None

    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        obj_name = ".".join(parts[i:])

        try:
            module = importlib.import_module(module_name)
            return module, obj_name
        except ImportError as exc:
            last_error = exc

    raise ImportError(f"Could not import any module from object name: {name}") from last_error


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverse an object path and return the rightmost Python object.

    Examples:
        get_obj_from_module(math, "sqrt") -> math.sqrt
        get_obj_from_module(pkg.module, "Outer.Inner") -> pkg.module.Outer.Inner
    """
    if obj_name == "":
        return module

    obj: Any = module

    for part in obj_name.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise AttributeError(
                f"Object path '{obj_name}' could not be resolved: missing attribute '{part}' on {obj!r}"
            ) from exc

    return obj


def get_obj(name: str) -> Any:
    """Find the Python object with the given fully qualified name."""
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func(
    func_name: Union[str, Callable[..., Any], object] = _MISSING,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Find a Python object by name and call it as a function."""
    if func_name is _MISSING:
        if "func_name" not in kwargs:
            raise ValueError("func_name must not be None.")

        func_name = kwargs.pop("func_name")

    if func_name is None:
        raise ValueError("func_name must not be None.")

    func_obj = get_obj(func_name) if isinstance(func_name, str) else func_name

    if not callable(func_obj):
        raise TypeError(f"Object is not callable: {func_obj!r}")

    return func_obj(*args, **kwargs)


def construct_class(
    class_name: Union[str, type, object] = _MISSING,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Find a Python class by name and construct it with the given arguments."""
    if class_name is _MISSING:
        if "class_name" not in kwargs:
            raise ValueError("class_name must not be None.")

        class_name = kwargs.pop("class_name")

    if class_name is None:
        raise ValueError("class_name must not be None.")

    return call_func(class_name, *args, **kwargs)
