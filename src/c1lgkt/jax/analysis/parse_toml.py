"""
This module contains functions to read and write TOML files for phase space setup.
"""
from __future__ import annotations

from typing import Any
import tomllib


def load_toml(path: str) -> dict[str, Any]:
    """
    Load a TOML file from disk.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def loads_toml(toml_text: str) -> dict[str, Any]:
    """
    Load TOML from a string.
    """
    return tomllib.loads(toml_text)
