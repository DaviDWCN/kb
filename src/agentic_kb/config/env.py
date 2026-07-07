"""`.env` loading for local CLI runs."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> int:
    """Load environment variables from a dotenv-style file.

    Existing environment variables are preserved by default so shell-provided
    secrets win over local `.env` defaults.
    """

    env_path = Path(path)
    if not env_path.exists():
        return 0

    loaded_count = 0
    for key, value in dotenv_values(env_path).items():
        if value is None:
            continue
        if key in os.environ and not override:
            continue

        os.environ[key] = value
        loaded_count += 1
    return loaded_count
