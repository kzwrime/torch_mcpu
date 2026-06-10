import json
from pathlib import Path
from typing import Any, Dict, List


_COMPILE_FLAGS_FILE = Path(__file__).resolve().parent / "_compile_flags.json"


def get_compile_config() -> Dict[str, Any]:
    """Return the compile-time configuration recorded when torch_mcpu was built."""
    with _COMPILE_FLAGS_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def get_compile_definitions() -> Dict[str, str]:
    """Return preprocessor definitions that external extensions should mirror."""
    return dict(get_compile_config()["definitions"])


def get_compile_flags() -> List[str]:
    """Return `-D...=...` flags matching this installed torch_mcpu build."""
    return list(get_compile_config()["compile_flags"])
