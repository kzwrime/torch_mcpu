from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_SOURCE_ROOT = _PACKAGE_DIR.parent
_STAGING_PACKAGE_DIR = _SOURCE_ROOT / "build" / "cmake_install" / "torch_mcpu"


def _candidate_roots() -> tuple[Path, ...]:
    if (_SOURCE_ROOT / "setup.py").is_file():
        return (_STAGING_PACKAGE_DIR, _PACKAGE_DIR)
    return (_PACKAGE_DIR, _STAGING_PACKAGE_DIR)


def _find_subdir(name: str, markers: tuple[str, ...]) -> Path:
    for root in _candidate_roots():
        candidate = root / name
        if candidate.is_dir() and any((candidate / marker).exists() for marker in markers):
            return candidate
    raise FileNotFoundError(
        f"torch_mcpu {name!r} directory was not found. "
        "Reinstall torch_mcpu to regenerate the CMake install tree."
    )


def get_include() -> str:
    """Return the directory external C++ extensions should add to includes."""
    return str(_find_subdir("include", ("runtime/McpuKernelLaunch.h",)))


def get_library_dir() -> str:
    """Return the directory external C++ extensions should add to library paths."""
    return str(
        _find_subdir(
            "lib",
            (
                "libtorch_mcpu.so",
                "libtorch_mcpu.dylib",
                "torch_mcpu.dll",
                "torch_mcpu.lib",
                "libtorch_mcpu.lib",
            ),
        )
    )
