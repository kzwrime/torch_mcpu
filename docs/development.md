# Development Guide

## Prerequisites

- Python >= 3.9
- PyTorch >= 2.9 (`pip install torch --index-url https://download.pytorch.org/whl/cpu`)
- GCC >= 11 or Clang >= 14 with C++17 support
- `pip install pytest`

## Build

```bash
# Standard editable install (recommended for development)
pip install -e . --no-build-isolation

# Debug build (O0, assertions enabled)
TORCH_MCPU_DEBUG=1 pip install -e . --no-build-isolation

# Clean rebuild
pip uninstall torch_mcpu -y && pip install -e . --no-build-isolation
```

The build compiles all `torch_mcpu/csrc/*.cpp` files into a single
`torch_mcpu._C` extension module.

## Run Tests

```bash
# All tests
pytest test/ -v

# Single file
pytest test/test_basics.py -v

# With fallback op warning enabled
MCPU_FALLBACK_WARN=1 pytest test/ -v
```

## Environment Variables

| Variable | Effect |
|---|---|
| `TORCH_MCPU_DEBUG=1` | Compile with `-O0 -g` |
| `MCPU_FALLBACK_WARN=1` | Print a warning each time an op hits the CPU fallback |
| `TORCH_SHOW_DISPATCH_TRACE=1` | PyTorch native: show full dispatch trace |

## Adding a New Operator

1. Add the kernel function to `torch_mcpu/csrc/Ops.cpp`:

```cpp
static at::Tensor mcpu_my_op(const at::Tensor& self) {
    // MCPU data is CPU-accessible; delegate to native if possible
    MemoryGuard guard(self);
    return at::cpu::my_op(self);
}
```

2. Register it in the `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` block
   at the bottom of `Ops.cpp`:

```cpp
m.impl("my_op", &mcpu_my_op);
```

3. Add a test case to `test/test_ops.py`.

## Code Style

- C++17, following PyTorch ATen conventions.
- `TORCH_CHECK` for user-facing errors.
- `TORCH_INTERNAL_ASSERT` for invariants that should never be violated.
- No raw `new` / `delete`; use `at::DataPtr` and RAII.
- Python: PEP 8, no type annotations on internal helpers.

## Repository Layout

```
torch_mcpu/
├── docs/               Architecture, roadmap, this file
├── torch_mcpu/
│   ├── csrc/           All C++ source (compiled into _C.so)
│   └── *.py            Python package files
├── test/               pytest test suite
├── .github/workflows/  CI configuration
├── setup.py            Build entry point
└── pyproject.toml      PEP 517 metadata
```

## Release Process

1. Bump `version` in `pyproject.toml` and `setup.py`.
2. Tag with `git tag v<version>`.
3. CI builds and publishes to the internal package index.
