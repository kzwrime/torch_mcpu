# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`torch_mcpu` is a minimal reference implementation for integrating custom hardware backends with PyTorch using the PrivateUse1 mechanism. It serves as both a test backend for PyTorch's PrivateUse1 integration and a reference example for new backend implementations.

**Important**: This is NOT a full-featured, high-performance backend. It's a minimalist implementation focused on validating integration mechanisms.

## Build and Development Commands

### Building the Project

```bash
# Development installation (editable)
python -m pip install --no-build-isolation -e .

# Regular installation
python -m pip install --no-build-isolation .

# Clean build artifacts
python setup.py clean
```

**Critical**: Always use `--no-build-isolation` flag to avoid issues with PyTorch detection during setup.

### Running Tests

```bash
# Run all tests from outside the source directory (to avoid Python path issues)
cd /tmp && python -m pytest /shared/torch-dev-kit/torch_mcpu/tests/ -v

# Run specific test file
cd /tmp && python -m pytest /shared/torch-dev-kit/torch_mcpu/tests/test_device.py -v

# Run specific test case
cd /tmp && python -m pytest /shared/torch-dev-kit/torch_mcpu/tests/test_device.py::TestDevice::test_device_count -v
```

**Important**: Always run tests from `/tmp` or another directory outside the source tree to prevent Python from importing the local source instead of the installed package.

### Testing Individual Components

```bash
# Test device operations
cd /tmp && python -c "
import torch
device = torch.device('mcpu')
x = torch.randn(3, 3, device=device)
print(f'Created tensor on {x.device}')
y = x + 1
print(f'Operation result: {y}')
"

# Test custom operators
cd /tmp && python -c "
import torch
x = torch.randn(2, 3, device='mcpu')
y = torch.ops.mcpu.custom_abs(x)
print(f'custom_abs result: {y}')
"
```

## Architecture

### DSO Dependency Chain

The project uses a 4-layer shared object (DSO) architecture:

1. **`_C.so`** (torch_mcpu/csrc/stub.c)
   - Python C module entry point
   - Minimal stub that delegates to libtorch_bindings

2. **`libtorch_bindings.so`** (torch_mcpu/csrc)
   - Thin Python-C++ glue layer
   - Contains Module.cpp with Python-exposed functions

3. **`libtorch_mcpu.so`** (csrc/)
   - Core backend implementation
   - Contains runtime, ATEN operators, device management

4. **`libmcpu.so`** (third_party/openreg/)
   - CPU-emulated CUDA-like device
   - Provides low-level device functionality (like libcudart.so)
   - **Do not modify** - this is internal implementation

### Key Directories and Their Roles

**`csrc/runtime/`** - Core device runtime
- `OpenRegFunctions.{h,cpp}` - Device management functions (set_device, device_count, etc.)
- `OpenRegStream.{h,cpp}` - Stream management for async operations
- `OpenRegEvent.{h,cpp}` - Event synchronization
- `OpenRegGuard.{h,cpp}` - Device guard RAII wrapper
- `OpenRegGenerator.{h,cpp}` - Random number generation
- `OpenRegDeviceAllocator.{h,cpp}` - Device memory allocation
- `OpenRegHostAllocator.{h,cpp}` - Pinned host memory allocation
- `OpenRegHooks.{h,cpp}` - PyTorch hooks interface
- `OpenRegSerialization.{h,cpp}` - Serialization/deserialization support

**`csrc/aten/`** - ATEN operator registration
- `OpenRegMinimal.cpp` - Minimal operator implementations (empty, tensor creation)
- `OpenRegExtra.cpp` - Extended operators (custom_abs, quantization, SDPA, autograd)
- Uses `TORCH_LIBRARY(mcpu, ...)` and `TORCH_LIBRARY_IMPL(mcpu, ...)` macros

**`csrc/aten/native/`** - Native operator implementations
- `Minimal.cpp` - Core native operations
- `Extra.cpp` - Additional native operations
- All in `at::native::mcpu` namespace

**`torch_mcpu/openreg/`** - Python API layer
- `__init__.py` - Main Python API (device context, device_count, set_device, etc.)
- `meta.py` - Custom operator meta implementations
- `random.py` - RNG state management
- `amp/` - Automatic mixed precision support

**`third_party/openreg/`** - Device emulation library
- **Never modify** this directory - it's internal implementation
- C++ library using CPU to emulate CUDA-like behavior
- Has its own build system independent of the main project

### C++ Namespace Structure

The codebase uses `c10::mcpu` namespace for core runtime functionality:

```cpp
namespace c10::mcpu {
    int device_count();
    int current_device();
    void set_device(int index);
    int ExchangeDevice(int index);
}
```

ATEN operations use `at::native::mcpu` namespace:

```cpp
namespace at::native::mcpu {
    at::Tensor empty_symint(IntArrayRef size, c10::optional<c10::MemoryFormat> format);
    at::Tensor custom_abs(const at::Tensor& x);
}
```

### Backend Registration Flow

1. **Python-level** (`torch_mcpu/__init__.py`):
   ```python
   torch.utils.rename_privateuse1_backend("mcpu")
   torch._register_device_module("mcpu", torch_mcpu.openreg)
   ```

2. **C++ Operator Registration** (`csrc/aten/OpenRegExtra.cpp`):
   ```cpp
   TORCH_LIBRARY(mcpu, m) {
       m.def("custom_abs(Tensor input) -> Tensor");
   }
   TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
       m.impl("custom_abs", &wrapper_custom_abs);
   }
   ```

3. **Entry Point** (`setup.py`):
   ```python
   entry_points={
       "torch.backends": [
           "torch_mcpu = torch_mcpu:_autoload",
       ],
   }
   ```

## Important Development Notes

### File Naming Convention

Despite the namespace rename to `mcpu`, **source files still use `OpenReg*` naming**:
- Files: `OpenRegStream.h`, `OpenRegFunctions.cpp`, etc.
- Namespaces: `c10::mcpu`, `at::native::mcpu`
- Classes: `McpuStream`, `McpuFunctions`, etc.

This separation is intentional - the public API uses `mcpu` but file names retain `OpenReg` for compatibility.

### Python Path Issues

**Critical**: When testing or running code from the source directory, Python may import the local `torch_mcpu` directory instead of the installed package. This causes import errors because the local source lacks compiled C++ extensions.

**Solution**: Always run tests from `/tmp` or another directory outside the source tree:
```bash
cd /tmp && python -m pytest /path/to/torch_mcpu/tests/
```

### CMake RPATH Configuration

The project uses specific RPATH settings to ensure proper library loading:
```cmake
# Linux
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib:$ORIGIN:${PYTORCH_INSTALL_DIR}/lib")
```

This ensures that `libtorch_mcpu.so` can find PyTorch libraries at runtime.

### Working with PrivateUse1

This backend uses PyTorch's PrivateUse1 device type, which is a generic mechanism for custom hardware:
- Device type: `c10::DeviceType::PrivateUse1`
- Python name: "mcpu" (registered via `rename_privateuse1_backend`)
- Device strings: `"mcpu"`, `"mcpu:0"`, `"mcpu:1"`

### Modifying Operator Implementations

When adding or modifying operators:

1. **Schema Registration** (if creating new op):
   ```cpp
   TORCH_LIBRARY(mcpu, m) {
       m.def("my_op(Tensor input) -> Tensor");
   }
   ```

2. **Kernel Implementation**:
   ```cpp
   TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
       m.impl("my_op", &my_op_impl);
   }
   ```

3. **Meta Implementation** (torch_mcpu/openreg/meta.py):
   ```python
   lib = torch.library.Library("mcpu", "IMPL", "Meta")
   @torch.library.impl(lib, "my_op")
   def my_op_meta(x):
       return torch.empty_like(x)
   ```

### LITERALINCLUDE Markers

The codebase contains `LITERALINCLUDE` markers that identify code sections intended for documentation:
```cpp
// LITERALINCLUDE MCPU GET DEFAULT GENERATOR
static PyObject* _getDefaultGenerator(...) { ... }
// LITERALINCLUDE MCPU GET DEFAULT GENERATOR
```

These markers are used by PyTorch documentation to extract code examples. When modifying code with these markers, maintain the marker structure if the code is intended for documentation.

## Testing Strategy

### Test Coverage

The test suite covers:
- **Device operations**: test_device.py
- **Memory management**: test_memory.py (allocation, pinning, cross-device)
- **Operators**: test_ops.py (custom ops, fallback, quantization)
- **Streams and Events**: test_streams.py, test_event.py
- **Autograd**: test_autograd.py
- **AMP**: test_autocast.py
- **Profiling**: test_profiler.py
- **Serialization**: test_storage.py
- **RNG**: test_rng.py
- **Utils**: test_utils.py

### Running Tests Before Committing

Always run the full test suite before committing changes:
```bash
cd /tmp && python -m pytest /shared/torch-dev-kit/torch_mcpu/tests/ -v
```

Expected results: 179 passed, 9-10 skipped (some tests are conditionally skipped).

## Common Issues and Solutions

### Build Failures

**Issue**: `error: MCPU_EXPORT was not declared`
**Solution**: Clean build completely: `rm -rf build/ && python setup.py clean && python -m pip install --no-build-isolation .`

**Issue**: Import errors about `torch_mcpu._C`
**Solution**: Run from `/tmp` directory, not from source directory

**Issue**: Link errors about PyTorch symbols
**Solution**: Verify RPATH settings in CMakeLists.txt include `${PYTORCH_INSTALL_DIR}/lib`

### Test Failures After Changes

**Issue**: Tests fail after modifying operator registration
**Solution**: Ensure you've updated all three layers:
1. C++ schema/kernel registration
2. Python meta implementation
3. Device string in tests (`device="mcpu"`)

### Naming Conflicts

**Issue**: References to both "openreg" and "mcpu" after partial rename
**Solution**: Search for remaining "openreg" references (excluding third_party/):
```bash
grep -r "openreg" csrc/ torch_mcpu/ --exclude-dir=third_party
```