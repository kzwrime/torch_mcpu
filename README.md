# PyTorch OpenReg

## 背景

基于 PrivateUse1 的第三方设备集成机制已成为新后端集成到 PyTorch 的官方主流方法。确保该机制的可用性对于丰富 PyTorch 的硬件生态系统至关重要。

**注意：**

`torch_mcpu` 的目标**不是实现一个功能完整、高性能的 PyTorch 后端**，而是作为**机制验证的极简参考实现**。

### 目的

- **测试后端**：作为 PrivateUse1 的树内测试后端，通过 CI/CD 确保质量稳定性。
- **集成示例**：作为新后端集成的参考示例。
- **集成文档**：提供与代码对应的模块级集成文档。

### 设计原则

- **最小化原则**：根本目标是启用/验证新后端集成到 PyTorch 的所有集成路径/机制。所有功能遵循"恰到好处"策略，确保相关集成能力的正确性。
- **真实性原则**：以真实加速器后端集成到 PyTorch 的相同方式完成 OpenReg 集成。

## 目录结构

**依赖关系**:

```mermaid
graph LR
    A[Python]
    B[_C.so]
    C[libtorch_bindings.so]
    D[libtorch_mcpu.so]
    E[libopenreg.so]

    A --> B --> C --> D --> E
```

torch_mcpu 中有 4 个 DSO，它们之间的依赖关系如下：

- `_C.so`:
  - **源文件**: torch_mcpu/csrc/stub.c
  - **描述**: Python C 模块入口点。
- `libtorch_bindings.so`: Python 和 C++ 之间的桥接代码应该放在这里。
  - **源文件**: torch_mcpu/csrc
  - **描述**: Python 和 C++ 之间的薄胶水层。
- `libtorch_mcpu.so`: 所有核心实现应该放在这里。
  - **源文件**: csrc
  - **描述**: 所有核心功能，如设备运行时、运算符等。
- `libopenreg.so`: 一个使用 CPU 模拟类似 CUDA 设备的 DSO，你可以忽略它。
  - **源文件**: third_party/openreg
  - **描述**: 提供类似于 libcudart.so 的底层设备功能。

**关键目录**:

- `csrc/`: 核心设备实现，包括运算符注册、运行时等。
  - `csrc/amp/`: AMP（自动混合精度）
  - `csrc/aten/`: 运算符注册
    - `csrc/aten/native/`: OpenReg 设备的特定运算符实现。
      - `csrc/aten/native/OpenRegMinimal.cpp`: 最小的运算符实现集（完成后允许创建 Tensor 及相关操作）。
      - `csrc/aten/native/OpenRegExtra.cpp`: 其他类型运算符的实现。
  - `csrc/runtime/`: Host 内存、设备内存、Guard、Hooks 等的实现。
- `third_party/`: 一个使用 CPU 模拟类似 CUDA 设备的 C++ 库。
- `torch_mcpu/`: Python 接口实现（Python 代码和 C++ 绑定）。
  - `torch_mcpu/csrc/`: Python C++ 绑定代码。
  - `torch_mcpu/openreg/`: Python API。

## 当前已实现的功能

### 运算符注册

- 运算符实现

  - 为内置 PyTorch 运算符注册
    - `TORCH_LIBRARY_IMPL` 形式：参见 `empty.memory_format
    - `STUB` 形式：参见 `abs_stub`
  - 为自定义运算符注册
    - Schema 注册：参见 `custom_abs`
    - Kernel 注册：参见 `custom_abs`
    - 为 `AutogradPriavateUse1` 注册 Fallback：参见 `custom_abs`
    - Meta 注册：参见 `custom_abs`
    - `torch.autograd.Function`：参见 `custom_autograd_fn_aliasing`
  - 为 fallback 注册
    - 按运算符 Fallback：参见 `sub.Tensor`
    - 全局 Fallback：参见 `wrapper_cpu_fallback`

### 自动加载

当 `import torch` 时，已安装的加速器（如 `torch_mcpu`）将被自动加载，实现与内置后端相同的体验。

- 使用 Python `entry points` 注册后端：参见 `setup.py` 中的 `setup`
- 添加后端初始化的可调用函数：参见 `torch_mcpu/__init__.py` 中的 `_autoload`
- 无需显式导入即可动态加载后端：参见 [使用示例](#使用示例)

### AMP（自动混合精度）

`AMP` 为混合精度提供便捷方法，某些操作使用 `torch.float32` 数据类型，其他操作使用`较低精度`的浮点数据类型：`torch.float16` 或 `torch.bfloat16`。

- 注册特定的运算符转换规则：参见 `csrc/amp` 中的 `autocat_mode.cpp`
- 为不同加速器添加新数据类型的支持：参见 `torch_mcpu/openreg/amp/__init__.py` 中的 `get_amp_supported_dtype`

## 安装和使用

### 安装

```python
python -m pip install --no-build-isolation -e . # 用于开发
python -m pip install --no-build-isolation . # 用于安装
```

### 使用示例

安装后，你可以像使用任何其他常规设备一样在 Python 中使用 `openreg` 设备。

```python
import torch

if not torch.openreg.is_available():
    print("OpenReg backend is not available in this build.")
    exit()

print("OpenReg backend is available!")

device = torch.device("openreg")

x = torch.tensor([[1., 2.], [3., 4.]], device=device)
y = x + 2
print("Result y:\n", y)
print(f"Device of y: {y.device}")

z = y.cpu()
print("Result z:\n", z)
print(f"Device of z: {z.device}")
```

## 文档

请参阅[此文档](https://docs.pytorch.org/docs/main/accelerator/index.html)，了解将新加速器集成到 PyTorch 的一系列文档，这些文档将与 `OpenReg` 代码库保持同步。

## 未来计划

- **增强功能**：
  - 设备无关 API
  - 内存管理
  - 生成器
  - 分布式
  - 自定义 Tensor&Storage
  - ...
- **改进测试**：添加更多与集成机制相关的测试用例。
