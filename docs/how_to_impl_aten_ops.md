# How To Implement Aten Ops In `torch_mcpu`

这份文档总结了在 `torch_mcpu` 里新增显式 `aten` 算子的推荐流程，目标是：

- 尽量复用 PyTorch 现成接口，不重复造 schema / wrapper。
- 对 `mcpu` Tensor 走“预分配 mcpu Tensor + CPU view 映射”的无额外复制路径。
- 让后续开发者知道去哪里找上游签名、注册名和 `out` 版本。

## 1. 先确认上游 schema / 注册名

优先看这几个文件：

- `pytorch/build/aten/src/ATen/RegisterSchema.cpp`
  - 查 schema 的正式名字，例如 `index.Tensor` / `index.Tensor_out`。
- `pytorch/build/aten/src/ATen/RegisterCUDAEverything.cpp`
  - 看 CUDA 是怎么注册的，能快速知道应该实现哪些 overload。
- `pytorch/build/aten/src/ATen/RegisterCPUEverything.cpp`
  - 看 CPU 侧对应注册。
- `pytorch/torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.cpp`
  - 看上游 CPU 接口是怎么调的，适合确认最终调用哪个 `at::xxx` / `at::xxx_out`。

常用搜索命令（pytorch 源码通常放在 torch_mcpu 同级）:

```bash
rg -n "index.Tensor|index.Tensor_out|cumsum.out|arange.start_out|fill_.Scalar" \
  ../pytorch/build/aten/src/ATen \
  ../pytorch/torch/csrc/inductor/aoti_torch/generated
```

## 2. 优先找 `out` / inplace 接口

如果算子有返回值，优先改写成：

- 先用 meta 推导输出形状/stride
- 再预分配 `mcpu` 输出
- 最后把 `mcpu` 输出映射成 CPU view，调用 CPU 的 `out` 版本

原因：

- 返回 Tensor 时，直接调 functional 接口通常会在 CPU 上新分配结果，不符合我们“先分配 mcpu Tensor”的要求。
- `out` / inplace 接口更容易做到不额外 copy。

上游接口声明一般在：

- `pytorch/build/aten/src/ATen/ops/*.h`
- `pytorch/build/aten/src/ATen/ops/*_native.h`
- `pytorch/build/aten/src/ATen/ops/*_cpu_dispatch.h`

例如：

- `ATen/ops/arange.h`
- `ATen/ops/cumsum.h`
- `ATen/ops/index.h`
- `ATen/ops/fill.h`

## 3. 在本仓库应该放哪里

显式 `aten` 算子现在推荐放在：

- `csrc/aten/ops/`

规则：

- 一个算子一个 `.cpp`
- 同一类型的多个 schema 可以放一起
  - 例如 `index.Tensor` 和 `index.Tensor_out` 放在同一个文件
- 算子实现和 `TORCH_LIBRARY_IMPL` 注册都放在该文件内
- 不再为这类显式 op 额外写 `Extra.h` 声明或 `wrapper_xxx`

当前示例：

- `csrc/aten/ops/arange.cpp`
- `csrc/aten/ops/cumsum.cpp`
- `csrc/aten/ops/fill.cpp`
- `csrc/aten/ops/index.cpp`

公共辅助函数放在：

- `csrc/aten/ops/Common.h`

## 4. 推荐实现模式

### 4.1 工厂类算子

典型如 `arange`。

做法：

1. 检查 `layout/device/pin_memory`
2. 用 meta 设备调用一次 `out` 接口，得到输出 shape/stride
3. 用 `empty_strided` 分配 `mcpu` Tensor
4. 用 CPU view 映射 `mcpu` Tensor
5. 调 CPU `out` 接口真正写结果

### 4.2 普通返回 Tensor 的算子

典型如 `cumsum`、`index`。

做法：

1. 把输入 `mcpu` Tensor 先转成 meta Tensor，只保留元信息
2. 用 meta `out` 接口推导结果 shape/stride
3. 分配 `mcpu` 输出
4. 把输入/输出映射成 CPU view
5. 调 CPU `out` 接口

### 4.3 inplace 算子

典型如 `fill_`。

做法：

1. 对输入 `self` 建 CPU view
2. 直接调用 CPU inplace 接口
3. 返回原 `self`

## 5. 不要做什么

- 不要新写一层 `wrapper_xxx` 再转发。
- 不要把这些显式 op 再塞回 `csrc/aten/native/Extra.cpp`。
- 不要在 `out` 变体里偷偷 `resize_`。

现在的约定是：

- `out` Tensor shape 不对，直接报错
- 让调用方自己提前准备正确大小的输出

## 6. 为什么需要 meta 推导

`mcpu -> cpu view` 只是共享 data_ptr 的新 Tensor 元数据。

如果先建一个空的 `mcpu out`，再把它映射为 CPU view，然后依赖 CPU `out` 内部自动 resize：

- CPU view 的 size 改了
- 但原始 `mcpu out` 的元数据不会同步变化

结果就是 data 写进去了，但 `mcpu` Tensor 仍然保留错误 shape。

所以必须先在 meta 上推导出正确 shape/stride，再分配真正的 `mcpu` 输出。

## 7. 运行时 fallback 打印

全局 fallback 打印在：

- `csrc/aten/native/MCPUFallback.cpp`

当前行为：

- 每次触发 boxed fallback，都打印一次
- 显式实现的 op 不会再打印

输出格式：

```text
[mcpu fallback] aten::add.out
```

这能帮助排查实际业务里哪些算子仍然走 fallback。

## 8. 改完后如何验证

先编译：

```bash
python setup.py build_ext --inplace
```

再跑定向测试：

```bash
python - <<'PY'
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch_mcpu
import pytest, sys
sys.exit(pytest.main(['tests/test_ops.py', '-k', 'Fallback', '-q']))
PY
```

如果要快速做烟测，可以直接写：

```python
import torch
import torch_mcpu

x = torch.tensor([[1, 2, 3], [4, 5, 6]], device="mcpu", dtype=torch.float32)
idx = torch.tensor([1, 0], device="mcpu", dtype=torch.long)

print(torch.arange(1, 7, 2, device="mcpu"))
print(torch.cumsum(x, dim=1))
print(x[idx, idx])
```

## 9. 简化决策准则

新增一个 `aten` op 时，优先按下面顺序判断：

1. 能不能直接显式注册到 `csrc/aten/ops/<op>.cpp`
2. 有没有 CPU `out` / inplace 版本可复用
3. 能不能用 meta 先推导输出，再分配 `mcpu` 输出
4. 输入里是否只需要做 view 映射，而不是 copy
5. `out` 尺寸不匹配时是否已经明确报错

满足这几条，基本就是本仓库当前推荐实现方式。
