# How To Implement Aten Ops In `torch_mcpu`

这份文档总结了在 `torch_mcpu` 里新增显式 `aten` 算子的推荐流程，目标是：

- 尽量复用 PyTorch 现成接口，不重复造 schema / wrapper。
- 对 `mcpu` Tensor 走“预分配 mcpu Tensor + CPU view 映射”的无额外复制路径。
- 避免在异步 kernel lambda 里捕获 Tensor；优先捕获指针和元信息，在 kernel 内
  重建临时 CPU view。
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
- `csrc/aten/ops/where.cpp`

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
4. launch 前抽取输入/输出的 `data_ptr`、shape、stride、options 等元信息
5. 调 CPU `out` 接口

默认不要在 `MCPU_LAUNCH_TIMED_KERNEL` 的 lambda 捕获列表里捕获 `at::Tensor`。
如果 kernel 内需要调用 ATen CPU `out` 接口，可以在 launch 前保存指针和元信息，
在 kernel 内用 `at::from_blob` / `at::empty_strided` 重建临时 CPU view。
这样不会因为异步任务持有 Tensor 而延长 Tensor / storage 生命周期。

### 4.2.1 elementwise binary 算子的例外

典型如 `add`、`sub`。

不要为了推导 shape 在 `mcpu` kernel 内调用 `at::add_out(meta_out, ...)`
或 `at::sub_out(meta_out, ...)`。当前 PyTorch 里这类 elementwise binary
Meta kernel 可能走 Python 注册路径，profiler 会看到
`torch/_prims_common/wrappers.py`、`torch/_refs/__init__.py` 和
`torch/_dynamo/eval_frame.py` 等栈，并且实际引入明显额外开销。

这类算子应优先直接用 C++ 元信息工具推导：

- 用 `at::infer_size(self.sizes(), other.sizes())` 做 broadcast shape 检查。
- functional 版本用 `at::result_type(self, other)` / `at::result_type(self, scalar)`
  推导输出 dtype 后分配 `mcpu` out。
- `out` / inplace 版本只检查预期 shape，然后直接映射 CPU view 并调用 CPU `out`
  或 inplace 计算。

同时要显式注册业务会直接触发的 schema，例如 `add.Tensor`、`add_.Tensor`、
`add.Scalar`、`sub.Scalar`，不能只注册 `add.out`。否则 Python 层可能先走
Composite / refs 分解，再间接调用 `out`，调用栈和性能都会变差。

### 4.2.2 什么时候可以保留原始 `meta_out`

不是所有 `meta_out` 都有问题。判断标准看 dispatcher 里 active `Meta` kernel 的来源：

```python
import torch
print(torch._C._dispatch_dump("aten::index.Tensor_out"))
```

可以保留 `meta_out` 的情况：

- `Meta:` 注册来自 `pytorch/build/aten/src/ATen/RegisterMeta_*.cpp`，通常是 C++
  meta kernel，调用栈简洁，例如 `index.Tensor_out`、`gather.out`、`topk.values`。
- 该 op 的 shape / dtype / stride 规则复杂，并且 C++ Meta 路径不会在 profiler
  中出现 `torch/_refs`、`torch/_prims_common`、`torch/_dynamo/eval_frame`。

应该替换 `meta_out` 的情况：

- `Meta:` 注册来自 `torch/_meta_registrations.py`，尤其是内部会走
  `torch/_refs` 或 `torch/_prims_common` 的算子。
- profiler trace 中能看到这类 Python 栈，并且这个 op 位于热路径。

替换时不能只看 shape。必须同时保持 PyTorch 语义：

- dtype promotion：例如 `cumsum` 在 `dtype=None` 且输入为 integral/bool 时，
  functional 结果 dtype 是 `int64`，不能简单继承 `self.scalar_type()`。
- 特殊 shape 规则：例如 `cat` 会忽略 1-D empty tensor 占位输入，
  `torch.cat([torch.empty(0), torch.ones(2, 3)], dim=0/1)` 都是合法的。
- `out` 版本保持“不主动 resize 原始 mcpu out”的约定，但 shape/dtype 规则要和
  PyTorch CPU 行为兼容。

### 4.3 inplace 算子

典型如 `fill_`。

做法：

1. launch 前抽取 `self` 的指针和元信息
2. kernel 内重建临时 CPU view
3. 直接调用 CPU inplace 接口
4. 返回原 `self`

如果 inplace 算子有简单连续布局快路径，也可以直接用 raw pointer kernel，
只在不满足快路径时 fallback 到 CPU view。

### 4.4 launch 捕获方式

推荐顺序：

1. **Pointer capture**：优先捕获 raw pointer、shape、stride、dtype、scalar 等元信息，
   配合 `KernelPointerMemoryGuard`。这是新增 `aten` op 的默认选择。
2. **Heap args capture**：如果捕获的元信息太多，超过 openreg inline queue slot，
   用 `std::make_unique<Args>(...)` 放到堆上，并用 move capture：

```cpp
auto args = std::make_unique<MyOpArgs>(MyOpArgs{
    out_ptr,
    input_ptr,
    sizes,
    strides,
    options,
});

MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::my_op", ([args = std::move(args)]), {
  at::mcpu::KernelPointerMemoryGuard guard({args->out_ptr, args->input_ptr});
  auto cpu_out = at::from_blob(
      args->out_ptr, args->sizes, args->strides,
      args->options.device(c10::DeviceType::CPU));
  // memory work
});
```

这里不用 `std::shared_ptr`，因为 queued task 是唯一消费者。`unique_ptr`
表达单一所有权，也没有引用计数开销。当前 openreg launch 路径是模板转发到
inline `TupleTask`，支持 move-only lambda；如果以后某条路径必须经过
`std::function`，那条路径才需要重新评估，因为 `std::function` 要求 callable
可拷贝。

3. **Tensor capture**：只在确实无法只靠指针和元信息安全重建临时 view 时使用。
   Tensor capture 会延长 Tensor / storage 生命周期，不能作为新增显式 `aten` op
   的默认写法。

### 4.5 何时仍然调用 CPU ATen

有些算子的 dtype promotion、broadcast、indexing 或随机数语义复杂。可以继续在
kernel 内调用 CPU ATen `out` / inplace 接口，但推荐方式仍是：

1. launch 前完成 shape / dtype 检查和输出分配；
2. launch 前抽取输入/输出指针和元信息；
3. kernel 内用这些元信息重建临时 CPU view；
4. 调 CPU ATen 接口执行内存读写。

不要为了调用 CPU ATen 而默认捕获 Tensor。

## 5. 不要做什么

- 不要新写一层 `wrapper_xxx` 再转发。
- 不要把这些显式 op 再塞回 `csrc/aten/native/Extra.cpp`。
- 不要在 `out` 变体里偷偷 `resize_`。
- 不要在 async kernel lambda 里默认捕获 Tensor；能用指针和元信息表达的都用
  pointer capture。
- 不要为单一 queued task 使用 `shared_ptr` 保存 args；优先 `unique_ptr + move`
  capture。

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
./build.sh
```

再跑定向测试：

```bash
cd /path/to/torch_mcpu
LD_LIBRARY_PATH="$PWD/build/cmake_install/torch_mcpu/lib" \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
pytest tests/test_ops.py -k 'Fallback' -q
```

从源码树运行测试时要先 `cd` 到仓库目录，并显式设置 staging lib 目录到
`LD_LIBRARY_PATH`。这样可以避免 Python 包名和源码目录重名导致导入到错误产物。

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
4. 能不能只捕获指针和元信息，在 kernel 内重建临时 CPU view
5. 输入里是否只需要做 view 映射，而不是 copy
6. `out` 尺寸不匹配时是否已经明确报错

满足这几条，基本就是本仓库当前推荐实现方式。
