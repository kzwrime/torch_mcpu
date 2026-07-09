# OpenReg Stream 并发入队修复设计

## 背景

`docs/vllm_stream_spsc_violation_analysis.md` 中的日志证明，vLLM 场景里同一个 OpenReg stream 会被多个线程提交任务：

- 主 vLLM worker 线程先向 pool stream 提交任务；
- 默认 stream 的 worker 线程在执行任务或释放捕获对象时，间接进入 allocator/event 路径；
- allocator 通过 `McpuEvent::record(stream)` 调用 `orEventRecord`，再次向同一个 pool stream 入队。

原实现是 SPSC ring：producer 直接读取 `tail`，构造 slot，再 `tail.store(index + 1)` 发布。多个 producer 同时进入时可能拿到同一个 `tail`，导致 slot 覆盖、任务丢失，或者 worker 读到未完整构造的任务。

## 性能目标

这次修复优先保证两个方向的性能：

- host 下发任务性能：vLLM 常见模式是一次提交数百个 kernel，提交端多一次原子 CAS 有成本，但通常会被后续 kernel 执行摊薄。
- worker 调度任务性能：worker 在流上串行执行所有任务；如果消费侧每个任务都增加重锁、堆分配或复杂分支，会直接影响短 kernel 和 event barrier 的进度。

因此本次最终方案把主要额外成本放在 producer 侧，worker 侧只增加一个 per-slot `sequence` acquire load，避免 mutex、condition_variable、`std::function` 或 heap task。

## Benchmark 设计校验

当前主要参考 benchmark 是 `benchmarks/task_queue_dispatch/openreg_queue_overhead.cpp`。

它的计时窗口是：

```text
start timer
  for each task:
    orLaunchKernel(...)
  orStreamSynchronize(stream)
stop timer
```

因此它测到的是批量 host enqueue 加 worker 执行完成的端到端时间，而不是严格拆分的两个指标。由于 benchmark task 是极轻量的 `lightweight_task`，worker 每任务调度成本在结果中占比很高；这适合暴露 queue/dispatch overhead，但不能直接代表矩阵乘等长 kernel 的真实占比。

对本次修复的解释方式应是：

- benchmark 结果可以用于发现“worker 热路径是否被明显拖慢”；
- host CAS 增量在这个微基准里会被放大；
- 实际 vLLM 中若每批包含大量长 kernel，host enqueue 开销更容易被摊薄；
- 后续若需要精确拆分，应新增两个 benchmark：只测 producer enqueue throughput，以及 worker 预填充任务后的 consume throughput。

## 最终设计

最终采用 bounded MPSC ring：

- `tail` 从普通发布指针变成多 producer 共享的 slot reservation 指针；
- producer 读取目标 slot 的 `sequence`，确认 slot 空闲后用 `tail.compare_exchange_weak` 预留唯一 `index`；
- producer 在该 slot placement-new `TupleTask`，写入 `run/destroy` 函数指针；
- producer 构造完成后用 `slot.sequence.store(index + 1, release)` 发布 ready；
- worker 按 `head` 顺序读取 slot，只在 `sequence == head + 1` 时执行；
- worker 执行并销毁任务后，用 `slot.sequence.store(head + kQueueCapacity, release)` 释放 slot。

这个设计保证：

- 多个 producer 不会写同一个 slot；
- worker 不会读取未构造完成的任务；
- worker 仍按 stream 顺序执行任务；
- allocator/event 从 worker 线程发起的交叉入队不再破坏队列结构；
- 队列容量仍是固定 ring，不引入每个普通任务的堆分配。

## 取舍

本轮也尝试过几个替代方向：

- producer spinlock：语义简单，但单 producer benchmark 明显退化，且所有 producer 被串行化。
- `fetch_add` reservation 加 ordered publish：正确但 producer/worker 同步压力更大，benchmark 最差。
- owner fast path 加 foreign queue：思路是保留主 producer 的 SPSC ring，把跨线程提交放到慢队列；实测代码复杂且热路径仍因 owner 检查和冷路径膨胀退化明显。

当前 CAS + per-slot sequence 方案是本轮实测中正确性和性能最均衡的方案。它牺牲一部分 host enqueue 成本，换取完整的多 producer 安全性，并尽量保持 worker 消费逻辑紧凑。

## 验证结果

OpenReg 单元测试：

```text
cmake --build build --target ortests -j 8
47 tests passed
```

新增回归测试：

- `StreamAcceptsConcurrentProducers`：8 个 host producer 同时向同一 stream 提交 32768 个任务，验证无丢失、无重复、无 hang。
- `WorkerThreadCanRecordEventWhileHostProducesTargetStream`：一个 stream worker 在线程内向目标 stream `orEventRecord`，同时 host 线程向目标 stream 提交普通任务，验证 event 和任务都完成。

最终 queue benchmark：

```text
taskset -c 76,78 env TORCH_MCPU_STREAM_WORKER_CORE=78 \
  benchmarks/task_queue_dispatch/build/openreg_queue_overhead \
  --tasks 500 --iterations 30000 --warmup 3000 --producer-core 76

func_args  54.2 ns/task
lambda     55.9 ns/task
```

这个结果应理解为极短任务下的端到端 queue overhead。对 vLLM 中更长的矩阵乘 kernel，producer 侧新增 CAS 的相对影响预计更小；真正需要持续关注的是 worker 侧是否出现明显调度退化。
