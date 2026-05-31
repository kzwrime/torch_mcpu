#pragma once

// Shared defaults for standalone task-dispatch benchmarks in this directory.
// The benchmark .cpp files remain self-contained enough to copy to other
// machines, while this header documents the common run shape.

namespace task_queue_dispatch_bench {

constexpr int kDefaultTasks = 500;
constexpr int kDefaultIterations = 2000;
constexpr int kDefaultWarmup = 200;
constexpr int kDefaultRingCapacity = 16384;

} // namespace task_queue_dispatch_bench
