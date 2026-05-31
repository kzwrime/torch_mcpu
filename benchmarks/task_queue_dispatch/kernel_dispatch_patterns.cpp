// Benchmark dispatch representations with kernel-like argument shapes.
//
// The kernel bodies are intentionally tiny. They keep the real launch shapes
// that matter for dispatch cost: tensor-like pointers, scalar arguments, and
// templated vLLM-style typed kernels, while avoiding large memory loops that
// would hide queue overhead.
//
// Build from the repository root:
//   g++ -O3 -DNDEBUG -std=c++17 -pthread \
//     -Ibenchmarks/task_queue_dispatch \
//     benchmarks/task_queue_dispatch/kernel_dispatch_patterns.cpp \
//     -o build/kernel_dispatch_patterns
//
// Run:
//   ./build/kernel_dispatch_patterns --tasks 500 --iterations 2000 --pin-core
//   75

#include "benchmark_common.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace {

struct Config {
  int tasks = task_queue_dispatch_bench::kDefaultTasks;
  int iterations = task_queue_dispatch_bench::kDefaultIterations;
  int warmup = task_queue_dispatch_bench::kDefaultWarmup;
  int pin_core = -1;
  size_t ring_capacity = task_queue_dispatch_bench::kDefaultRingCapacity;
};

struct Result {
  double batch_ns = 0.0;
  double task_ns = 0.0;
  uint64_t checksum = 0;
};

struct BufferSet {
  static constexpr int64_t kTokens = 1;
  static constexpr int64_t kVocab = 4;
  static constexpr int64_t kElements = 4;
  static constexpr int64_t kRequests = 4;

  std::vector<float> out;
  std::vector<float> src;
  std::vector<float> logits;
  std::vector<int32_t> idx;
  std::vector<float> params;

  BufferSet()
      : out(kElements),
        src(kElements),
        logits(kTokens * kVocab),
        idx(kTokens),
        params(kRequests) {}
};

enum class KernelKind : uint8_t {
  Arange,
  FillScalar,
  TemperatureFloat,
  MinPFloat,
};

struct WorkItem {
  KernelKind kind = KernelKind::Arange;
  int64_t task_index = 0;
};

[[noreturn]] void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--tasks N] [--iterations N] [--warmup N]"
               " [--pin-core N] [--ring-capacity N]\n";
  std::exit(2);
}

int parse_int(const char* value, const char* name, bool positive) {
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (*value == '\0' || *end != '\0' || (positive && parsed <= 0)) {
    throw std::invalid_argument(std::string(name) + " has invalid value");
  }
  return static_cast<int>(parsed);
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--tasks" && i + 1 < argc) {
      cfg.tasks = parse_int(argv[++i], "--tasks", true);
    } else if (arg == "--iterations" && i + 1 < argc) {
      cfg.iterations = parse_int(argv[++i], "--iterations", true);
    } else if (arg == "--warmup" && i + 1 < argc) {
      cfg.warmup = parse_int(argv[++i], "--warmup", true);
    } else if (arg == "--pin-core" && i + 1 < argc) {
      cfg.pin_core = parse_int(argv[++i], "--pin-core", false);
    } else if (arg == "--ring-capacity" && i + 1 < argc) {
      cfg.ring_capacity =
          static_cast<size_t>(parse_int(argv[++i], "--ring-capacity", true));
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
    } else {
      usage(argv[0]);
    }
  }
  return cfg;
}

uint64_t now_ns() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

size_t next_power_of_two(size_t value) {
  size_t power = 1;
  while (power < value) {
    power <<= 1;
  }
  return power;
}

void pin_current_thread(int core) {
#if defined(__linux__)
  if (core < 0) {
    return;
  }
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#else
  (void)core;
#endif
}

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

NOINLINE void kernel_arange_float(
    float* out,
    int64_t n,
    float start,
    float step) {
  out[0] = start;
  out[n - 1] = start + static_cast<float>(n - 1) * step;
}

NOINLINE void kernel_fill_scalar_float(float* out, int64_t n, float value) {
  out[0] = value;
  out[n - 1] = value + 1.0f;
}

class MemoryLikeRuntime {
 public:
  NOINLINE void fill_scalar_member(float* out, int64_t n, float value) {
    kernel_fill_scalar_float(out, n, value);
  }
};

template <typename scalar_t>
NOINLINE void kernel_temperature_typed(
    scalar_t* logits,
    int64_t tokens,
    int64_t stride,
    const int32_t* idx,
    const float* temperature,
    int64_t vocab) {
  for (int64_t tok = 0; tok < tokens; ++tok) {
    float temp = temperature[idx[tok]];
    if (temp == 0.0f || temp == 1.0f) {
      continue;
    }
    scalar_t* row = logits + tok * stride;
    row[0] = static_cast<scalar_t>(static_cast<float>(row[0]) / temp);
    row[vocab - 1] =
        static_cast<scalar_t>(static_cast<float>(row[vocab - 1]) / temp);
  }
}

template <typename scalar_t>
NOINLINE void kernel_min_p_typed(
    scalar_t* logits,
    int64_t tokens,
    int64_t stride,
    const int32_t* idx,
    const float* min_p,
    int64_t vocab) {
  for (int64_t tok = 0; tok < tokens; ++tok) {
    float min_p_val = min_p[idx[tok]];
    if (min_p_val == 0.0f) {
      continue;
    }
    scalar_t* row = logits + tok * stride;
    float max_val = row[0] > row[vocab - 1] ? row[0] : row[vocab - 1];
    const float threshold = max_val - min_p_val;
    if (row[0] < threshold) {
      row[0] = -3.402823466e+38f;
    }
    if (row[vocab - 1] < threshold) {
      row[vocab - 1] = -3.402823466e+38f;
    }
  }
}

void prepare_buffers(BufferSet& buffers, uint64_t iteration) {
  for (int64_t i = 0; i < BufferSet::kElements; ++i) {
    buffers.out[i] = 0.0f;
    buffers.src[i] = static_cast<float>((iteration + i) % 17) * 0.25f;
  }
  for (int64_t i = 0; i < BufferSet::kTokens * BufferSet::kVocab; ++i) {
    buffers.logits[i] = static_cast<float>((iteration + i) % 29) * 0.125f;
  }
  for (int64_t i = 0; i < BufferSet::kTokens; ++i) {
    buffers.idx[i] =
        static_cast<int32_t>((iteration + i) % BufferSet::kRequests);
  }
  for (int64_t i = 0; i < BufferSet::kRequests; ++i) {
    buffers.params[i] =
        0.75f + 0.125f * static_cast<float>((iteration + i) % 5);
  }
}

uint64_t checksum_buffers(const BufferSet& buffers) {
  uint64_t checksum = 0;
  auto mix = [&](float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    checksum ^= static_cast<uint64_t>(bits) + 0x9e3779b97f4a7c15ull +
        (checksum << 6) + (checksum >> 2);
  };
  for (float value : buffers.out) {
    mix(value);
  }
  for (float value : buffers.logits) {
    mix(value);
  }
  return checksum;
}

std::vector<WorkItem> make_work_items(int tasks) {
  std::vector<WorkItem> items(tasks);
  for (int i = 0; i < tasks; ++i) {
    items[i].kind = static_cast<KernelKind>(i % 4);
    items[i].task_index = i;
  }
  return items;
}

void run_work_item_direct(BufferSet& buffers, const WorkItem& item) {
  switch (item.kind) {
    case KernelKind::Arange:
      kernel_arange_float(
          buffers.out.data(),
          BufferSet::kElements,
          static_cast<float>(item.task_index),
          0.5f);
      break;
    case KernelKind::FillScalar:
      kernel_fill_scalar_float(
          buffers.out.data(),
          BufferSet::kElements,
          static_cast<float>(item.task_index % 13));
      break;
    case KernelKind::TemperatureFloat:
      kernel_temperature_typed<float>(
          buffers.logits.data(),
          BufferSet::kTokens,
          BufferSet::kVocab,
          buffers.idx.data(),
          buffers.params.data(),
          BufferSet::kVocab);
      break;
    case KernelKind::MinPFloat:
      kernel_min_p_typed<float>(
          buffers.logits.data(),
          BufferSet::kTokens,
          BufferSet::kVocab,
          buffers.idx.data(),
          buffers.params.data(),
          BufferSet::kVocab);
      break;
  }
}

template <typename Body>
Result run_loop(
    BufferSet& buffers,
    const std::vector<WorkItem>& work_items,
    const Config& cfg,
    Body&& body) {
  uint64_t total_ns = 0;
  uint64_t checksum = 0;
  const int total_iterations = cfg.warmup + cfg.iterations;

  for (int iter = 0; iter < total_iterations; ++iter) {
    prepare_buffers(buffers, static_cast<uint64_t>(iter));
    const uint64_t start = now_ns();
    body(buffers, work_items);
    const uint64_t elapsed = now_ns() - start;
    checksum ^= checksum_buffers(buffers);
    if (iter >= cfg.warmup) {
      total_ns += elapsed;
    }
  }

  const double batch_ns = static_cast<double>(total_ns) / cfg.iterations;
  return {batch_ns, batch_ns / cfg.tasks, checksum};
}

class SpscFunctionRing {
 public:
  explicit SpscFunctionRing(size_t capacity, int pin_core)
      : tasks_(next_power_of_two(capacity)),
        mask_(tasks_.size() - 1),
        worker_([this, pin_core] {
          pin_current_thread(pin_core);
          worker_loop();
        }) {}

  ~SpscFunctionRing() {
    stop_.store(true, std::memory_order_release);
    worker_.join();
  }

  void enqueue(std::function<void()> task) {
    uint64_t tail = tail_.load(std::memory_order_relaxed);
    while (tail - head_.load(std::memory_order_acquire) >= tasks_.size()) {
      std::this_thread::yield();
    }
    tasks_[tail & mask_] = std::move(task);
    tail_.store(tail + 1, std::memory_order_release);
  }

  void synchronize() {
    const uint64_t target = tail_.load(std::memory_order_acquire);
    while (completed_.load(std::memory_order_acquire) < target) {
      std::this_thread::yield();
    }
  }

 private:
  void worker_loop() {
    while (true) {
      uint64_t head = head_.load(std::memory_order_relaxed);
      if (head == tail_.load(std::memory_order_acquire)) {
        if (stop_.load(std::memory_order_acquire)) {
          return;
        }
        std::this_thread::yield();
        continue;
      }
      std::function<void()> task = std::move(tasks_[head & mask_]);
      head_.store(head + 1, std::memory_order_release);
      task();
      completed_.store(head + 1, std::memory_order_release);
    }
  }

  std::vector<std::function<void()>> tasks_;
  const uint64_t mask_;
  alignas(64) std::atomic<uint64_t> head_{0};
  alignas(64) std::atomic<uint64_t> tail_{0};
  alignas(64) std::atomic<uint64_t> completed_{0};
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

template <bool UseInvoke>
class SpscInlineTupleRing {
 public:
  explicit SpscInlineTupleRing(size_t capacity, int pin_core)
      : slots_(next_power_of_two(capacity)),
        mask_(slots_.size() - 1),
        worker_([this, pin_core] {
          pin_current_thread(pin_core);
          worker_loop();
        }) {}

  ~SpscInlineTupleRing() {
    stop_.store(true, std::memory_order_release);
    worker_.join();
  }

  template <typename Func, typename... Args>
  void enqueue(Func&& func, Args&&... args) {
    using Tuple = std::tuple<std::decay_t<Func>, std::decay_t<Args>...>;
    static_assert(sizeof(Tuple) <= Slot::kStorageSize);
    static_assert(alignof(Tuple) <= alignof(std::max_align_t));

    uint64_t tail = tail_.load(std::memory_order_relaxed);
    while (tail - head_.load(std::memory_order_acquire) >= slots_.size()) {
      std::this_thread::yield();
    }

    Slot& slot = slots_[tail & mask_];
    new (slot.storage)
        Tuple(std::forward<Func>(func), std::forward<Args>(args)...);
    slot.run = [](void* storage) {
      Tuple& tuple = *reinterpret_cast<Tuple*>(storage);
      std::apply(
          [](auto& callable, auto&... unpacked) {
            if constexpr (UseInvoke) {
              std::invoke(callable, unpacked...);
            } else {
              callable(unpacked...);
            }
          },
          tuple);
    };
    slot.destroy = [](void* storage) {
      reinterpret_cast<Tuple*>(storage)->~Tuple();
    };
    tail_.store(tail + 1, std::memory_order_release);
  }

  void synchronize() {
    const uint64_t target = tail_.load(std::memory_order_acquire);
    while (completed_.load(std::memory_order_acquire) < target) {
      std::this_thread::yield();
    }
  }

 private:
  struct Slot {
    static constexpr size_t kStorageSize = 128;
    void (*run)(void*) = nullptr;
    void (*destroy)(void*) = nullptr;
    alignas(std::max_align_t) unsigned char storage[kStorageSize];
  };

  void worker_loop() {
    while (true) {
      uint64_t head = head_.load(std::memory_order_relaxed);
      if (head == tail_.load(std::memory_order_acquire)) {
        if (stop_.load(std::memory_order_acquire)) {
          return;
        }
        std::this_thread::yield();
        continue;
      }

      Slot& slot = slots_[head & mask_];
      head_.store(head + 1, std::memory_order_release);
      slot.run(slot.storage);
      slot.destroy(slot.storage);
      completed_.store(head + 1, std::memory_order_release);
    }
  }

  std::vector<Slot> slots_;
  const uint64_t mask_;
  alignas(64) std::atomic<uint64_t> head_{0};
  alignas(64) std::atomic<uint64_t> tail_{0};
  alignas(64) std::atomic<uint64_t> completed_{0};
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

struct ArangeTask {
  float* out = nullptr;
  int64_t n = 0;
  float start = 0.0f;
  float step = 0.0f;
};

struct FillTask {
  float* out = nullptr;
  int64_t n = 0;
  float value = 0.0f;
};

struct VllmTask {
  float* logits = nullptr;
  int64_t tokens = 0;
  int64_t stride = 0;
  const int32_t* idx = nullptr;
  const float* params = nullptr;
  int64_t vocab = 0;
};

struct KernelDescriptor {
  KernelKind kind = KernelKind::Arange;
  union {
    ArangeTask arange;
    FillTask fill;
    VllmTask vllm;
  };

  KernelDescriptor() : kind(KernelKind::Arange), arange{} {}
};

class SpscTypedKernelRing {
 public:
  explicit SpscTypedKernelRing(size_t capacity, int pin_core)
      : tasks_(next_power_of_two(capacity)),
        mask_(tasks_.size() - 1),
        worker_([this, pin_core] {
          pin_current_thread(pin_core);
          worker_loop();
        }) {}

  ~SpscTypedKernelRing() {
    stop_.store(true, std::memory_order_release);
    worker_.join();
  }

  void enqueue(const KernelDescriptor& task) {
    uint64_t tail = tail_.load(std::memory_order_relaxed);
    while (tail - head_.load(std::memory_order_acquire) >= tasks_.size()) {
      std::this_thread::yield();
    }
    tasks_[tail & mask_] = task;
    tail_.store(tail + 1, std::memory_order_release);
  }

  void synchronize() {
    const uint64_t target = tail_.load(std::memory_order_acquire);
    while (completed_.load(std::memory_order_acquire) < target) {
      std::this_thread::yield();
    }
  }

 private:
  static void run_descriptor(const KernelDescriptor& task) {
    switch (task.kind) {
      case KernelKind::Arange:
        kernel_arange_float(
            task.arange.out,
            task.arange.n,
            task.arange.start,
            task.arange.step);
        break;
      case KernelKind::FillScalar:
        kernel_fill_scalar_float(task.fill.out, task.fill.n, task.fill.value);
        break;
      case KernelKind::TemperatureFloat:
        kernel_temperature_typed<float>(
            task.vllm.logits,
            task.vllm.tokens,
            task.vllm.stride,
            task.vllm.idx,
            task.vllm.params,
            task.vllm.vocab);
        break;
      case KernelKind::MinPFloat:
        kernel_min_p_typed<float>(
            task.vllm.logits,
            task.vllm.tokens,
            task.vllm.stride,
            task.vllm.idx,
            task.vllm.params,
            task.vllm.vocab);
        break;
    }
  }

  void worker_loop() {
    while (true) {
      uint64_t head = head_.load(std::memory_order_relaxed);
      if (head == tail_.load(std::memory_order_acquire)) {
        if (stop_.load(std::memory_order_acquire)) {
          return;
        }
        std::this_thread::yield();
        continue;
      }
      KernelDescriptor task = tasks_[head & mask_];
      head_.store(head + 1, std::memory_order_release);
      run_descriptor(task);
      completed_.store(head + 1, std::memory_order_release);
    }
  }

  std::vector<KernelDescriptor> tasks_;
  const uint64_t mask_;
  alignas(64) std::atomic<uint64_t> head_{0};
  alignas(64) std::atomic<uint64_t> tail_{0};
  alignas(64) std::atomic<uint64_t> completed_{0};
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

KernelDescriptor make_descriptor(BufferSet& buffers, const WorkItem& item) {
  KernelDescriptor task;
  task.kind = item.kind;
  switch (item.kind) {
    case KernelKind::Arange:
      task.arange = {
          buffers.out.data(),
          BufferSet::kElements,
          static_cast<float>(item.task_index),
          0.5f};
      break;
    case KernelKind::FillScalar:
      task.fill = {
          buffers.out.data(),
          BufferSet::kElements,
          static_cast<float>(item.task_index % 13)};
      break;
    case KernelKind::TemperatureFloat:
    case KernelKind::MinPFloat:
      task.vllm = {
          buffers.logits.data(),
          BufferSet::kTokens,
          BufferSet::kVocab,
          buffers.idx.data(),
          buffers.params.data(),
          BufferSet::kVocab};
      break;
  }
  return task;
}

void print_result(const char* name, const Result& result, double baseline) {
  std::cout << std::setw(24) << name << std::setw(16) << result.batch_ns
            << std::setw(16) << result.task_ns << std::setw(16)
            << (result.task_ns - baseline) << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    BufferSet buffers;
    const std::vector<WorkItem> work_items = make_work_items(cfg.tasks);

    const Result direct =
        run_loop(buffers, work_items, cfg, [](BufferSet& b, const auto& items) {
          for (const WorkItem& item : items) {
            run_work_item_direct(b, item);
          }
        });

    Result function_ring_result;
    {
      SpscFunctionRing queue(cfg.ring_capacity, cfg.pin_core);
      function_ring_result = run_loop(
          buffers, work_items, cfg, [&queue](BufferSet& b, const auto& items) {
            for (const WorkItem& item : items) {
              switch (item.kind) {
                case KernelKind::Arange:
                  queue.enqueue(
                      [out = b.out.data(), task_index = item.task_index] {
                        kernel_arange_float(
                            out,
                            BufferSet::kElements,
                            static_cast<float>(task_index),
                            0.5f);
                      });
                  break;
                case KernelKind::FillScalar:
                  queue.enqueue(
                      [out = b.out.data(), task_index = item.task_index] {
                        kernel_fill_scalar_float(
                            out,
                            BufferSet::kElements,
                            static_cast<float>(task_index % 13));
                      });
                  break;
                case KernelKind::TemperatureFloat:
                  queue.enqueue([logits = b.logits.data(),
                                 idx = b.idx.data(),
                                 params = b.params.data()] {
                    kernel_temperature_typed<float>(
                        logits,
                        BufferSet::kTokens,
                        BufferSet::kVocab,
                        idx,
                        params,
                        BufferSet::kVocab);
                  });
                  break;
                case KernelKind::MinPFloat:
                  queue.enqueue([logits = b.logits.data(),
                                 idx = b.idx.data(),
                                 params = b.params.data()] {
                    kernel_min_p_typed<float>(
                        logits,
                        BufferSet::kTokens,
                        BufferSet::kVocab,
                        idx,
                        params,
                        BufferSet::kVocab);
                  });
                  break;
              }
            }
            queue.synchronize();
          });
    }

    Result tuple_direct_result;
    {
      SpscInlineTupleRing<false> queue(cfg.ring_capacity, cfg.pin_core);
      tuple_direct_result = run_loop(
          buffers, work_items, cfg, [&queue](BufferSet& b, const auto& items) {
            for (const WorkItem& item : items) {
              switch (item.kind) {
                case KernelKind::Arange:
                  queue.enqueue(
                      kernel_arange_float,
                      b.out.data(),
                      BufferSet::kElements,
                      static_cast<float>(item.task_index),
                      0.5f);
                  break;
                case KernelKind::FillScalar:
                  queue.enqueue(
                      kernel_fill_scalar_float,
                      b.out.data(),
                      BufferSet::kElements,
                      static_cast<float>(item.task_index % 13));
                  break;
                case KernelKind::TemperatureFloat:
                  queue.enqueue(
                      kernel_temperature_typed<float>,
                      b.logits.data(),
                      BufferSet::kTokens,
                      BufferSet::kVocab,
                      b.idx.data(),
                      b.params.data(),
                      BufferSet::kVocab);
                  break;
                case KernelKind::MinPFloat:
                  queue.enqueue(
                      kernel_min_p_typed<float>,
                      b.logits.data(),
                      BufferSet::kTokens,
                      BufferSet::kVocab,
                      b.idx.data(),
                      b.params.data(),
                      BufferSet::kVocab);
                  break;
              }
            }
            queue.synchronize();
          });
    }

    Result tuple_invoke_result;
    {
      SpscInlineTupleRing<true> queue(cfg.ring_capacity, cfg.pin_core);
      MemoryLikeRuntime runtime;
      tuple_invoke_result = run_loop(
          buffers,
          work_items,
          cfg,
          [&queue, &runtime](BufferSet& b, const auto& items) {
            for (const WorkItem& item : items) {
              switch (item.kind) {
                case KernelKind::Arange:
                  queue.enqueue(
                      kernel_arange_float,
                      b.out.data(),
                      BufferSet::kElements,
                      static_cast<float>(item.task_index),
                      0.5f);
                  break;
                case KernelKind::FillScalar:
                  queue.enqueue(
                      &MemoryLikeRuntime::fill_scalar_member,
                      &runtime,
                      b.out.data(),
                      BufferSet::kElements,
                      static_cast<float>(item.task_index % 13));
                  break;
                case KernelKind::TemperatureFloat:
                  queue.enqueue(
                      kernel_temperature_typed<float>,
                      b.logits.data(),
                      BufferSet::kTokens,
                      BufferSet::kVocab,
                      b.idx.data(),
                      b.params.data(),
                      BufferSet::kVocab);
                  break;
                case KernelKind::MinPFloat:
                  queue.enqueue(
                      kernel_min_p_typed<float>,
                      b.logits.data(),
                      BufferSet::kTokens,
                      BufferSet::kVocab,
                      b.idx.data(),
                      b.params.data(),
                      BufferSet::kVocab);
                  break;
              }
            }
            queue.synchronize();
          });
    }

    Result typed_ring_result;
    {
      SpscTypedKernelRing queue(cfg.ring_capacity, cfg.pin_core);
      typed_ring_result = run_loop(
          buffers, work_items, cfg, [&queue](BufferSet& b, const auto& items) {
            for (const WorkItem& item : items) {
              queue.enqueue(make_descriptor(b, item));
            }
            queue.synchronize();
          });
    }

    std::cout << "tasks: " << cfg.tasks << "\n";
    std::cout << "iterations: " << cfg.iterations << " warmup: " << cfg.warmup
              << "\n";
    std::cout << "pin_core: " << cfg.pin_core
              << " ring_capacity: " << next_power_of_two(cfg.ring_capacity)
              << "\n";
    std::cout << "workload: arange, fill_scalar, temperature<float>, "
                 "min_p<float> round-robin\n\n";
    std::cout << "tuple_with_invoke uses a member-function-pointer fill task "
                 "to model memory.cpp-style launches\n\n";

    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(24) << "mode" << std::setw(16) << "batch_ns"
              << std::setw(16) << "task_ns" << std::setw(16) << "extra_ns"
              << "\n";
    print_result("direct", direct, direct.task_ns);
    print_result("std_function_ring", function_ring_result, direct.task_ns);
    print_result("tuple_direct_call", tuple_direct_result, direct.task_ns);
    print_result("tuple_with_invoke", tuple_invoke_result, direct.task_ns);
    print_result("typed_kernel_ring", typed_ring_result, direct.task_ns);

    std::cout << "\nchecksum: 0x" << std::hex << direct.checksum << std::dec
              << "\n";
    std::cout << "all queue checksums match direct: "
              << ((function_ring_result.checksum == direct.checksum &&
                   tuple_direct_result.checksum == direct.checksum &&
                   tuple_invoke_result.checksum == direct.checksum &&
                   typed_ring_result.checksum == direct.checksum)
                      ? "yes"
                      : "no")
              << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
