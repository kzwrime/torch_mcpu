// Progressively add production-stream costs on top of tuple_with_invoke.
//
// This benchmark keeps the same tiny kernel mix for all modes and changes one
// queue-design dimension at a time, so the per-task delta is easier to read.

#include "benchmark_common.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
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

enum class Status {
  Success = 0,
  Error = 1,
};

struct Config {
  int tasks = task_queue_dispatch_bench::kDefaultTasks;
  int iterations = task_queue_dispatch_bench::kDefaultIterations;
  int warmup = task_queue_dispatch_bench::kDefaultWarmup;
  int producer_core = -1;
  int worker_core = -1;
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
  std::vector<float> logits;
  std::vector<int32_t> idx;
  std::vector<float> params;

  BufferSet()
      : out(kElements),
        logits(kTokens * kVocab),
        idx(kTokens),
        params(kRequests) {}
};

enum class KernelKind : uint8_t {
  Arange,
  FillMember,
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
               " [--producer-core N] [--worker-core N]"
               " [--ring-capacity N]\n";
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
    } else if (arg == "--producer-core" && i + 1 < argc) {
      cfg.producer_core = parse_int(argv[++i], "--producer-core", false);
    } else if (arg == "--worker-core" && i + 1 < argc) {
      cfg.worker_core = parse_int(argv[++i], "--worker-core", false);
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

#if defined(__x86_64__) || defined(__i386__)
inline void cpu_relax() {
  __builtin_ia32_pause();
}
#else
inline void cpu_relax() {
  std::this_thread::yield();
}
#endif

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

class RuntimeLike {
 public:
  NOINLINE void fill_scalar_member(float* out, int64_t n, float value) {
    out[0] = value;
    out[n - 1] = value + 1.0f;
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

void run_work_item_direct(
    BufferSet& buffers,
    RuntimeLike& runtime,
    const WorkItem& item) {
  switch (item.kind) {
    case KernelKind::Arange:
      kernel_arange_float(
          buffers.out.data(),
          BufferSet::kElements,
          static_cast<float>(item.task_index),
          0.5f);
      break;
    case KernelKind::FillMember:
      runtime.fill_scalar_member(
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
    RuntimeLike& runtime,
    const std::vector<WorkItem>& work_items,
    const Config& cfg,
    Body&& body) {
  uint64_t total_ns = 0;
  uint64_t checksum = 0;
  const int total_iterations = cfg.warmup + cfg.iterations;

  for (int iter = 0; iter < total_iterations; ++iter) {
    prepare_buffers(buffers, static_cast<uint64_t>(iter));
    const uint64_t start = now_ns();
    body(buffers, runtime, work_items);
    const uint64_t elapsed = now_ns() - start;
    checksum ^= checksum_buffers(buffers);
    if (iter >= cfg.warmup) {
      total_ns += elapsed;
    }
  }

  const double batch_ns = static_cast<double>(total_ns) / cfg.iterations;
  return {batch_ns, batch_ns / cfg.tasks, checksum};
}

template <size_t StorageSize>
struct QueueSlot {
  void (*run)(void*) = nullptr;
  void (*destroy)(void*) = nullptr;
  alignas(std::max_align_t) unsigned char storage[StorageSize];
};

template <typename Func, typename... Args>
struct TupleTask {
  std::tuple<Func, Args...> payload;

  template <typename F, typename... A>
  explicit TupleTask(F&& func, A&&... args)
      : payload(std::forward<F>(func), std::forward<A>(args)...) {}

  void operator()() {
    invoke(std::make_index_sequence<sizeof...(Args)>{});
  }

  template <size_t... I>
  void invoke(std::index_sequence<I...>) {
    std::invoke(std::get<0>(payload), std::get<I + 1>(payload)...);
  }
};

template <typename Task>
void run_task(void* ptr) {
  (*static_cast<Task*>(ptr))();
}

template <typename Task>
void destroy_task(void* ptr) {
  static_cast<Task*>(ptr)->~Task();
}

template <size_t StorageSize>
class VectorHeadRing {
 public:
  explicit VectorHeadRing(size_t capacity, int worker_core)
      : slots_(next_power_of_two(capacity)),
        mask_(slots_.size() - 1),
        worker_([this, worker_core] {
          pin_current_thread(worker_core);
          worker_loop();
        }) {}

  ~VectorHeadRing() {
    stop_.store(true, std::memory_order_release);
    worker_.join();
  }

  template <typename Func, typename... Args>
  void enqueue(Func&& func, Args&&... args) {
    using Task = TupleTask<std::decay_t<Func>, std::decay_t<Args>...>;
    static_assert(sizeof(Task) <= StorageSize);
    static_assert(alignof(Task) <= alignof(std::max_align_t));

    uint64_t tail = tail_.load(std::memory_order_relaxed);
    while (tail - head_.load(std::memory_order_acquire) >= slots_.size()) {
      cpu_relax();
    }

    auto& slot = slots_[tail & mask_];
    new (slot.storage)
        Task(std::forward<Func>(func), std::forward<Args>(args)...);
    slot.run = &run_task<Task>;
    slot.destroy = &destroy_task<Task>;
    tail_.store(tail + 1, std::memory_order_release);
  }

  void synchronize() {
    const uint64_t target = tail_.load(std::memory_order_acquire);
    while (completed_.load(std::memory_order_acquire) < target) {
      cpu_relax();
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
        cpu_relax();
        continue;
      }

      auto& slot = slots_[head & mask_];
      head_.store(head + 1, std::memory_order_release);
      slot.run(slot.storage);
      slot.destroy(slot.storage);
      completed_.store(head + 1, std::memory_order_release);
    }
  }

  std::vector<QueueSlot<StorageSize>> slots_;
  const uint64_t mask_;
  alignas(64) std::atomic<uint64_t> head_{0};
  alignas(64) std::atomic<uint64_t> tail_{0};
  alignas(64) std::atomic<uint64_t> completed_{0};
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

template <size_t StorageSize, size_t Capacity>
class FixedHeadRing {
 public:
  explicit FixedHeadRing(int worker_core)
      : worker_([this, worker_core] {
          pin_current_thread(worker_core);
          worker_loop();
        }) {}

  ~FixedHeadRing() {
    stop_.store(true, std::memory_order_release);
    worker_.join();
  }

  template <typename Func, typename... Args>
  void enqueue(Func&& func, Args&&... args) {
    using Task = TupleTask<std::decay_t<Func>, std::decay_t<Args>...>;
    static_assert(sizeof(Task) <= StorageSize);
    static_assert(alignof(Task) <= alignof(std::max_align_t));

    uint64_t tail = tail_.load(std::memory_order_relaxed);
    while (tail - head_.load(std::memory_order_acquire) >= Capacity) {
      cpu_relax();
    }

    auto& slot = slots_[tail & (Capacity - 1)];
    new (slot.storage)
        Task(std::forward<Func>(func), std::forward<Args>(args)...);
    slot.run = &run_task<Task>;
    slot.destroy = &destroy_task<Task>;
    tail_.store(tail + 1, std::memory_order_release);
  }

  void synchronize() {
    const uint64_t target = tail_.load(std::memory_order_acquire);
    while (completed_.load(std::memory_order_acquire) < target) {
      cpu_relax();
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
        cpu_relax();
        continue;
      }

      auto& slot = slots_[head & (Capacity - 1)];
      head_.store(head + 1, std::memory_order_release);
      slot.run(slot.storage);
      slot.destroy(slot.storage);
      completed_.store(head + 1, std::memory_order_release);
    }
  }

  QueueSlot<StorageSize> slots_[Capacity];
  alignas(64) std::atomic<uint64_t> head_{0};
  alignas(64) std::atomic<uint64_t> tail_{0};
  alignas(64) std::atomic<uint64_t> completed_{0};
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

template <size_t StorageSize>
class VectorCachedRing {
 public:
  explicit VectorCachedRing(size_t capacity, int worker_core)
      : slots_(next_power_of_two(capacity)),
        mask_(slots_.size() - 1),
        worker_([this, worker_core] {
          pin_current_thread(worker_core);
          worker_loop();
        }) {}

  ~VectorCachedRing() {
    stop_.store(true, std::memory_order_release);
    worker_.join();
  }

  template <typename Func, typename... Args>
  void enqueue(Func&& func, Args&&... args) {
    using Task = TupleTask<std::decay_t<Func>, std::decay_t<Args>...>;
    static_assert(sizeof(Task) <= StorageSize);
    static_assert(alignof(Task) <= alignof(std::max_align_t));

    auto index = producer_tail_;
    if (index - producer_completed_cache_ >= slots_.size()) {
      do {
        producer_completed_cache_ = completed_.load(std::memory_order_acquire);
        cpu_relax();
      } while (index - producer_completed_cache_ >= slots_.size());
    }

    auto& slot = slots_[index & mask_];
    new (slot.storage)
        Task(std::forward<Func>(func), std::forward<Args>(args)...);
    slot.run = &run_task<Task>;
    slot.destroy = &destroy_task<Task>;
    producer_tail_ = index + 1;
    tail_.store(index + 1, std::memory_order_release);
  }

  void synchronize() {
    const uint64_t target = tail_.load(std::memory_order_acquire);
    while (completed_.load(std::memory_order_acquire) < target) {
      cpu_relax();
    }
  }

 private:
  void worker_loop() {
    uint64_t local_head = 0;
    while (true) {
      const uint64_t local_tail = tail_.load(std::memory_order_acquire);
      if (local_head == local_tail) {
        if (stop_.load(std::memory_order_acquire)) {
          return;
        }
        cpu_relax();
        continue;
      }

      auto& slot = slots_[local_head & mask_];
      slot.run(slot.storage);
      slot.destroy(slot.storage);
      ++local_head;
      completed_.store(local_head, std::memory_order_release);
    }
  }

  std::vector<QueueSlot<StorageSize>> slots_;
  const uint64_t mask_;
  alignas(64) uint64_t producer_tail_ = 0;
  uint64_t producer_completed_cache_ = 0;
  alignas(64) std::atomic<uint64_t> tail_{0};
  alignas(64) std::atomic<uint64_t> completed_{0};
  alignas(64) std::atomic<bool> stop_{false};
  std::thread worker_;
};

template <size_t StorageSize, size_t Capacity>
class FixedCachedRing {
 public:
  explicit FixedCachedRing(int worker_core)
      : worker_([this, worker_core] {
          pin_current_thread(worker_core);
          worker_loop();
        }) {}

  ~FixedCachedRing() {
    stop_.store(true, std::memory_order_release);
    worker_.join();
  }

  template <typename Func, typename... Args>
  void enqueue(Func&& func, Args&&... args) {
    using Task = TupleTask<std::decay_t<Func>, std::decay_t<Args>...>;
    static_assert(sizeof(Task) <= StorageSize);
    static_assert(alignof(Task) <= alignof(std::max_align_t));

    auto index = producer_tail_;
    if (index - producer_completed_cache_ >= Capacity) {
      do {
        producer_completed_cache_ = completed_.load(std::memory_order_acquire);
        cpu_relax();
      } while (index - producer_completed_cache_ >= Capacity);
    }

    auto& slot = slots_[index & (Capacity - 1)];
    new (slot.storage)
        Task(std::forward<Func>(func), std::forward<Args>(args)...);
    slot.run = &run_task<Task>;
    slot.destroy = &destroy_task<Task>;
    producer_tail_ = index + 1;
    tail_.store(index + 1, std::memory_order_release);
  }

  void synchronize() {
    const uint64_t target = tail_.load(std::memory_order_acquire);
    while (completed_.load(std::memory_order_acquire) < target) {
      cpu_relax();
    }
  }

 private:
  void worker_loop() {
    uint64_t local_head = 0;
    while (true) {
      const uint64_t local_tail = tail_.load(std::memory_order_acquire);
      if (local_head == local_tail) {
        if (stop_.load(std::memory_order_acquire)) {
          return;
        }
        cpu_relax();
        continue;
      }

      auto& slot = slots_[local_head & (Capacity - 1)];
      slot.run(slot.storage);
      slot.destroy(slot.storage);
      ++local_head;
      completed_.store(local_head, std::memory_order_release);
    }
  }

  QueueSlot<StorageSize> slots_[Capacity];
  alignas(64) uint64_t producer_tail_ = 0;
  uint64_t producer_completed_cache_ = 0;
  alignas(64) std::atomic<uint64_t> tail_{0};
  alignas(64) std::atomic<uint64_t> completed_{0};
  alignas(64) std::atomic<bool> stop_{false};
  std::thread worker_;
};

template <typename Ring>
class ApiWrapper {
 public:
  explicit ApiWrapper(Ring& ring) : ring_(ring) {}

  template <typename Func, typename... Args>
  Status launch(Func&& func, Args&&... args) {
    if (this == nullptr) {
      return Status::Error;
    }
    ring_.enqueue(std::forward<Func>(func), std::forward<Args>(args)...);
    return Status::Success;
  }

  Status synchronize() {
    if (this == nullptr) {
      return Status::Error;
    }
    ring_.synchronize();
    return Status::Success;
  }

 private:
  Ring& ring_;
};

template <typename Enqueue>
void enqueue_work_item(
    Enqueue&& enqueue,
    BufferSet& buffers,
    RuntimeLike& runtime,
    const WorkItem& item) {
  switch (item.kind) {
    case KernelKind::Arange:
      enqueue(
          kernel_arange_float,
          buffers.out.data(),
          BufferSet::kElements,
          static_cast<float>(item.task_index),
          0.5f);
      break;
    case KernelKind::FillMember:
      enqueue(
          &RuntimeLike::fill_scalar_member,
          &runtime,
          buffers.out.data(),
          BufferSet::kElements,
          static_cast<float>(item.task_index % 13));
      break;
    case KernelKind::TemperatureFloat:
      enqueue(
          kernel_temperature_typed<float>,
          buffers.logits.data(),
          BufferSet::kTokens,
          BufferSet::kVocab,
          buffers.idx.data(),
          buffers.params.data(),
          BufferSet::kVocab);
      break;
    case KernelKind::MinPFloat:
      enqueue(
          kernel_min_p_typed<float>,
          buffers.logits.data(),
          BufferSet::kTokens,
          BufferSet::kVocab,
          buffers.idx.data(),
          buffers.params.data(),
          BufferSet::kVocab);
      break;
  }
}

template <typename Ring>
Result run_ring(
    Ring& ring,
    BufferSet& buffers,
    RuntimeLike& runtime,
    const std::vector<WorkItem>& work_items,
    const Config& cfg) {
  return run_loop(
      buffers,
      runtime,
      work_items,
      cfg,
      [&](BufferSet& loop_buffers,
          RuntimeLike& loop_runtime,
          const std::vector<WorkItem>& loop_items) {
        for (const WorkItem& item : loop_items) {
          enqueue_work_item(
              [&](auto&& func, auto&&... args) {
                ring.enqueue(
                    std::forward<decltype(func)>(func),
                    std::forward<decltype(args)>(args)...);
              },
              loop_buffers,
              loop_runtime,
              item);
        }
        ring.synchronize();
      });
}

template <typename Api>
Result run_api(
    Api& api,
    BufferSet& buffers,
    RuntimeLike& runtime,
    const std::vector<WorkItem>& work_items,
    const Config& cfg) {
  return run_loop(
      buffers,
      runtime,
      work_items,
      cfg,
      [&](BufferSet& loop_buffers,
          RuntimeLike& loop_runtime,
          const std::vector<WorkItem>& loop_items) {
        for (const WorkItem& item : loop_items) {
          enqueue_work_item(
              [&](auto&& func, auto&&... args) {
                if (api.launch(
                        std::forward<decltype(func)>(func),
                        std::forward<decltype(args)>(args)...) !=
                    Status::Success) {
                  throw std::runtime_error("launch failed");
                }
              },
              loop_buffers,
              loop_runtime,
              item);
        }
        if (api.synchronize() != Status::Success) {
          throw std::runtime_error("synchronize failed");
        }
      });
}

void print_result(
    const char* name,
    const Result& direct,
    const Result& result) {
  std::cout << std::setw(30) << name << std::setw(14) << result.batch_ns
            << std::setw(14) << result.task_ns << std::setw(14)
            << result.task_ns - direct.task_ns << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    pin_current_thread(cfg.producer_core);

    BufferSet direct_buffers;
    BufferSet queue_buffers;
    RuntimeLike direct_runtime;
    RuntimeLike queue_runtime;
    const auto work_items = make_work_items(cfg.tasks);

    const Result direct = run_loop(
        direct_buffers,
        direct_runtime,
        work_items,
        cfg,
        [](BufferSet& buffers,
           RuntimeLike& runtime,
           const std::vector<WorkItem>& items) {
          for (const WorkItem& item : items) {
            run_work_item_direct(buffers, runtime, item);
          }
        });

    Result vector_128_result;
    {
      VectorHeadRing<128> ring(cfg.ring_capacity, cfg.worker_core);
      vector_128_result =
          run_ring(ring, queue_buffers, queue_runtime, work_items, cfg);
    }

    Result vector_256_result;
    {
      VectorHeadRing<256> ring(cfg.ring_capacity, cfg.worker_core);
      vector_256_result =
          run_ring(ring, queue_buffers, queue_runtime, work_items, cfg);
    }

    using FixedHead =
        FixedHeadRing<256, task_queue_dispatch_bench::kDefaultRingCapacity>;
    Result fixed_head_256_result;
    {
      auto ring = std::make_unique<FixedHead>(cfg.worker_core);
      fixed_head_256_result =
          run_ring(*ring, queue_buffers, queue_runtime, work_items, cfg);
    }

    Result vector_cached_256_result;
    {
      VectorCachedRing<256> ring(cfg.ring_capacity, cfg.worker_core);
      vector_cached_256_result =
          run_ring(ring, queue_buffers, queue_runtime, work_items, cfg);
    }

    using FixedRing =
        FixedCachedRing<256, task_queue_dispatch_bench::kDefaultRingCapacity>;
    Result fixed_cached_256_result;
    {
      auto ring = std::make_unique<FixedRing>(cfg.worker_core);
      fixed_cached_256_result =
          run_ring(*ring, queue_buffers, queue_runtime, work_items, cfg);
    }

    Result fixed_cached_api_result;
    {
      auto ring = std::make_unique<FixedRing>(cfg.worker_core);
      ApiWrapper<FixedRing> api(*ring);
      fixed_cached_api_result =
          run_api(api, queue_buffers, queue_runtime, work_items, cfg);
    }

    std::cout << "tasks: " << cfg.tasks << "\n";
    std::cout << "iterations: " << cfg.iterations << " warmup: " << cfg.warmup
              << "\n";
    std::cout << "producer_core: " << cfg.producer_core
              << " worker_core: " << cfg.worker_core << "\n";
    std::cout << "ring_capacity: " << cfg.ring_capacity << "\n";
    std::cout << "workload: arange, member fill, temperature<float>, "
                 "min_p<float> round-robin\n\n";

    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(30) << "mode" << std::setw(14) << "batch_ns"
              << std::setw(14) << "task_ns" << std::setw(14) << "extra_ns"
              << "\n";
    print_result("direct", direct, direct);
    print_result("vector_slot_128", direct, vector_128_result);
    print_result("vector_slot_256", direct, vector_256_result);
    print_result("fixed_head_256", direct, fixed_head_256_result);
    print_result("vector_cached_256", direct, vector_cached_256_result);
    print_result("fixed_cached_256", direct, fixed_cached_256_result);
    print_result("fixed_cached_256_api", direct, fixed_cached_api_result);

    const uint64_t expected = direct.checksum;
    const bool ok = vector_128_result.checksum == expected &&
        vector_256_result.checksum == expected &&
        fixed_head_256_result.checksum == expected &&
        vector_cached_256_result.checksum == expected &&
        fixed_cached_256_result.checksum == expected &&
        fixed_cached_api_result.checksum == expected;
    std::cout << "\nchecksum: 0x" << std::hex << expected << std::dec << "\n";
    std::cout << "all queue checksums match direct: " << (ok ? "yes" : "no")
              << "\n";
    return ok ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
