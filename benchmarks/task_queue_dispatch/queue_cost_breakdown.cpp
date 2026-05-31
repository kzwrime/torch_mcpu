// Standalone cost breakdown for stream task dispatch.
//
// Build from the repository root:
//   g++ -O3 -DNDEBUG -std=c++17 -pthread \
//     -Ibenchmarks/task_queue_dispatch \
//     benchmarks/task_queue_dispatch/queue_cost_breakdown.cpp \
//     -o build/queue_cost_breakdown
//
// Run:
//   ./build/queue_cost_breakdown --tasks 500 --iterations 2000 --pin-core 75

#include "benchmark_common.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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

struct Payload {
  uint64_t seed = 0;
  uint64_t* out = nullptr;
};

struct Result {
  double batch_ns = 0.0;
  double task_ns = 0.0;
  uint64_t checksum = 0;
};

using Clock = std::chrono::steady_clock;

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
__attribute__((noinline))
#endif
void
lightweight_task(Payload* payload) {
  uint64_t x = payload->seed;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  payload->out[0] = x + 0x9e3779b97f4a7c15ull;
}

void typed_task_trampoline(void* ctx) {
  lightweight_task(static_cast<Payload*>(ctx));
}

void refresh_payloads(std::vector<Payload>& payloads, uint64_t iteration) {
  for (size_t i = 0; i < payloads.size(); ++i) {
    payloads[i].seed =
        (iteration + 1) * 0x9e3779b97f4a7c15ull + static_cast<uint64_t>(i);
  }
}

uint64_t consume_outputs(const std::vector<uint64_t>& outputs) {
  uint64_t checksum = 0;
  for (uint64_t value : outputs) {
    checksum ^=
        value + 0x9e3779b97f4a7c15ull + (checksum << 6) + (checksum >> 2);
  }
  return checksum;
}

template <typename Body>
Result run_loop(
    std::vector<Payload>& payloads,
    const std::vector<uint64_t>& outputs,
    const Config& cfg,
    Body&& body) {
  uint64_t total_ns = 0;
  uint64_t checksum = 0;
  const int total_iterations = cfg.warmup + cfg.iterations;

  for (int iter = 0; iter < total_iterations; ++iter) {
    refresh_payloads(payloads, static_cast<uint64_t>(iter));

    const uint64_t start = now_ns();
    body(payloads);
    const uint64_t elapsed = now_ns() - start;

    checksum ^= consume_outputs(outputs);
    if (iter >= cfg.warmup) {
      total_ns += elapsed;
    }
  }

  const double batch_ns = static_cast<double>(total_ns) / cfg.iterations;
  return {batch_ns, batch_ns / cfg.tasks, checksum};
}

class OriginalLikeQueue {
 public:
  explicit OriginalLikeQueue(int pin_core)
      : worker_([this, pin_core] {
          pin_current_thread(pin_core);
          worker_loop();
        }) {}

  ~OriginalLikeQueue() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      stop_ = true;
    }
    cv_.notify_one();
    worker_.join();
  }

  void enqueue(std::function<void()> task) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      tasks_.push_back(std::move(task));
    }
    cv_.notify_one();
  }

  void synchronize() {
    auto done = std::make_shared<SyncFlag>();
    enqueue([done] {
      {
        std::lock_guard<std::mutex> lock(done->mtx);
        done->complete = true;
      }
      done->cv.notify_one();
    });

    std::unique_lock<std::mutex> lock(done->mtx);
    done->cv.wait(lock, [&] { return done->complete; });
  }

 private:
  struct SyncFlag {
    std::mutex mtx;
    std::condition_variable cv;
    bool complete = false;
  };

  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();
    }
  }

  std::deque<std::function<void()>> tasks_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::thread worker_;
};

class SequenceSyncQueue {
 public:
  explicit SequenceSyncQueue(int pin_core, bool notify_only_empty)
      : notify_only_empty_(notify_only_empty), worker_([this, pin_core] {
          pin_current_thread(pin_core);
          worker_loop();
        }) {}

  ~SequenceSyncQueue() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      stop_ = true;
    }
    cv_.notify_one();
    worker_.join();
  }

  void enqueue(std::function<void()> task) {
    bool was_empty = false;
    {
      std::lock_guard<std::mutex> lock(mtx_);
      was_empty = tasks_.empty();
      tasks_.push_back(std::move(task));
      submitted_.fetch_add(1, std::memory_order_release);
    }
    if (!notify_only_empty_ || was_empty) {
      cv_.notify_one();
    }
  }

  void synchronize() {
    const uint64_t target = submitted_.load(std::memory_order_acquire);
    const uint64_t deadline = now_ns() + 5'000'000'000ull;
    while (completed_.load(std::memory_order_acquire) < target) {
      if (now_ns() > deadline) {
        std::lock_guard<std::mutex> queue_lock(mtx_);
        std::cerr << "sequence sync timeout: submitted="
                  << submitted_.load(std::memory_order_acquire)
                  << " completed=" << completed_.load(std::memory_order_acquire)
                  << " target=" << target << " queue=" << tasks_.size() << "\n";
        std::abort();
      }
      std::this_thread::yield();
    }
  }

 private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;
      uint64_t seq = 0;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop_front();
        seq = next_to_complete_++;
      }
      task();
      completed_.store(seq, std::memory_order_release);
    }
  }

  std::deque<std::function<void()>> tasks_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<uint64_t> submitted_{0};
  std::atomic<uint64_t> completed_{0};
  uint64_t next_to_complete_ = 1;
  bool notify_only_empty_ = false;
  bool stop_ = false;
  std::thread worker_;
};

class SpscFunctionRingQueue {
 public:
  explicit SpscFunctionRingQueue(size_t capacity, int pin_core)
      : tasks_(next_power_of_two(capacity)),
        mask_(tasks_.size() - 1),
        worker_([this, pin_core] {
          pin_current_thread(pin_core);
          worker_loop();
        }) {}

  ~SpscFunctionRingQueue() {
    stop_.store(true, std::memory_order_release);
    worker_.join();
  }

  void enqueue(std::function<void()> task) {
    uint64_t tail = tail_.load(std::memory_order_relaxed);
    while (tail - head_.load(std::memory_order_acquire) >= tasks_.size()) {
      std::this_thread::yield();
    }
    const bool was_empty = tail == head_.load(std::memory_order_acquire);
    tasks_[tail & mask_] = std::move(task);
    tail_.store(tail + 1, std::memory_order_release);
    (void)was_empty;
  }

  void synchronize() {
    const uint64_t target = tail_.load(std::memory_order_acquire);
    const uint64_t deadline = now_ns() + 5'000'000'000ull;
    while (completed_.load(std::memory_order_acquire) < target) {
      if (now_ns() > deadline) {
        std::cerr << "spsc function sync timeout: tail="
                  << tail_.load(std::memory_order_acquire)
                  << " head=" << head_.load(std::memory_order_acquire)
                  << " completed=" << completed_.load(std::memory_order_acquire)
                  << " target=" << target << "\n";
        std::abort();
      }
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

struct TypedTask {
  void (*run)(void*) = nullptr;
  void* ctx = nullptr;
};

class SpscTypedRingQueue {
 public:
  explicit SpscTypedRingQueue(size_t capacity, int pin_core)
      : tasks_(next_power_of_two(capacity)),
        mask_(tasks_.size() - 1),
        worker_([this, pin_core] {
          pin_current_thread(pin_core);
          worker_loop();
        }) {}

  ~SpscTypedRingQueue() {
    stop_.store(true, std::memory_order_release);
    worker_.join();
  }

  void enqueue(TypedTask task) {
    uint64_t tail = tail_.load(std::memory_order_relaxed);
    while (tail - head_.load(std::memory_order_acquire) >= tasks_.size()) {
      std::this_thread::yield();
    }
    const bool was_empty = tail == head_.load(std::memory_order_acquire);
    tasks_[tail & mask_] = task;
    tail_.store(tail + 1, std::memory_order_release);
    (void)was_empty;
  }

  void synchronize() {
    const uint64_t target = tail_.load(std::memory_order_acquire);
    const uint64_t deadline = now_ns() + 5'000'000'000ull;
    while (completed_.load(std::memory_order_acquire) < target) {
      if (now_ns() > deadline) {
        std::cerr << "spsc typed sync timeout: tail="
                  << tail_.load(std::memory_order_acquire)
                  << " head=" << head_.load(std::memory_order_acquire)
                  << " completed=" << completed_.load(std::memory_order_acquire)
                  << " target=" << target << "\n";
        std::abort();
      }
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

      TypedTask task = tasks_[head & mask_];
      head_.store(head + 1, std::memory_order_release);
      task.run(task.ctx);
      completed_.store(head + 1, std::memory_order_release);
    }
  }

  std::vector<TypedTask> tasks_;
  const uint64_t mask_;
  alignas(64) std::atomic<uint64_t> head_{0};
  alignas(64) std::atomic<uint64_t> tail_{0};
  alignas(64) std::atomic<uint64_t> completed_{0};
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

void print_result(const char* name, const Result& result, double baseline) {
  std::cout << std::setw(24) << name << std::setw(16) << result.batch_ns
            << std::setw(16) << result.task_ns << std::setw(16)
            << (result.task_ns - baseline) << "\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    std::vector<uint64_t> outputs(cfg.tasks, 0);
    std::vector<Payload> payloads(cfg.tasks);
    for (int i = 0; i < cfg.tasks; ++i) {
      payloads[i].out = &outputs[i];
    }

    const Result direct = run_loop(payloads, outputs, cfg, [](auto& tasks) {
      for (Payload& payload : tasks) {
        lightweight_task(&payload);
      }
    });

    const Result function_direct =
        run_loop(payloads, outputs, cfg, [](auto& tasks) {
          for (Payload& payload : tasks) {
            Payload* payload_ptr = &payload;
            std::function<void()> task = [payload_ptr] {
              lightweight_task(payload_ptr);
            };
            task();
          }
        });

    Result original_result;
    {
      OriginalLikeQueue original(cfg.pin_core);
      original_result =
          run_loop(payloads, outputs, cfg, [&original](auto& tasks) {
            for (Payload& payload : tasks) {
              Payload* payload_ptr = &payload;
              original.enqueue(
                  [payload_ptr] { lightweight_task(payload_ptr); });
            }
            original.synchronize();
          });
    }

    Result seq_sync_result;
    {
      SequenceSyncQueue seq_sync(cfg.pin_core, false);
      seq_sync_result =
          run_loop(payloads, outputs, cfg, [&seq_sync](auto& tasks) {
            for (Payload& payload : tasks) {
              Payload* payload_ptr = &payload;
              seq_sync.enqueue(
                  [payload_ptr] { lightweight_task(payload_ptr); });
            }
            seq_sync.synchronize();
          });
    }

    Result notify_empty_result;
    {
      SequenceSyncQueue notify_empty(cfg.pin_core, true);
      notify_empty_result =
          run_loop(payloads, outputs, cfg, [&notify_empty](auto& tasks) {
            for (Payload& payload : tasks) {
              Payload* payload_ptr = &payload;
              notify_empty.enqueue(
                  [payload_ptr] { lightweight_task(payload_ptr); });
            }
            notify_empty.synchronize();
          });
    }

    Result spsc_function_result;
    {
      SpscFunctionRingQueue spsc_function(cfg.ring_capacity, cfg.pin_core);
      spsc_function_result =
          run_loop(payloads, outputs, cfg, [&spsc_function](auto& tasks) {
            for (Payload& payload : tasks) {
              Payload* payload_ptr = &payload;
              spsc_function.enqueue(
                  [payload_ptr] { lightweight_task(payload_ptr); });
            }
            spsc_function.synchronize();
          });
    }

    Result spsc_typed_result;
    {
      SpscTypedRingQueue spsc_typed(cfg.ring_capacity, cfg.pin_core);
      spsc_typed_result =
          run_loop(payloads, outputs, cfg, [&spsc_typed](auto& tasks) {
            for (Payload& payload : tasks) {
              spsc_typed.enqueue({typed_task_trampoline, &payload});
            }
            spsc_typed.synchronize();
          });
    }

    const double direct_task_ns = direct.task_ns;

    std::cout << "tasks: " << cfg.tasks << "\n";
    std::cout << "iterations: " << cfg.iterations << " warmup: " << cfg.warmup
              << "\n";
    std::cout << "pin_core: " << cfg.pin_core
              << " ring_capacity: " << next_power_of_two(cfg.ring_capacity)
              << "\n\n";

    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(24) << "mode" << std::setw(16) << "batch_ns"
              << std::setw(16) << "task_ns" << std::setw(16) << "extra_ns"
              << "\n";
    print_result("direct", direct, direct_task_ns);
    print_result("std_function_direct", function_direct, direct_task_ns);
    print_result("original_like", original_result, direct_task_ns);
    print_result("sequence_sync", seq_sync_result, direct_task_ns);
    print_result("notify_empty", notify_empty_result, direct_task_ns);
    print_result("spsc_function_ring", spsc_function_result, direct_task_ns);
    print_result("spsc_typed_ring", spsc_typed_result, direct_task_ns);

    std::cout << "\nchecksum: 0x" << std::hex << direct.checksum << std::dec
              << "\n";
    std::cout << "all queue checksums match direct: "
              << ((function_direct.checksum == direct.checksum &&
                   original_result.checksum == direct.checksum &&
                   seq_sync_result.checksum == direct.checksum &&
                   notify_empty_result.checksum == direct.checksum &&
                   spsc_function_result.checksum == direct.checksum &&
                   spsc_typed_result.checksum == direct.checksum)
                      ? "yes"
                      : "no")
              << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
