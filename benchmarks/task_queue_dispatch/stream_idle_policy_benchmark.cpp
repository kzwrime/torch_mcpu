// Standalone benchmark for openreg stream worker idle policies.
//
// Measures, for the default, high-priority, and low-priority streams:
//   launch: producer-side cost per launch while the worker is awake, and the
//           burst completion cost per task.
//   wake:   launch-to-run latency after an idle gap, the producer-side cost of
//           that launch, and the worker CPU usage while alternating work/gap.
//   idle:   worker CPU usage while the stream stays idle.
//
// Worker CPU time is sampled by tasks running on the worker thread itself via
// CLOCK_THREAD_CPUTIME_ID, so it excludes the producer.
//
// Build from the repository root:
//   g++ -O3 -DNDEBUG -std=c++17 -pthread \
//     -DTORCH_MCPU_ENABLE_MEMORY_PROTECTION=0 \
//     -Ithird_party/openreg \
//     -Ibenchmarks/task_queue_dispatch \
//     benchmarks/task_queue_dispatch/stream_idle_policy_benchmark.cpp \
//     third_party/openreg/csrc/device.cpp \
//     third_party/openreg/csrc/stream.cpp \
//     third_party/openreg/csrc/memory.cpp \
//     -o build/stream_idle_policy_benchmark
//
// Run (keep producer and worker on idle physical cores of one NUMA node):
//   taskset -c 1,3 ./build/stream_idle_policy_benchmark --producer-core 1

#include "benchmark_common.hpp"

#include <include/openreg.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
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
  int wake_samples = 400;
  int idle_ms = 1000;
  int producer_core = -1;
  std::string run;
};

[[noreturn]] void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--tasks N] [--iterations N] [--warmup N]"
               " [--wake-samples N] [--idle-ms N] [--producer-core N]"
               " [--run launch|wake|idle]\n";
  std::exit(2);
}

int parse_positive(const char* value, const char* name) {
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (*value == '\0' || *end != '\0' || parsed <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return static_cast<int>(parsed);
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--tasks" && i + 1 < argc) {
      cfg.tasks = parse_positive(argv[++i], "--tasks");
    } else if (arg == "--iterations" && i + 1 < argc) {
      cfg.iterations = parse_positive(argv[++i], "--iterations");
    } else if (arg == "--warmup" && i + 1 < argc) {
      cfg.warmup = parse_positive(argv[++i], "--warmup");
    } else if (arg == "--wake-samples" && i + 1 < argc) {
      cfg.wake_samples = parse_positive(argv[++i], "--wake-samples");
    } else if (arg == "--idle-ms" && i + 1 < argc) {
      cfg.idle_ms = parse_positive(argv[++i], "--idle-ms");
    } else if (arg == "--producer-core" && i + 1 < argc) {
      cfg.producer_core = parse_positive(argv[++i], "--producer-core");
    } else if (arg == "--run" && i + 1 < argc) {
      cfg.run = argv[++i];
    } else {
      usage(argv[0]);
    }
  }
  return cfg;
}

#if defined(__linux__)
using AffinityMask = cpu_set_t;
#else
using AffinityMask = int;
#endif

AffinityMask current_affinity() {
  AffinityMask mask{};
#if defined(__linux__)
  pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask);
#endif
  return mask;
}

void set_affinity(const AffinityMask& mask) {
#if defined(__linux__)
  pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
#else
  (void)mask;
#endif
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

void check(orError_t status, const char* what) {
  if (status != orSuccess) {
    throw std::runtime_error(std::string(what) + " failed");
  }
}

uint64_t now_ns() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
      .count();
}

uint64_t thread_cpu_ns() {
  timespec ts{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
      static_cast<uint64_t>(ts.tv_nsec);
}

void wait_idle(orStream_t stream) {
  while (!stream->idle()) {
    openreg::cpu_relax();
  }
}

// Waits until `deadline_ns`, sleeping for most of long gaps so the producer
// does not compete with the worker for the core.
void wait_until(uint64_t deadline_ns) {
  constexpr uint64_t kSpinTailNs = 50000;
  uint64_t now = now_ns();
  if (deadline_ns > now + kSpinTailNs) {
    std::this_thread::sleep_for(
        std::chrono::nanoseconds(deadline_ns - now - kSpinTailNs));
  }
  while (now_ns() < deadline_ns) {
    openreg::cpu_relax();
  }
}

struct WorkerClock {
  uint64_t wall_ns = 0;
  uint64_t cpu_ns = 0;
};

// Samples the worker thread's CPU clock from a task running on that worker.
WorkerClock sample_worker_clock(orStream_t stream) {
  WorkerClock sample;
  WorkerClock* out = &sample;
  check(
      orLaunchKernel(
          stream,
          [out] {
            out->wall_ns = now_ns();
            out->cpu_ns = thread_cpu_ns();
          }),
      "orLaunchKernel");
  check(orStreamSynchronize(stream), "orStreamSynchronize");
  return sample;
}

double worker_cpu_percent(const WorkerClock& begin, const WorkerClock& end) {
  return 100.0 * static_cast<double>(end.cpu_ns - begin.cpu_ns) /
      static_cast<double>(end.wall_ns - begin.wall_ns);
}

double percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  const size_t idx = std::min(
      values.size() - 1, static_cast<size_t>(p * (values.size() - 1) + 0.5));
  return values[idx];
}

struct StreamCase {
  const char* name;
  orStream_t stream;
};

const char* policy_name(const orStream* stream) {
  switch (stream->idle_policy) {
    case orStream::IdlePolicy::Busy:
      return "busy";
    case orStream::IdlePolicy::Hybrid:
      return "hybrid";
    case orStream::IdlePolicy::Block:
      return "block";
  }
  return "unknown";
}

void run_launch(const std::vector<StreamCase>& cases, const Config& cfg) {
  std::cout << "[launch] tasks=" << cfg.tasks
            << " iterations=" << cfg.iterations << "\n";
  std::cout << std::setw(10) << "stream" << std::setw(10) << "policy"
            << std::setw(18) << "launch_ns/task" << std::setw(18)
            << "p99_launch_ns" << std::setw(18) << "burst_ns/task" << "\n";
  for (const auto& c : cases) {
    std::vector<double> launch_per_task;
    std::vector<double> burst_per_task;
    for (int iter = 0; iter < cfg.warmup + cfg.iterations; ++iter) {
      const uint64_t start = now_ns();
      for (int i = 0; i < cfg.tasks; ++i) {
        check(orLaunchKernel(c.stream, [] {}), "orLaunchKernel");
      }
      const uint64_t launched = now_ns();
      wait_idle(c.stream);
      const uint64_t done = now_ns();
      if (iter >= cfg.warmup) {
        launch_per_task.push_back(
            static_cast<double>(launched - start) / cfg.tasks);
        burst_per_task.push_back(static_cast<double>(done - start) / cfg.tasks);
      }
    }
    std::cout << std::fixed << std::setprecision(1) << std::setw(10) << c.name
              << std::setw(10) << policy_name(c.stream) << std::setw(18)
              << percentile(launch_per_task, 0.5) << std::setw(18)
              << percentile(launch_per_task, 0.99) << std::setw(18)
              << percentile(burst_per_task, 0.5) << "\n";
  }
}

void run_wake(const std::vector<StreamCase>& cases, const Config& cfg) {
  const std::vector<uint64_t> gaps_us = {
      0, 5, 20, 50, 100, 200, 500, 1000, 5000};
  std::cout << "[wake] samples=" << cfg.wake_samples
            << " (launch-to-run latency after an idle gap)\n";
  std::cout << std::setw(10) << "stream" << std::setw(10) << "policy"
            << std::setw(10) << "gap_us" << std::setw(14) << "p50_wake_ns"
            << std::setw(14) << "p99_wake_ns" << std::setw(16)
            << "p50_launch_ns" << std::setw(16) << "p99_launch_ns"
            << std::setw(14) << "worker_cpu%" << "\n";
  for (const auto& c : cases) {
    for (uint64_t gap_us : gaps_us) {
      // Bound each gap's runtime to about 0.5 s.
      const int samples = std::max<int>(
          20,
          std::min<int>(
              cfg.wake_samples,
              static_cast<int>(500000 / std::max<uint64_t>(gap_us, 1))));
      std::vector<double> wake_ns;
      std::vector<double> launch_ns;
      std::atomic<uint64_t> ran_at{0};
      std::atomic<uint64_t>* ran_ptr = &ran_at;
      const WorkerClock begin = sample_worker_clock(c.stream);
      uint64_t last_done = begin.wall_ns;
      for (int s = 0; s < samples; ++s) {
        wait_until(last_done + gap_us * 1000);
        const uint64_t t0 = now_ns();
        check(
            orLaunchKernel(
                c.stream,
                [ran_ptr] {
                  ran_ptr->store(now_ns(), std::memory_order_release);
                }),
            "orLaunchKernel");
        const uint64_t t1 = now_ns();
        wait_idle(c.stream);
        const uint64_t ran = ran_at.load(std::memory_order_acquire);
        wake_ns.push_back(static_cast<double>(ran - t0));
        launch_ns.push_back(static_cast<double>(t1 - t0));
        last_done = ran;
      }
      const WorkerClock end = sample_worker_clock(c.stream);
      std::cout << std::fixed << std::setprecision(0) << std::setw(10) << c.name
                << std::setw(10) << policy_name(c.stream) << std::setw(10)
                << gap_us << std::setw(14) << percentile(wake_ns, 0.5)
                << std::setw(14) << percentile(wake_ns, 0.99) << std::setw(16)
                << percentile(launch_ns, 0.5) << std::setw(16)
                << percentile(launch_ns, 0.99) << std::setprecision(1)
                << std::setw(14) << worker_cpu_percent(begin, end) << "\n";
    }
  }
}

void run_idle(const std::vector<StreamCase>& cases, const Config& cfg) {
  std::cout << "[idle] idle_ms=" << cfg.idle_ms << "\n";
  std::cout << std::setw(10) << "stream" << std::setw(10) << "policy"
            << std::setw(14) << "worker_cpu%" << "\n";
  for (const auto& c : cases) {
    const WorkerClock begin = sample_worker_clock(c.stream);
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.idle_ms));
    const WorkerClock end = sample_worker_clock(c.stream);
    std::cout << std::fixed << std::setprecision(1) << std::setw(10) << c.name
              << std::setw(10) << policy_name(c.stream) << std::setw(14)
              << worker_cpu_percent(begin, end) << "\n";
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    // Stream workers inherit the creating thread's affinity, so the producer
    // is pinned only after each stream exists.
    const AffinityMask process_affinity = current_affinity();

    int least_priority = 0;
    int greatest_priority = 0;
    check(
        orDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
        "orDeviceGetStreamPriorityRange");

    struct CaseSpec {
      const char* name;
      unsigned int flags;
      int priority;
    };
    // Each stream runs alone so always-spinning workers of other cases do not
    // compete for the cores under measurement.
    const std::vector<CaseSpec> specs = {
        {"default", orStreamDefault, least_priority},
        {"high", 0, greatest_priority},
        {"low", 0, least_priority},
    };

    std::cout << "priority_range: least=" << least_priority
              << " greatest=" << greatest_priority << "\n";
    std::cout << "producer_core: " << cfg.producer_core << "\n";

    for (const auto& spec : specs) {
      orStream_t stream = nullptr;
      set_affinity(process_affinity);
      check(
          orStreamCreateWithPriority(&stream, spec.flags, spec.priority),
          "orStreamCreateWithPriority");
      pin_current_thread(cfg.producer_core);
      const std::vector<StreamCase> cases = {{spec.name, stream}};
      std::cout << "\n== " << spec.name << " (" << policy_name(stream) << ")\n";
      if (cfg.run.empty() || cfg.run == "launch") {
        run_launch(cases, cfg);
      }
      if (cfg.run.empty() || cfg.run == "wake") {
        run_wake(cases, cfg);
      }
      if (cfg.run.empty() || cfg.run == "idle") {
        run_idle(cases, cfg);
      }
      check(orStreamDestroy(stream), "orStreamDestroy");
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
