// Standalone benchmark for the openreg stream task queue overhead.
//
// Build from the repository root:
//   g++ -O3 -DNDEBUG -std=c++17 -pthread \
//     -DTORCH_MCPU_ENABLE_MEMORY_PROTECTION=0 \
//     -Ithird_party/openreg \
//     -Ibenchmarks/task_queue_dispatch \
//     benchmarks/task_queue_dispatch/openreg_queue_overhead.cpp \
//     third_party/openreg/csrc/device.cpp \
//     third_party/openreg/csrc/stream.cpp \
//     third_party/openreg/csrc/memory.cpp \
//     -o build/openreg_queue_overhead
//
// Run:
//   ./build/openreg_queue_overhead --tasks 500 --iterations 2000

#include "benchmark_common.hpp"

#include <include/openreg.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
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
  int producer_core = -1;
};

struct Payload {
  uint64_t seed;
  uint64_t* out;
};

struct Result {
  double batch_ns = 0.0;
  double task_ns = 0.0;
  uint64_t checksum = 0;
};

[[noreturn]] void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--tasks N] [--iterations N] [--warmup N]"
               " [--producer-core N]\n";
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
    } else if (arg == "--producer-core" && i + 1 < argc) {
      cfg.producer_core = parse_positive(argv[++i], "--producer-core");
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
    } else {
      usage(argv[0]);
    }
  }
  return cfg;
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

uint64_t consume_outputs(const std::vector<uint64_t>& outputs) {
  uint64_t checksum = 0;
  for (uint64_t value : outputs) {
    checksum ^=
        value + 0x9e3779b97f4a7c15ull + (checksum << 6) + (checksum >> 2);
  }
  return checksum;
}

void refresh_payloads(std::vector<Payload>& payloads, uint64_t iteration) {
  for (size_t i = 0; i < payloads.size(); ++i) {
    payloads[i].seed =
        (iteration + 1) * 0x9e3779b97f4a7c15ull + static_cast<uint64_t>(i);
  }
}

Result run_direct(
    std::vector<Payload>& payloads,
    const std::vector<uint64_t>& outputs,
    const Config& cfg) {
  uint64_t total_ns = 0;
  uint64_t checksum = 0;
  const int total_iterations = cfg.warmup + cfg.iterations;

  for (int iter = 0; iter < total_iterations; ++iter) {
    refresh_payloads(payloads, static_cast<uint64_t>(iter));

    const uint64_t start = now_ns();
    for (Payload& payload : payloads) {
      lightweight_task(&payload);
    }
    const uint64_t elapsed = now_ns() - start;

    checksum ^= consume_outputs(outputs);
    if (iter >= cfg.warmup) {
      total_ns += elapsed;
    }
  }

  const double batch_ns = static_cast<double>(total_ns) / cfg.iterations;
  return {batch_ns, batch_ns / cfg.tasks, checksum};
}

Result run_queue(
    orStream_t stream,
    std::vector<Payload>& payloads,
    const std::vector<uint64_t>& outputs,
    const Config& cfg,
    bool wrap_in_lambda) {
  uint64_t total_ns = 0;
  uint64_t checksum = 0;
  const int total_iterations = cfg.warmup + cfg.iterations;

  for (int iter = 0; iter < total_iterations; ++iter) {
    refresh_payloads(payloads, static_cast<uint64_t>(iter));

    const uint64_t start = now_ns();
    for (Payload& payload : payloads) {
      Payload* payload_ptr = &payload;
      if (wrap_in_lambda) {
        check(
            orLaunchKernel(
                stream, [payload_ptr] { lightweight_task(payload_ptr); }),
            "orLaunchKernel");
      } else {
        check(
            orLaunchKernel(stream, lightweight_task, payload_ptr),
            "orLaunchKernel");
      }
    }
    check(orStreamSynchronize(stream), "orStreamSynchronize");
    const uint64_t elapsed = now_ns() - start;

    checksum ^= consume_outputs(outputs);
    if (iter >= cfg.warmup) {
      total_ns += elapsed;
    }
  }

  const double batch_ns = static_cast<double>(total_ns) / cfg.iterations;
  return {batch_ns, batch_ns / cfg.tasks, checksum};
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    pin_current_thread(cfg.producer_core);

    std::vector<uint64_t> direct_outputs(cfg.tasks, 0);
    std::vector<uint64_t> queue_outputs(cfg.tasks, 0);
    std::vector<Payload> direct_payloads(cfg.tasks);
    std::vector<Payload> queue_payloads(cfg.tasks);
    for (int i = 0; i < cfg.tasks; ++i) {
      direct_payloads[i].out = &direct_outputs[i];
      queue_payloads[i].out = &queue_outputs[i];
    }

    orStream_t stream = nullptr;
    check(orStreamCreate(&stream), "orStreamCreate");

    const Result direct = run_direct(direct_payloads, direct_outputs, cfg);
    const Result queued_func_args =
        run_queue(stream, queue_payloads, queue_outputs, cfg, false);
    const Result queued_lambda =
        run_queue(stream, queue_payloads, queue_outputs, cfg, true);

    check(orStreamDestroy(stream), "orStreamDestroy");

    std::cout << "tasks: " << cfg.tasks << "\n";
    std::cout << "iterations: " << cfg.iterations << " warmup: " << cfg.warmup
              << "\n";
    std::cout << "producer_core: " << cfg.producer_core << "\n";
    std::cout << "memory_protection: off\n\n";

    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(12) << "mode" << std::setw(18) << "batch_ns"
              << std::setw(18) << "per_task_ns" << "\n";
    std::cout << std::setw(12) << "direct" << std::setw(18) << direct.batch_ns
              << std::setw(18) << direct.task_ns << "\n";
    std::cout << std::setw(12) << "func_args" << std::setw(18)
              << queued_func_args.batch_ns << std::setw(18)
              << queued_func_args.task_ns << "\n";
    std::cout << std::setw(12) << "lambda" << std::setw(18)
              << queued_lambda.batch_ns << std::setw(18)
              << queued_lambda.task_ns << "\n\n";

    std::cout << std::setprecision(2);
    std::cout << "func_args/direct batch ratio: "
              << queued_func_args.batch_ns / direct.batch_ns << "x\n";
    std::cout << "func_args extra per task: "
              << queued_func_args.task_ns - direct.task_ns << " ns\n";
    std::cout << "lambda/direct batch ratio: "
              << queued_lambda.batch_ns / direct.batch_ns << "x\n";
    std::cout << "lambda extra per task: "
              << queued_lambda.task_ns - direct.task_ns << " ns\n";
    std::cout << std::hex;
    std::cout << "direct checksum: 0x" << direct.checksum << "\n";
    std::cout << "func_args checksum: 0x" << queued_func_args.checksum << "\n";
    std::cout << "lambda checksum: 0x" << queued_lambda.checksum << "\n";
    if (direct.checksum != queued_func_args.checksum ||
        direct.checksum != queued_lambda.checksum) {
      std::cerr << "checksum mismatch\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
}
