// Standalone benchmark for pointer-only kernel launch shapes.
//
// The kernel body intentionally looks like a tiny vLLM-style CPU kernel: the
// launch API receives raw pointers and POD sizes, not Tensor objects.  This
// isolates OpenReg dispatch overhead from ATen/Tensor lifetime costs.

#include "benchmark_common.hpp"

#include <include/openreg.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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
  int producer_core = -1;
  int vocab = 1;
  int worker_delay_us = 1000;
};

struct PointerKernelArgs {
  float* logits;
  const int32_t* request_index;
  const float* temperature;
  int64_t token;
  int64_t logits_stride;
  int64_t vocab_size;
};

struct EmptyKernelArgs {
  uint64_t* out;
  uint64_t value;
};

struct Result {
  double batch_ns = 0.0;
  double task_ns = 0.0;
  double worker_span_ns = 0.0;
  double worker_task_ns = 0.0;
  double extra_task_ns = 0.0;
  uint64_t checksum = 0;
};

struct WorkerSpan {
  std::atomic<uint64_t> first_begin_ns{
      std::numeric_limits<uint64_t>::max()};
  std::atomic<uint64_t> last_end_ns{0};

  void reset() {
    first_begin_ns.store(
        std::numeric_limits<uint64_t>::max(), std::memory_order_relaxed);
    last_end_ns.store(0, std::memory_order_relaxed);
  }
};

[[noreturn]] void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--tasks N] [--iterations N] [--warmup N]"
               " [--producer-core N] [--vocab N] [--worker-delay-us N]\n";
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
    } else if (arg == "--vocab" && i + 1 < argc) {
      cfg.vocab = parse_positive(argv[++i], "--vocab");
    } else if (arg == "--worker-delay-us" && i + 1 < argc) {
      cfg.worker_delay_us = parse_positive(argv[++i], "--worker-delay-us");
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

void worker_delay_kernel(int64_t delay_us) {
  std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
}

void worker_span_start_kernel(WorkerSpan* span) {
  span->first_begin_ns.store(now_ns(), std::memory_order_relaxed);
}

void worker_span_end_kernel(WorkerSpan* span) {
  span->last_end_ns.store(now_ns(), std::memory_order_release);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void
pointer_temperature_kernel(
    float* logits,
    const int32_t* request_index,
    const float* temperature,
    int64_t token,
    int64_t logits_stride,
    int64_t vocab_size) {
  const float temp = temperature[request_index[token]];
  float* row = logits + token * logits_stride;
  for (int64_t i = 0; i < vocab_size; ++i) {
    row[i] = row[i] / temp + 0.001f;
  }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void
pointer_temperature_kernel_args(PointerKernelArgs* args) {
  pointer_temperature_kernel(
      args->logits,
      args->request_index,
      args->temperature,
      args->token,
      args->logits_stride,
      args->vocab_size);
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void
empty_pointer_kernel(uint64_t* out, uint64_t value) {
  *out = value + 0x9e3779b97f4a7c15ull;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void
empty_pointer_kernel_args(EmptyKernelArgs* args) {
  *args->out = args->value + 0x9e3779b97f4a7c15ull;
}

void refresh_inputs(
    std::vector<float>& logits,
    std::vector<int32_t>& request_index,
    std::vector<float>& temperature,
    int tasks,
    int vocab,
    uint64_t iteration) {
  for (int task = 0; task < tasks; ++task) {
    request_index[task] = task & 1;
    for (int i = 0; i < vocab; ++i) {
      logits[static_cast<size_t>(task) * vocab + i] =
          static_cast<float>((iteration + 1) * 0.01 + task * 0.001 + i);
    }
  }
  temperature[0] = 0.75f;
  temperature[1] = 1.25f;
}

uint64_t consume_outputs(const std::vector<float>& logits) {
  uint64_t checksum = 0;
  for (float value : logits) {
    const auto bits = static_cast<uint64_t>(value * 1000003.0f);
    checksum ^=
        bits + 0x9e3779b97f4a7c15ull + (checksum << 6) + (checksum >> 2);
  }
  return checksum;
}

Result run_direct(
    std::vector<float>& logits,
    std::vector<int32_t>& request_index,
    std::vector<float>& temperature,
    const Config& cfg,
    bool use_args_struct) {
  uint64_t total_ns = 0;
  uint64_t checksum = 0;
  std::vector<PointerKernelArgs> args(static_cast<size_t>(cfg.tasks));
  const int total_iterations = cfg.warmup + cfg.iterations;

  for (int iter = 0; iter < total_iterations; ++iter) {
    refresh_inputs(logits, request_index, temperature, cfg.tasks, cfg.vocab, iter);
    for (int task = 0; task < cfg.tasks; ++task) {
      args[task] = {
          logits.data(),
          request_index.data(),
          temperature.data(),
          task,
          cfg.vocab,
          cfg.vocab};
    }

    const uint64_t start = now_ns();
    for (int task = 0; task < cfg.tasks; ++task) {
      if (use_args_struct) {
        pointer_temperature_kernel_args(&args[task]);
      } else {
        pointer_temperature_kernel(
            logits.data(),
            request_index.data(),
            temperature.data(),
            task,
            cfg.vocab,
            cfg.vocab);
      }
    }
    const uint64_t elapsed = now_ns() - start;

    checksum ^= consume_outputs(logits);
    if (iter >= cfg.warmup) {
      total_ns += elapsed;
    }
  }

  const double batch_ns = static_cast<double>(total_ns) / cfg.iterations;
  return {batch_ns, batch_ns / cfg.tasks, 0.0, 0.0, 0.0, checksum};
}

Result run_queue(
    orStream_t stream,
    std::vector<float>& logits,
    std::vector<int32_t>& request_index,
    std::vector<float>& temperature,
    const Config& cfg,
    const Result& direct_baseline,
    int mode) {
  uint64_t total_ns = 0;
  uint64_t total_worker_span_ns = 0;
  uint64_t checksum = 0;
  std::vector<PointerKernelArgs> args(static_cast<size_t>(cfg.tasks));
  WorkerSpan worker_span;
  const int total_iterations = cfg.warmup + cfg.iterations;

  for (int iter = 0; iter < total_iterations; ++iter) {
    refresh_inputs(logits, request_index, temperature, cfg.tasks, cfg.vocab, iter);
    for (int task = 0; task < cfg.tasks; ++task) {
      args[task] = {
          logits.data(),
          request_index.data(),
          temperature.data(),
          task,
          cfg.vocab,
          cfg.vocab};
    }

    worker_span.reset();
    const uint64_t start = now_ns();
    check(
        orLaunchKernel(
            stream,
            worker_delay_kernel,
            static_cast<int64_t>(cfg.worker_delay_us)),
        "orLaunchKernel");
    check(
        orLaunchKernel(stream, worker_span_start_kernel, &worker_span),
        "orLaunchKernel");
    for (int task = 0; task < cfg.tasks; ++task) {
      if (mode == 0) {
        check(
            orLaunchKernel(
                stream,
                pointer_temperature_kernel,
                logits.data(),
                request_index.data(),
                temperature.data(),
                static_cast<int64_t>(task),
                static_cast<int64_t>(cfg.vocab),
                static_cast<int64_t>(cfg.vocab)),
            "orLaunchKernel");
      } else if (mode == 1) {
        check(
            orLaunchKernel(
                stream, pointer_temperature_kernel_args, &args[task]),
            "orLaunchKernel");
      } else if (mode == 2) {
        PointerKernelArgs* arg = &args[task];
        check(
            orLaunchKernel(
                stream, [arg] { pointer_temperature_kernel_args(arg); }),
            "orLaunchKernel");
      } else {
        throw std::invalid_argument("unknown queue mode");
      }
    }
    check(
        orLaunchKernel(stream, worker_span_end_kernel, &worker_span),
        "orLaunchKernel");
    check(orStreamSynchronize(stream), "orStreamSynchronize");
    const uint64_t elapsed = now_ns() - start;

    checksum ^= consume_outputs(logits);
    if (iter >= cfg.warmup) {
      total_ns += elapsed;
      const uint64_t first =
          worker_span.first_begin_ns.load(std::memory_order_acquire);
      const uint64_t last = worker_span.last_end_ns.load(std::memory_order_acquire);
      if (first != std::numeric_limits<uint64_t>::max() && last >= first) {
        total_worker_span_ns += last - first;
      }
    }
  }

  const double batch_ns = static_cast<double>(total_ns) / cfg.iterations;
  const double task_ns = batch_ns / cfg.tasks;
  const double worker_span_ns =
      static_cast<double>(total_worker_span_ns) / cfg.iterations;
  const double worker_task_ns = worker_span_ns / cfg.tasks;
  return {
      batch_ns,
      task_ns,
      worker_span_ns,
      worker_task_ns,
      task_ns - direct_baseline.task_ns,
      checksum};
}

Result run_empty_queue(
    orStream_t stream,
    const Config& cfg,
    int mode) {
  uint64_t total_ns = 0;
  uint64_t total_worker_span_ns = 0;
  uint64_t checksum = 0;
  std::vector<uint64_t> outputs(static_cast<size_t>(cfg.tasks), 0);
  std::vector<EmptyKernelArgs> args(static_cast<size_t>(cfg.tasks));
  WorkerSpan worker_span;
  const int total_iterations = cfg.warmup + cfg.iterations;

  for (int iter = 0; iter < total_iterations; ++iter) {
    for (int task = 0; task < cfg.tasks; ++task) {
      args[task] = {&outputs[task], static_cast<uint64_t>(iter * cfg.tasks + task)};
    }

    worker_span.reset();
    const uint64_t start = now_ns();
    check(
        orLaunchKernel(
            stream,
            worker_delay_kernel,
            static_cast<int64_t>(cfg.worker_delay_us)),
        "orLaunchKernel");
    check(
        orLaunchKernel(stream, worker_span_start_kernel, &worker_span),
        "orLaunchKernel");
    for (int task = 0; task < cfg.tasks; ++task) {
      if (mode == 0) {
        check(
            orLaunchKernel(
                stream,
                empty_pointer_kernel,
                &outputs[task],
                static_cast<uint64_t>(iter * cfg.tasks + task)),
            "orLaunchKernel");
      } else if (mode == 1) {
        check(
            orLaunchKernel(stream, empty_pointer_kernel_args, &args[task]),
            "orLaunchKernel");
      } else if (mode == 2) {
        EmptyKernelArgs* arg = &args[task];
        check(
            orLaunchKernel(
                stream, [arg] { empty_pointer_kernel_args(arg); }),
            "orLaunchKernel");
      } else {
        throw std::invalid_argument("unknown empty queue mode");
      }
    }
    check(
        orLaunchKernel(stream, worker_span_end_kernel, &worker_span),
        "orLaunchKernel");
    check(orStreamSynchronize(stream), "orStreamSynchronize");
    const uint64_t elapsed = now_ns() - start;

    for (uint64_t value : outputs) {
      checksum ^=
          value + 0x9e3779b97f4a7c15ull + (checksum << 6) + (checksum >> 2);
    }
    if (iter >= cfg.warmup) {
      total_ns += elapsed;
      const uint64_t first =
          worker_span.first_begin_ns.load(std::memory_order_acquire);
      const uint64_t last = worker_span.last_end_ns.load(std::memory_order_acquire);
      if (first != std::numeric_limits<uint64_t>::max() && last >= first) {
        total_worker_span_ns += last - first;
      }
    }
  }

  const double batch_ns = static_cast<double>(total_ns) / cfg.iterations;
  const double worker_span_ns =
      static_cast<double>(total_worker_span_ns) / cfg.iterations;
  return {
      batch_ns,
      batch_ns / cfg.tasks,
      worker_span_ns,
      worker_span_ns / cfg.tasks,
      0.0,
      checksum};
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    pin_current_thread(cfg.producer_core);

    std::vector<float> direct_logits(static_cast<size_t>(cfg.tasks) * cfg.vocab);
    std::vector<float> queue_logits(static_cast<size_t>(cfg.tasks) * cfg.vocab);
    std::vector<int32_t> direct_index(static_cast<size_t>(cfg.tasks));
    std::vector<int32_t> queue_index(static_cast<size_t>(cfg.tasks));
    std::vector<float> direct_temp{0.75f, 1.25f};
    std::vector<float> queue_temp{0.75f, 1.25f};

    orStream_t stream = nullptr;
    check(orStreamCreate(&stream), "orStreamCreate");

    const Result direct_raw = run_direct(
        direct_logits, direct_index, direct_temp, cfg, false);
    const Result direct_args = run_direct(
        direct_logits, direct_index, direct_temp, cfg, true);
    const Result queued_raw_args = run_queue(
        stream, queue_logits, queue_index, queue_temp, cfg, direct_raw, 0);
    const Result queued_struct_ptr = run_queue(
        stream, queue_logits, queue_index, queue_temp, cfg, direct_args, 1);
    const Result queued_lambda = run_queue(
        stream, queue_logits, queue_index, queue_temp, cfg, direct_args, 2);
    const Result empty_raw_args = run_empty_queue(stream, cfg, 0);
    const Result empty_struct_ptr = run_empty_queue(stream, cfg, 1);
    const Result empty_lambda = run_empty_queue(stream, cfg, 2);

    check(orStreamDestroy(stream), "orStreamDestroy");

    std::cout << "tasks: " << cfg.tasks << "\n";
    std::cout << "iterations: " << cfg.iterations << " warmup: " << cfg.warmup
              << "\n";
    std::cout << "producer_core: " << cfg.producer_core << "\n";
    std::cout << "vocab: " << cfg.vocab << "\n";
    std::cout << "worker_delay_us: " << cfg.worker_delay_us << "\n";
    std::cout << "memory_protection: off\n\n";

    std::cout << std::fixed << std::setprecision(1);
    std::cout << std::setw(18) << "mode" << std::setw(18) << "batch_ns"
              << std::setw(18) << "per_task_ns" << std::setw(20)
              << "worker_span_ns" << std::setw(20) << "worker_task_ns"
              << std::setw(18) << "extra_task_ns" << "\n";
    auto print = [](const char* name, const Result& result) {
      std::cout << std::setw(18) << name << std::setw(18) << result.batch_ns
                << std::setw(18) << result.task_ns << std::setw(20)
                << result.worker_span_ns << std::setw(20)
                << result.worker_task_ns << std::setw(18)
                << result.extra_task_ns << "\n";
    };
    print("direct_raw_args", direct_raw);
    print("direct_struct_ptr", direct_args);
    print("queue_raw_args", queued_raw_args);
    print("queue_struct_ptr", queued_struct_ptr);
    print("queue_lambda", queued_lambda);
    print("empty_raw_args", empty_raw_args);
    print("empty_struct_ptr", empty_struct_ptr);
    print("empty_lambda", empty_lambda);

    std::cout << "\nchecksums:\n";
    std::cout << "  direct_raw_args: " << direct_raw.checksum << "\n";
    std::cout << "  direct_struct_ptr: " << direct_args.checksum << "\n";
    std::cout << "  queue_raw_args: " << queued_raw_args.checksum << "\n";
    std::cout << "  queue_struct_ptr: " << queued_struct_ptr.checksum << "\n";
    std::cout << "  queue_lambda: " << queued_lambda.checksum << "\n";
    std::cout << "  empty_raw_args: " << empty_raw_args.checksum << "\n";
    std::cout << "  empty_struct_ptr: " << empty_struct_ptr.checksum << "\n";
    std::cout << "  empty_lambda: " << empty_lambda.checksum << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
