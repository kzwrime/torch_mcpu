// SPDX-License-Identifier: Apache-2.0
//
// Benchmark-only pointer kernels for measuring worker-side task gaps:
// next body_begin timestamp minus previous body_end timestamp.

#include "runtime/McpuKernelLaunch.h"
#include "runtime/McpuKernelTiming.h"
#include "runtime/OpenRegException.h"
#include "runtime/OpenRegStream.h"

#include <include/openreg.h>

#include <ATen/ATen.h>
#include <c10/util/ApproximateClock.h>
#include <torch/library.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

namespace {

constexpr int64_t kTimerClock = 0;
constexpr int64_t kTimerTsc = 1;
constexpr int64_t kModeRaw = 0;
constexpr int64_t kModeLambda = 1;
constexpr int64_t kModeScoped = 2;
constexpr int64_t kModeScopedLambda = 3;

int64_t read_tsc() {
#if defined(__x86_64__) || defined(__i386__)
  unsigned int aux = 0;
  return static_cast<int64_t>(__rdtscp(&aux));
#else
  return c10::getTime(/*allow_monotonic=*/true);
#endif
}

int64_t read_timer(int64_t timer) {
  if (timer == kTimerTsc) {
    return read_tsc();
  }
  return c10::getTime(/*allow_monotonic=*/true);
}

void pointer_gap_for_loop_kernel(
    float* data,
    int64_t elements,
    int64_t task,
    int64_t* begin_ns,
    int64_t* end_ns,
    int64_t timer) {
  begin_ns[task] = read_timer(timer);
  for (int64_t i = 0; i < elements; ++i) {
    data[i] = data[i] * 1.0000001f + 0.000001f;
  }
  end_ns[task] = read_timer(timer);
}

void pointer_gap_matmul_128_kernel(
    const float* x,
    const float* weight,
    float* out,
    int64_t task,
    int64_t* begin_ns,
    int64_t* end_ns,
    int64_t timer) {
  begin_ns[task] = read_timer(timer);
  for (int64_t j = 0; j < 128; ++j) {
    float acc = 0.0f;
    for (int64_t k = 0; k < 128; ++k) {
      acc += x[k] * weight[k * 128 + j];
    }
    out[j] = acc;
  }
  end_ns[task] = read_timer(timer);
}

void check_gap_inputs(
    const at::Tensor& data,
    const at::Tensor& begin_ns,
    const at::Tensor& end_ns,
    int64_t task,
    int64_t elements) {
  TORCH_CHECK(data.device().type() == c10::DeviceType::PrivateUse1);
  TORCH_CHECK(begin_ns.device().type() == c10::DeviceType::PrivateUse1);
  TORCH_CHECK(end_ns.device().type() == c10::DeviceType::PrivateUse1);
  TORCH_CHECK(data.scalar_type() == at::kFloat);
  TORCH_CHECK(begin_ns.scalar_type() == at::kLong);
  TORCH_CHECK(end_ns.scalar_type() == at::kLong);
  TORCH_CHECK(data.is_contiguous());
  TORCH_CHECK(begin_ns.is_contiguous());
  TORCH_CHECK(end_ns.is_contiguous());
  TORCH_CHECK(task >= 0);
  TORCH_CHECK(elements >= 0);
  TORCH_CHECK(data.numel() >= elements);
  TORCH_CHECK(begin_ns.numel() > task);
  TORCH_CHECK(end_ns.numel() > task);
}

void pointer_gap_for_loop_impl(
    at::Tensor& data,
    at::Tensor& begin_ns,
    at::Tensor& end_ns,
    int64_t task,
    int64_t elements,
    int64_t mode,
    int64_t timer) {
  check_gap_inputs(data, begin_ns, end_ns, task, elements);

  float* data_ptr = data.mutable_data_ptr<float>();
  int64_t* begin_ptr = begin_ns.mutable_data_ptr<int64_t>();
  int64_t* end_ptr = end_ns.mutable_data_ptr<int64_t>();
  auto stream = c10::mcpu::getCurrentMcpuStream();

  if (mode == kModeRaw) {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
    at::mcpu::launch_timed_kernel_on_stream(
        stream,
        "mcpu::pointer_gap.for_loop.raw",
        [data_ptr, elements, task, begin_ptr, end_ptr, timer](
            at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::pointer_gap.for_loop.raw", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard(
              {data_ptr, begin_ptr, end_ptr});
          pointer_gap_for_loop_kernel(
              data_ptr, elements, task, begin_ptr, end_ptr, timer);
        });
#else
    MCPU_CHECK(orLaunchKernel(
        stream,
        pointer_gap_for_loop_kernel,
        data_ptr,
        elements,
        task,
        begin_ptr,
        end_ptr,
        timer));
#endif
    return;
  }

  if (mode == kModeLambda) {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
    at::mcpu::launch_timed_kernel_on_stream(
        stream,
        "mcpu::pointer_gap.for_loop.lambda",
        [data_ptr, elements, task, begin_ptr, end_ptr, timer](
            at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::pointer_gap.for_loop.lambda", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard(
              {data_ptr, begin_ptr, end_ptr});
          pointer_gap_for_loop_kernel(
              data_ptr, elements, task, begin_ptr, end_ptr, timer);
        });
#else
    at::mcpu::launch_kernel_on_stream(
        stream,
        [data_ptr, elements, task, begin_ptr, end_ptr, timer]() {
          pointer_gap_for_loop_kernel(
              data_ptr, elements, task, begin_ptr, end_ptr, timer);
        });
#endif
    return;
  }

  if (mode == kModeScoped) {
    at::mcpu::launch_timed_kernel_on_stream(
        stream,
        "mcpu::pointer_gap.for_loop.scoped",
        [data_ptr, elements](
            at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::pointer_gap.for_loop.scoped", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard({data_ptr});
          for (int64_t i = 0; i < elements; ++i) {
            data_ptr[i] = data_ptr[i] * 1.0000001f + 0.000001f;
          }
        });
    return;
  }

  if (mode == kModeScopedLambda) {
    at::mcpu::launch_timed_kernel_on_stream(
        stream,
        "mcpu::pointer_gap.for_loop.scoped_lambda",
        [data_ptr, elements](
            at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::pointer_gap.for_loop.scoped_lambda", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard({data_ptr});
          for (int64_t i = 0; i < elements; ++i) {
            data_ptr[i] = data_ptr[i] * 1.0000001f + 0.000001f;
          }
        });
    return;
  }

  TORCH_CHECK(
      false,
      "pointer_gap_for_loop mode must be 0(raw), 1(lambda), 2(scoped), or 3(scoped_lambda)");
}

void pointer_gap_matmul_128_impl(
    at::Tensor& data,
    at::Tensor& begin_ns,
    at::Tensor& end_ns,
    int64_t task,
    int64_t mode,
    int64_t timer) {
  constexpr int64_t kInputElements = 128;
  constexpr int64_t kWeightElements = 128 * 128;
  constexpr int64_t kOutElements = 128;
  constexpr int64_t kRequiredElements =
      kInputElements + kWeightElements + kOutElements;
  check_gap_inputs(data, begin_ns, end_ns, task, kRequiredElements);

  float* data_ptr = data.mutable_data_ptr<float>();
  const float* x_ptr = data_ptr;
  const float* weight_ptr = data_ptr + kInputElements;
  float* out_ptr = data_ptr + kInputElements + kWeightElements;
  int64_t* begin_ptr = begin_ns.mutable_data_ptr<int64_t>();
  int64_t* end_ptr = end_ns.mutable_data_ptr<int64_t>();
  auto stream = c10::mcpu::getCurrentMcpuStream();

  if (mode == kModeRaw) {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
    at::mcpu::launch_timed_kernel_on_stream(
        stream,
        "mcpu::pointer_gap.matmul_128.raw",
        [x_ptr, weight_ptr, out_ptr, task, begin_ptr, end_ptr, timer](
            at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::pointer_gap.matmul_128.raw", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard(
              {x_ptr, weight_ptr, out_ptr, begin_ptr, end_ptr});
          pointer_gap_matmul_128_kernel(
              x_ptr, weight_ptr, out_ptr, task, begin_ptr, end_ptr, timer);
        });
#else
    MCPU_CHECK(orLaunchKernel(
        stream,
        pointer_gap_matmul_128_kernel,
        x_ptr,
        weight_ptr,
        out_ptr,
        task,
        begin_ptr,
        end_ptr,
        timer));
#endif
    return;
  }

  if (mode == kModeLambda) {
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
    at::mcpu::launch_timed_kernel_on_stream(
        stream,
        "mcpu::pointer_gap.matmul_128.lambda",
        [x_ptr, weight_ptr, out_ptr, task, begin_ptr, end_ptr, timer](
            at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::pointer_gap.matmul_128.lambda", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard(
              {x_ptr, weight_ptr, out_ptr, begin_ptr, end_ptr});
          pointer_gap_matmul_128_kernel(
              x_ptr, weight_ptr, out_ptr, task, begin_ptr, end_ptr, timer);
        });
#else
    at::mcpu::launch_kernel_on_stream(
        stream,
        [x_ptr, weight_ptr, out_ptr, task, begin_ptr, end_ptr, timer]() {
          pointer_gap_matmul_128_kernel(
              x_ptr, weight_ptr, out_ptr, task, begin_ptr, end_ptr, timer);
        });
#endif
    return;
  }

  if (mode == kModeScoped) {
    at::mcpu::launch_timed_kernel_on_stream(
        stream,
        "mcpu::pointer_gap.matmul_128.scoped",
        [x_ptr, weight_ptr, out_ptr](
            at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::pointer_gap.matmul_128.scoped", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard(
              {x_ptr, weight_ptr, out_ptr});
          for (int64_t j = 0; j < 128; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < 128; ++k) {
              acc += x_ptr[k] * weight_ptr[k * 128 + j];
            }
            out_ptr[j] = acc;
          }
        });
    return;
  }

  if (mode == kModeScopedLambda) {
    at::mcpu::launch_timed_kernel_on_stream(
        stream,
        "mcpu::pointer_gap.matmul_128.scoped_lambda",
        [x_ptr, weight_ptr, out_ptr](
            at::mcpu::kernel_timing::Event* timing_event) {
          MCPU_KERNEL_TIMING_SCOPE_EVENT(
              "mcpu::pointer_gap.matmul_128.scoped_lambda", timing_event);
          at::mcpu::KernelPointerMemoryGuard guard(
              {x_ptr, weight_ptr, out_ptr});
          for (int64_t j = 0; j < 128; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < 128; ++k) {
              acc += x_ptr[k] * weight_ptr[k * 128 + j];
            }
            out_ptr[j] = acc;
          }
        });
    return;
  }

  TORCH_CHECK(
      false,
      "pointer_gap_matmul_128 mode must be 0(raw), 1(lambda), 2(scoped), or 3(scoped_lambda)");
}

} // namespace

int64_t pointer_gap_read_tsc() {
  return read_tsc();
}

TORCH_LIBRARY_FRAGMENT(mcpu, m) {
  m.def(
      "pointer_gap_for_loop(Tensor(a!) data, Tensor(b!) begin_ns, Tensor(c!) end_ns, int task, int elements, int mode, int timer) -> ()");
  m.def(
      "pointer_gap_matmul_128(Tensor(a!) data, Tensor(b!) begin_ns, Tensor(c!) end_ns, int task, int mode, int timer) -> ()");
  m.def("pointer_gap_read_tsc() -> int");
}

TORCH_LIBRARY_IMPL(mcpu, PrivateUse1, m) {
  m.impl("pointer_gap_for_loop", &pointer_gap_for_loop_impl);
  m.impl("pointer_gap_matmul_128", &pointer_gap_matmul_128_impl);
}

TORCH_LIBRARY_IMPL(mcpu, CompositeExplicitAutograd, m) {
  m.impl("pointer_gap_read_tsc", &pointer_gap_read_tsc);
}
