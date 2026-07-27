#include "Common.h"
#include "runtime/OpenRegGenerator.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/uniform.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <type_traits>

namespace at::mcpu {
namespace {

constexpr int64_t kUniformParallelGrainSize = 262144;

struct UniformBounds {
  double from;
  double to;
};

UniformBounds check_uniform_bounds(
    at::ScalarType dtype,
    double from,
    double to) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      dtype,
      "check_mcpu_uniform_bounds",
      [&] {
        const auto min =
            static_cast<double>(std::numeric_limits<scalar_t>::lowest());
        const auto max =
            static_cast<double>(std::numeric_limits<scalar_t>::max());
        TORCH_CHECK(
            from >= min && from <= max, "from is out of bounds for ", dtype);
        TORCH_CHECK(to >= min && to <= max, "to is out of bounds for ", dtype);
        TORCH_CHECK(
            from <= to,
            "uniform_ expects to return a [from, to) range, but found from=",
            from,
            " > to=",
            to);
        TORCH_CHECK(
            (to - from) <= max,
            "uniform_ expects to-from <= std::numeric_limits<",
            c10::toString(dtype),
            ">::max(), but found to=",
            to,
            " and from=",
            from,
            " which result in to-from to exceed the limit");
        from = std::min(std::max(from, min), max);
        to = std::max(std::min(to, max), min);
      });
  return {from, to};
}

at::CPUGeneratorImpl* get_uniform_generator(
    c10::DeviceIndex device_index,
    const std::optional<at::Generator>& generator) {
  if (generator.has_value() && generator->defined()) {
    TORCH_CHECK(
        generator->device().type() == c10::DeviceType::PrivateUse1,
        "Expected an mcpu generator for mcpu uniform_, but got ",
        generator->device());
    return generator->get<c10::mcpu::McpuGeneratorImpl>();
  }

  const at::Generator& default_generator =
      c10::mcpu::getDefaultMcpuGenerator(device_index);
  return default_generator.get<c10::mcpu::McpuGeneratorImpl>();
}

uint64_t reserve_uniform_seed(
    c10::DeviceIndex device_index,
    const std::optional<at::Generator>& generator) {
  auto* gen = get_uniform_generator(device_index, generator);
  std::lock_guard<std::mutex> lock(gen->mutex_);
  return gen->random64();
}

inline float uniform_float(uint32_t value) {
  return std::ldexp(static_cast<float>(value >> 8), -24);
}

inline double uniform_double(uint32_t high, uint32_t low) {
  const auto mantissa = (static_cast<uint64_t>(high >> 5) << 26) | (low >> 6);
  return std::ldexp(static_cast<double>(mantissa), -53);
}

template <typename scalar_t>
void uniform_contiguous_kernel(
    scalar_t* data,
    int64_t numel,
    double from,
    double to,
    uint64_t seed) {
  using opmath_t = at::opmath_type<scalar_t>;
  constexpr int64_t words_per_value = std::is_same_v<scalar_t, double> ? 2 : 1;
  constexpr int64_t values_per_block = 4 / words_per_value;
  const auto num_blocks = (numel + values_per_block - 1) / values_per_block;
  const auto grain_blocks =
      std::max<int64_t>(1, kUniformParallelGrainSize / values_per_block);
  const auto from_value = static_cast<scalar_t>(from);
  const auto to_value = static_cast<scalar_t>(to);
  const auto range = static_cast<opmath_t>(to_value - from_value);

  at::parallel_for(
      0, num_blocks, grain_blocks, [&](int64_t begin, int64_t end) {
        at::Philox4_32 engine(
            seed, /*subsequence=*/0, static_cast<uint64_t>(begin));
        for (int64_t block = begin; block < end; ++block) {
          for (int64_t lane = 0; lane < values_per_block; ++lane) {
            const auto index = block * values_per_block + lane;
            if (index >= numel) {
              return;
            }
            opmath_t random;
            if constexpr (std::is_same_v<scalar_t, double>) {
              random = uniform_double(engine(), engine());
            } else {
              random = static_cast<opmath_t>(uniform_float(engine()));
            }
            auto value = static_cast<scalar_t>(random * range + from_value);
            data[index] = value == to_value ? from_value : value;
          }
        }
      });
}

void uniform_mcpu_impl(
    const at::Tensor& cpu_self,
    double from,
    double to,
    uint64_t seed) {
  if (cpu_self.numel() == 0) {
    return;
  }

  if (cpu_self.is_contiguous()) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        cpu_self.scalar_type(),
        "mcpu_uniform_contiguous",
        [&] {
          uniform_contiguous_kernel(
              cpu_self.mutable_data_ptr<scalar_t>(),
              cpu_self.numel(),
              from,
              to,
              seed);
        });
    return;
  }

  at::CPUGeneratorImpl generator(seed);
  auto iter = at::TensorIterator::borrowing_nullary_op(cpu_self);
  at::native::templates::cpu::uniform_kernel(iter, from, to, &generator);
}

at::Tensor& uniform_(
    at::Tensor& self,
    double from,
    double to,
    std::optional<at::Generator> generator) {
  const auto bounds = check_uniform_bounds(self.scalar_type(), from, to);
  from = bounds.from;
  to = bounds.to;
  auto device_index = self.device().index();
  const auto seed = self.numel() == 0
      ? uint64_t{0}
      : reserve_uniform_seed(device_index, generator);
  MCPU_LAUNCH_TIMED_KERNEL("mcpu::aten::uniform_", ([ self, from, to, seed ]), {
    KernelMemoryGuard guard(self);
    auto cpu_self = ops::get_cpu_view_from_mcpu_tensor(self);
    uniform_mcpu_impl(cpu_self, from, to, seed);
  });
  return self;
}

at::Tensor& uniform_out(
    const at::Tensor& self,
    double from,
    double to,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  ops::check_out_sizes("aten::uniform.out", out, self.sizes());
  const auto bounds = check_uniform_bounds(out.scalar_type(), from, to);
  from = bounds.from;
  to = bounds.to;
  auto device_index = out.device().index();
  const auto seed = out.numel() == 0
      ? uint64_t{0}
      : reserve_uniform_seed(device_index, generator);

  MCPU_LAUNCH_TIMED_KERNEL(
      "mcpu::aten::uniform.out", ([ out, from, to, seed ]), {
        KernelMemoryGuard guard(out);
        auto cpu_out = ops::get_cpu_view_from_mcpu_tensor(out);
        uniform_mcpu_impl(cpu_out, from, to, seed);
      });
  return out;
}

at::Tensor uniform(
    const at::Tensor& self,
    double from,
    double to,
    std::optional<at::Generator> generator) {
  auto out = at::empty_like(self, self.options(), at::MemoryFormat::Preserve);
  uniform_out(self, from, to, generator, out);
  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("uniform", &uniform);
  m.impl("uniform.out", &uniform_out);
  m.impl("uniform_", &uniform_);
}

} // namespace at::mcpu
