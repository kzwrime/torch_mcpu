#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef _WIN32
#define MCPU_OP_TIMING_EXPORT __declspec(dllexport)
#else
#define MCPU_OP_TIMING_EXPORT __attribute__((visibility("default")))
#endif

namespace at::mcpu::op_timing {

struct Record {
  const char* op;
  const char* phase;
  std::uint64_t t_ns;
};

struct ThreadSnapshot {
  const char* role;
  std::vector<Record> records;
};

MCPU_OP_TIMING_EXPORT bool enabled();
MCPU_OP_TIMING_EXPORT void set_enabled(bool value);
MCPU_OP_TIMING_EXPORT void reset();
MCPU_OP_TIMING_EXPORT void record(const char* role, const char* op, const char* phase);
MCPU_OP_TIMING_EXPORT std::vector<ThreadSnapshot> snapshot();

} // namespace at::mcpu::op_timing
