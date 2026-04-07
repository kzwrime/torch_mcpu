#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace openreg {

constexpr size_t kAlignment = 32;

int alloc(void** mem, size_t alignment, size_t size) {
#ifdef _WIN32
  *mem = _aligned_malloc(size, alignment);
  return *mem ? 0 : -1;
#else
  return posix_memalign(mem, alignment, size);
#endif
}

void free(void* mem) {
#ifdef _WIN32
  _aligned_free(mem);
#else
  ::free(mem);
#endif
}

} // namespace openreg
