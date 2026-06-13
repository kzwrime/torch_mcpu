#pragma once

#include <openreg.h>

#include "OpenRegException.h"
#include "OpenRegFunctions.h"

#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

#if defined(__unix__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace c10::mcpu {

namespace detail {

class OptionalPythonGilRelease {
 public:
  OptionalPythonGilRelease() {
#if defined(__unix__) || defined(__APPLE__)
    auto* is_initialized =
        reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "Py_IsInitialized"));
    auto* gil_state_check =
        reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT, "PyGILState_Check"));
    save_thread_ =
        reinterpret_cast<void* (*)()>(dlsym(RTLD_DEFAULT, "PyEval_SaveThread"));
    restore_thread_ =
        reinterpret_cast<void (*)(void*)>(
            dlsym(RTLD_DEFAULT, "PyEval_RestoreThread"));

    if (is_initialized && gil_state_check && save_thread_ && restore_thread_ &&
        is_initialized() && gil_state_check()) {
      state_ = save_thread_();
    }
#endif
  }

  ~OptionalPythonGilRelease() {
    if (state_ && restore_thread_) {
      restore_thread_(state_);
    }
  }

  OptionalPythonGilRelease(const OptionalPythonGilRelease&) = delete;
  OptionalPythonGilRelease& operator=(const OptionalPythonGilRelease&) = delete;

 private:
  void* state_ = nullptr;
  void* (*save_thread_)() = nullptr;
  void (*restore_thread_)(void*) = nullptr;
};

} // namespace detail

// Derive compile-time priority count from shared mcpu backend constant.
static constexpr int max_compile_time_stream_priorities = 2;

class MCPU_EXPORT McpuStream {
 public:
  enum Unchecked { UNCHECKED };

  explicit McpuStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::PrivateUse1);
  }

  explicit McpuStream(Unchecked, Stream stream) : stream_(stream) {}

  bool operator==(const McpuStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const McpuStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  operator orStream_t() const {
    return stream();
  }

  operator Stream() const {
    return unwrap();
  }

  DeviceType device_type() const {
    return DeviceType::PrivateUse1;
  }

  DeviceIndex device_index() const {
    return stream_.device_index();
  }

  Device device() const {
    return Device(DeviceType::PrivateUse1, device_index());
  }

  StreamId id() const {
    return stream_.id();
  }

  bool query() const {
    DeviceGuard guard{stream_.device()};

    if (orStreamQuery(stream()) == orSuccess) {
      return true;
    }

    return false;
  }

  void synchronize() const {
    DeviceGuard guard{stream_.device()};
    detail::OptionalPythonGilRelease release_gil;
    MCPU_CHECK(orStreamSynchronize(stream()));
  }

  int priority() const {
    DeviceGuard guard{stream_.device()};
    int priority = 0;
    MCPU_CHECK(orStreamGetPriority(stream(), &priority));
    return priority;
  }

  orStream_t stream() const;

  Stream unwrap() const {
    return stream_;
  }

  struct c10::StreamData3 pack3() const {
    return stream_.pack3();
  }

  static McpuStream unpack3(
      StreamId stream_id,
      DeviceIndex device_index,
      DeviceType device_type) {
    return McpuStream(Stream::unpack3(stream_id, device_index, device_type));
  }

 private:
  Stream stream_;
};

/*
 * Get a stream from the pool in a round-robin fashion.
 *
 * You can request a stream from the highest priority pool by setting
 * isHighPriority to true for a specific device.
 */
MCPU_EXPORT McpuStream
getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);

/*
 * Get a stream from the pool in a round-robin fashion.
 *
 * You can request a stream by setting a priority value for a specific device.
 * The priority number lower, the priority higher.
 */
MCPU_EXPORT McpuStream
getStreamFromPool(const int priority, DeviceIndex device = -1);

/*
 * Get a McpuStream from a externally allocated one.
 *
 * This is mainly for interoperability with different libraries where we
 * want to operate on a non-torch allocated stream for data exchange or similar
 * purposes
 */
MCPU_EXPORT McpuStream
getStreamFromExternal(orStream_t ext_stream, DeviceIndex device_index);

/*
 * Get the default Mcpu stream, for the passed Mcpu device, or for the
 * current device if no device index is passed.
 */
MCPU_EXPORT McpuStream getDefaultMcpuStream(DeviceIndex device_index = -1);

/*
 * Get the current Mcpu stream, for the passed Mcpu device, or for the
 * current device if no device index is passed.
 */
MCPU_EXPORT McpuStream getCurrentMcpuStream(DeviceIndex device_index = -1);

/*
 * Set the current stream on the device of the passed in stream to be the passed
 * in stream.
 */
MCPU_EXPORT void setCurrentMcpuStream(McpuStream stream);

MCPU_EXPORT std::ostream& operator<<(std::ostream& stream, const McpuStream& s);

} // namespace c10::mcpu

namespace std {
template <>
struct hash<c10::mcpu::McpuStream> {
  size_t operator()(c10::mcpu::McpuStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std
