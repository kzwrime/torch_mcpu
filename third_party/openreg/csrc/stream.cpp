#include <include/openreg.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

static std::mutex g_mutex;
static std::once_flag g_flag;
static std::vector<std::set<orStream_t>> g_streams_per_device;
#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
thread_local bool g_in_kernel_task = false;
#endif

static void initialize_registries() {
  int device_count = 0;
  orGetDeviceCount(&device_count);
  g_streams_per_device.resize(device_count);
}

static int choose_worker_core() {
  if (const char* env = std::getenv("TORCH_MCPU_STREAM_WORKER_CORE")) {
    return std::atoi(env);
  }
  // Do not pin production stream workers by default. OpenMP kernels launched
  // from the worker inherit the worker affinity mask, so single-core pinning
  // collapses OMP_NUM_THREADS parallel regions onto one CPU.
  return -1;
}

static void pin_this_thread_to_core(int core) {
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

struct orEventRecordState {
  std::mutex mtx;
  std::condition_variable cv;
  bool completed = false;
  std::uint64_t version = 0;
  std::chrono::high_resolution_clock::time_point completion_time;
};

struct orEventImpl {
  std::mutex mtx;
  std::uint64_t next_version = 0;
  std::shared_ptr<orEventRecordState> latest_record;
  int device_index = -1;
  bool timing_enabled{false};
};

struct orEvent {
  std::shared_ptr<orEventImpl> impl;
};

orStream::orStream() {
  for (std::uint64_t i = 0; i < kQueueCapacity; ++i) {
    slots[i].sequence.store(i, std::memory_order_relaxed);
  }
  worker = std::thread([this] {
    pin_this_thread_to_core(choose_worker_core());
    workerLoop();
  });
}

orStream::~orStream() {
  synchronize();
  stop.store(true, std::memory_order_release);
  if (worker.joinable()) {
    worker.join();
  }
}

void orStream::workerLoop() {
  std::uint64_t local_head = head.load(std::memory_order_relaxed);
  while (true) {
    auto& slot = slots[local_head & kQueueMask];
    const auto sequence = slot.sequence.load(std::memory_order_acquire);
    if (sequence != local_head + 1) {
      if (stop.load(std::memory_order_acquire) &&
          local_head == tail.load(std::memory_order_acquire)) {
        return;
      }
      openreg::cpu_relax();
      continue;
    }

    slot.run(slot.storage);
    slot.destroy(slot.storage);
    slot.sequence.store(local_head + kQueueCapacity, std::memory_order_release);
    ++local_head;
    head.store(local_head, std::memory_order_release);
    completed.store(local_head, std::memory_order_release);
  }
}

void orStream::synchronize() {
  const auto target = tail.load(std::memory_order_acquire);
  while (completed.load(std::memory_order_acquire) < target) {
    openreg::cpu_relax();
  }
}

bool orStream::idle() const {
  return completed.load(std::memory_order_acquire) ==
      tail.load(std::memory_order_acquire);
}

#if TORCH_MCPU_ENABLE_MEMORY_PROTECTION
bool openreg::isInKernelTask() {
  return g_in_kernel_task;
}

void openreg::setInKernelTask(bool enabled) {
  g_in_kernel_task = enabled;
}
#endif

orError_t orEventCreateWithFlags(orEvent_t* event, unsigned int flags) {
  if (!event)
    return orErrorUnknown;

  auto impl = std::make_shared<orEventImpl>();
  orGetDevice(&(impl->device_index));
  if (flags & orEventEnableTiming) {
    impl->timing_enabled = true;
  }

  *event = new orEvent{std::move(impl)};
  return orSuccess;
}

orError_t orEventCreate(orEvent_t* event) {
  return orEventCreateWithFlags(event, orEventDisableTiming);
}

orError_t orEventDestroy(orEvent_t event) {
  if (!event)
    return orErrorUnknown;

  delete event;
  return orSuccess;
}

orError_t orEventRecord(orEvent_t event, orStream_t stream) {
  if (!event || !stream)
    return orErrorUnknown;

  auto event_impl = event->impl;
  if (event_impl->device_index != stream->device_index) {
    return orErrorUnknown;
  }

  auto record = std::make_shared<orEventRecordState>();
  const bool timing_enabled = event_impl->timing_enabled;
  {
    std::lock_guard<std::mutex> lock(event_impl->mtx);
    record->version = ++event_impl->next_version;
    event_impl->latest_record = record;
  }

  return orLaunchKernel(stream, [record, timing_enabled]() {
    std::lock_guard<std::mutex> lock(record->mtx);
    if (timing_enabled) {
      record->completion_time = std::chrono::high_resolution_clock::now();
    }
    record->completed = true;
    record->cv.notify_all();
  });
}

orError_t orEventSynchronize(orEvent_t event) {
  if (!event)
    return orErrorUnknown;

  auto event_impl = event->impl;
  std::shared_ptr<orEventRecordState> record;
  {
    std::lock_guard<std::mutex> lock(event_impl->mtx);
    record = event_impl->latest_record;
  }

  if (!record) {
    return orSuccess;
  }

  std::unique_lock<std::mutex> lock(record->mtx);
  record->cv.wait(lock, [&] {
    return record->completed;
  });

  return orSuccess;
}

orError_t orEventQuery(orEvent_t event) {
  if (!event)
    return orErrorUnknown;

  auto event_impl = event->impl;
  std::shared_ptr<orEventRecordState> record;
  {
    std::lock_guard<std::mutex> lock(event_impl->mtx);
    record = event_impl->latest_record;
  }

  if (!record) {
    return orSuccess;
  }

  std::lock_guard<std::mutex> lock(record->mtx);
  return record->completed ? orSuccess : orErrorNotReady;
}

orError_t orEventElapsedTime(float* ms, orEvent_t start, orEvent_t end) {
  if (!ms || !start || !end)
    return orErrorUnknown;

  auto start_impl = start->impl;
  auto end_impl = end->impl;

  if (start_impl->device_index != end_impl->device_index) {
    return orErrorUnknown;
  }

  if (!start_impl->timing_enabled || !end_impl->timing_enabled) {
    return orErrorUnknown;
  }

  auto snapshot_record = [](const std::shared_ptr<orEventImpl>& impl) {
    std::lock_guard<std::mutex> lock(impl->mtx);
    return impl->latest_record;
  };

  auto start_record = snapshot_record(start_impl);
  auto end_record = snapshot_record(end_impl);
  if (!start_record || !end_record) {
    return orErrorUnknown;
  }

  auto snapshot_completion_time =
      [](const std::shared_ptr<orEventRecordState>& record,
         auto* completion_time) {
        std::lock_guard<std::mutex> lock(record->mtx);
        if (!record->completed) {
          return false;
        }
        *completion_time = record->completion_time;
        return true;
      };

  std::chrono::high_resolution_clock::time_point start_completion_time;
  std::chrono::high_resolution_clock::time_point end_completion_time;
  if (!snapshot_completion_time(start_record, &start_completion_time) ||
      !snapshot_completion_time(end_record, &end_completion_time)) {
    return orErrorUnknown;
  }

  auto duration = end_completion_time - start_completion_time;
  *ms = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
            duration)
            .count();

  return orSuccess;
}

orError_t orStreamCreateWithPriority(
    orStream_t* stream,
    [[maybe_unused]] unsigned int flag,
    int priority) {
  if (!stream) {
    return orErrorUnknown;
  }

  int min_p, max_p;
  orDeviceGetStreamPriorityRange(&min_p, &max_p);
  if (priority < min_p || priority > max_p) {
    return orErrorUnknown;
  }

  int current_device = 0;
  orGetDevice(&current_device);

  orStream_t new_stream = new orStream();
  new_stream->device_index = current_device;

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::call_once(g_flag, initialize_registries);
    g_streams_per_device[current_device].insert(new_stream);
  }

  *stream = new_stream;

  return orSuccess;
}

orError_t orStreamCreate(orStream_t* stream) {
  int min_p, max_p;
  orDeviceGetStreamPriorityRange(&min_p, &max_p);

  return orStreamCreateWithPriority(stream, 0, max_p);
}

orError_t orStreamGetPriority(
    [[maybe_unused]] orStream_t stream,
    int* priority) {
  *priority = 0;
  return orSuccess;
}

orError_t orStreamDestroy(orStream_t stream) {
  if (!stream)
    return orErrorUnknown;

  {
    std::lock_guard<std::mutex> lock(g_mutex);

    int device_idx = stream->device_index;
    if (device_idx >= 0 &&
        device_idx < static_cast<int>(g_streams_per_device.size())) {
      g_streams_per_device[device_idx].erase(stream);
    }
  }

  delete stream;
  return orSuccess;
}

orError_t orStreamQuery(orStream_t stream) {
  if (!stream) {
    return orErrorUnknown;
  }

  return stream->idle() ? orSuccess : orErrorNotReady;
}

orError_t orStreamSynchronize(orStream_t stream) {
  if (!stream)
    return orErrorUnknown;

  stream->synchronize();
  return orSuccess;
}

orError_t orStreamWaitEvent(orStream_t stream, orEvent_t event, unsigned int) {
  if (!stream || !event)
    return orErrorUnknown;

  auto event_impl = event->impl;
  std::shared_ptr<orEventRecordState> record;
  {
    std::lock_guard<std::mutex> lock(event_impl->mtx);
    record = event_impl->latest_record;
  }

  if (!record) {
    return orSuccess;
  }

  return orLaunchKernel(stream, [record]() {
    std::unique_lock<std::mutex> lock(record->mtx);
    record->cv.wait(lock, [&] {
      return record->completed;
    });
  });
}

orError_t orDeviceGetStreamPriorityRange(
    int* leastPriority,
    int* greatestPriority) {
  if (!leastPriority || !greatestPriority) {
    return orErrorUnknown;
  }

  *leastPriority = 0;
  *greatestPriority = 1;
  return orSuccess;
}

orError_t orDeviceSynchronize(void) {
  int current_device = 0;
  orGetDevice(&current_device);

  std::vector<orStream_t> streams;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::call_once(g_flag, initialize_registries);

    auto& streams_on_device = g_streams_per_device[current_device];
    streams.assign(streams_on_device.begin(), streams_on_device.end());
  }

  for (orStream_t stream : streams) {
    orError_t status = orStreamSynchronize(stream);
    if (status != orSuccess) {
      return status;
    }
  }

  return orSuccess;
}
