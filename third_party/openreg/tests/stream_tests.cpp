#include <gtest/gtest.h>
#include <include/openreg.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

class StreamTest : public ::testing::Test {
 protected:
  void SetUp() override {
    orSetDevice(0);
  }
};

class EnvGuard {
 public:
  explicit EnvGuard(const char* name, const char* value) : name_(name) {
    if (const char* existing = std::getenv(name)) {
      had_value_ = true;
      old_value_ = existing;
    }
    if (value) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }

  ~EnvGuard() {
    if (had_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  bool had_value_ = false;
  std::string old_value_;
};

TEST_F(StreamTest, StreamCreateAndDestroy) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);
  EXPECT_NE(stream, nullptr);

  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, StreamCreateNullptr) {
  // Creation API should reject null double-pointer inputs.
  EXPECT_EQ(orStreamCreate(nullptr), orErrorUnknown);
}

TEST_F(StreamTest, StreamCreateWithInvalidPriority) {
  orStream_t stream = nullptr;
  int min_p, max_p;
  orDeviceGetStreamPriorityRange(&min_p, &max_p);

  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, min_p - 1), orErrorUnknown);
  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, max_p + 1), orErrorUnknown);
}

TEST_F(StreamTest, StreamCreateWithPriorityValidBounds) {
  orStream_t stream = nullptr;
  int min_p, max_p;
  orDeviceGetStreamPriorityRange(&min_p, &max_p);

  // Lowest priority should be accepted.
  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, min_p), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);

  // Highest priority should also be accepted.
  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, max_p), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, StreamDestroyNullptr) {
  // Destroying nullptr should follow CUDA error behavior.
  EXPECT_EQ(orStreamDestroy(nullptr), orErrorUnknown);
}

TEST_F(StreamTest, StreamGetPriority) {
  orStream_t stream = nullptr;
  int min_p, max_p;
  orDeviceGetStreamPriorityRange(&min_p, &max_p);
  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, max_p), orSuccess);

  int priority = -1;
  EXPECT_EQ(orStreamGetPriority(stream, &priority), orSuccess);
  EXPECT_EQ(priority, max_p);

  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, StreamTaskExecution) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  std::atomic<int> counter{0};
  EXPECT_EQ(openreg::addTaskToStream(stream, [&] { counter++; }), orSuccess);

  EXPECT_EQ(orStreamSynchronize(stream), orSuccess);
  EXPECT_EQ(counter.load(), 1);

  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, BlockingIdleWorkerWakesForPublishedTask) {
  orStream_t stream = nullptr;
  int min_p = 0;
  int unused_max_p = 0;
  orDeviceGetStreamPriorityRange(&min_p, &unused_max_p);
  EXPECT_EQ(orStreamCreateWithPriority(&stream, 0, min_p), orSuccess);
  EXPECT_EQ(stream->idle_policy, orStream::IdlePolicy::Block);

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  std::atomic<int> counter{0};
  EXPECT_EQ(openreg::addTaskToStream(stream, [&] { counter++; }), orSuccess);
  EXPECT_EQ(orStreamSynchronize(stream), orSuccess);
  EXPECT_EQ(counter.load(), 1);

  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, BlockingIdleStreamDestroysWithoutHang) {
  orStream_t blocking_stream = nullptr;
  int min_p = 0;
  int unused_max_p = 0;
  orDeviceGetStreamPriorityRange(&min_p, &unused_max_p);
  EXPECT_EQ(orStreamCreateWithPriority(&blocking_stream, 0, min_p), orSuccess);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_EQ(orStreamDestroy(blocking_stream), orSuccess);
}

TEST_F(StreamTest, StreamAcceptsConcurrentProducers) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  constexpr int kProducerCount = 8;
  constexpr int kTasksPerProducer = 4096;
  constexpr int kTotalTasks = kProducerCount * kTasksPerProducer;

  std::atomic<bool> start{false};
  std::atomic<int> launch_errors{0};
  std::vector<std::atomic<int>> seen(kTotalTasks);
  for (auto& value : seen) {
    value.store(0, std::memory_order_relaxed);
  }

  std::vector<std::thread> producers;
  producers.reserve(kProducerCount);
  for (int producer = 0; producer < kProducerCount; ++producer) {
    producers.emplace_back([&, producer] {
      while (!start.load(std::memory_order_acquire)) {
        openreg::cpu_relax();
      }
      const int base = producer * kTasksPerProducer;
      for (int task = 0; task < kTasksPerProducer; ++task) {
        const int id = base + task;
        if (openreg::addTaskToStream(stream, [&, id] {
              seen[id].fetch_add(1, std::memory_order_relaxed);
            }) != orSuccess) {
          launch_errors.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& producer : producers) {
    producer.join();
  }

  EXPECT_EQ(orStreamSynchronize(stream), orSuccess);
  EXPECT_EQ(launch_errors.load(std::memory_order_relaxed), 0);
  for (int id = 0; id < kTotalTasks; ++id) {
    EXPECT_EQ(seen[id].load(std::memory_order_relaxed), 1) << "task " << id;
  }

  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, WorkerThreadCanRecordEventWhileHostProducesTargetStream) {
  orStream_t worker_stream = nullptr;
  orStream_t target_stream = nullptr;
  orEvent_t event = nullptr;
  EXPECT_EQ(orStreamCreate(&worker_stream), orSuccess);
  EXPECT_EQ(orStreamCreate(&target_stream), orSuccess);
  EXPECT_EQ(orEventCreate(&event), orSuccess);

  constexpr int kHostTasks = 4096;
  std::atomic<bool> start{false};
  std::atomic<int> host_launch_errors{0};
  std::atomic<int> record_status{orErrorUnknown};
  std::atomic<int> counter{0};

  EXPECT_EQ(openreg::addTaskToStream(worker_stream, [&] {
    while (!start.load(std::memory_order_acquire)) {
      openreg::cpu_relax();
    }
    record_status.store(
        orEventRecord(event, target_stream), std::memory_order_release);
  }), orSuccess);

  std::thread host_producer([&] {
    while (!start.load(std::memory_order_acquire)) {
      openreg::cpu_relax();
    }
    for (int i = 0; i < kHostTasks; ++i) {
      if (openreg::addTaskToStream(target_stream, [&] {
            counter.fetch_add(1, std::memory_order_relaxed);
          }) != orSuccess) {
        host_launch_errors.fetch_add(1, std::memory_order_relaxed);
      }
    }
  });

  start.store(true, std::memory_order_release);
  host_producer.join();

  EXPECT_EQ(orStreamSynchronize(worker_stream), orSuccess);
  EXPECT_EQ(record_status.load(std::memory_order_acquire), orSuccess);
  EXPECT_EQ(orStreamSynchronize(target_stream), orSuccess);
  EXPECT_EQ(orEventQuery(event), orSuccess);
  EXPECT_EQ(host_launch_errors.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(counter.load(std::memory_order_relaxed), kHostTasks);

  EXPECT_EQ(orEventDestroy(event), orSuccess);
  EXPECT_EQ(orStreamDestroy(worker_stream), orSuccess);
  EXPECT_EQ(orStreamDestroy(target_stream), orSuccess);
}

TEST_F(StreamTest, HighPriorityStreamsDefaultToBusyIdle) {
  int unused_min_p = 0;
  int max_p = 0;
  orDeviceGetStreamPriorityRange(&unused_min_p, &max_p);
  orStream_t stream = nullptr;
  ASSERT_EQ(orStreamCreateWithPriority(&stream, 0, max_p), orSuccess);
  EXPECT_EQ(stream->priority, max_p);
  EXPECT_EQ(stream->idle_policy, orStream::IdlePolicy::Busy);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, DefaultStreamFlagForcesBusyIdle) {
  int min_p = 0;
  int unused_max_p = 0;
  orDeviceGetStreamPriorityRange(&min_p, &unused_max_p);
  orStream_t stream = nullptr;
  ASSERT_EQ(orStreamCreateWithPriority(&stream, orStreamDefault, min_p),
            orSuccess);
  EXPECT_TRUE(stream->is_default_stream);
  EXPECT_EQ(stream->priority, min_p);
  EXPECT_EQ(stream->idle_policy, orStream::IdlePolicy::Busy);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, StreamAffinityOnlyAppliesToDefaultStream) {
  EnvGuard legacy("TORCH_MCPU_STREAM_WORKER_CORE", "2");

  int min_p = 0;
  int max_p = 0;
  orDeviceGetStreamPriorityRange(&min_p, &max_p);

  orStream_t default_stream = nullptr;
  ASSERT_EQ(orStreamCreateWithPriority(&default_stream, orStreamDefault, min_p),
            orSuccess);
  EXPECT_EQ(default_stream->affinity_core, 2);
  EXPECT_EQ(orStreamDestroy(default_stream), orSuccess);

  orStream_t pool_stream = nullptr;
  ASSERT_EQ(orStreamCreateWithPriority(&pool_stream, 0, max_p), orSuccess);
  EXPECT_EQ(pool_stream->affinity_core, -1);
  EXPECT_EQ(orStreamDestroy(pool_stream), orSuccess);
}

TEST_F(StreamTest, StreamAffinityInvalidCoreListFallsBackToNoPinning) {
  EnvGuard legacy("TORCH_MCPU_STREAM_WORKER_CORE", "7,nope");

  orStream_t stream = nullptr;
  ASSERT_EQ(orStreamCreateWithPriority(&stream, orStreamDefault, 0), orSuccess);
  EXPECT_EQ(stream->affinity_core, -1);
  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, AddTaskToStreamNullptr) {
  // Queueing work should fail fast if the stream handle is invalid.
  EXPECT_EQ(openreg::addTaskToStream(nullptr, [] {}), orErrorUnknown);
}

TEST_F(StreamTest, StreamQuery) {
  orStream_t stream = nullptr;
  EXPECT_EQ(orStreamCreate(&stream), orSuccess);

  EXPECT_EQ(orStreamQuery(stream), orSuccess);

  std::atomic<int> counter{0};
  openreg::addTaskToStream(stream, [&] { counter++; });

  EXPECT_EQ(orStreamSynchronize(stream), orSuccess);
  EXPECT_EQ(orStreamQuery(stream), orSuccess);

  EXPECT_EQ(orStreamDestroy(stream), orSuccess);
}

TEST_F(StreamTest, DeviceSynchronize) {
  orStream_t stream1 = nullptr;
  orStream_t stream2 = nullptr;

  EXPECT_EQ(orStreamCreate(&stream1), orSuccess);
  EXPECT_EQ(orStreamCreate(&stream2), orSuccess);

  std::atomic<int> counter{0};
  openreg::addTaskToStream(stream1, [&] { counter++; });
  openreg::addTaskToStream(stream2, [&] { counter++; });

  EXPECT_EQ(orDeviceSynchronize(), orSuccess);
  EXPECT_EQ(counter.load(), 2);

  EXPECT_EQ(orStreamDestroy(stream1), orSuccess);
  EXPECT_EQ(orStreamDestroy(stream2), orSuccess);
}

TEST_F(StreamTest, DeviceSynchronizeWithNoStreams) {
  // Even without registered streams, device sync should succeed.
  EXPECT_EQ(orDeviceSynchronize(), orSuccess);
}

TEST_F(StreamTest, StreamPriorityRange) {
  int min_p = -1;
  int max_p = -1;
  // OpenReg currently exposes only one priority level; verify the fixed range.
  EXPECT_EQ(orDeviceGetStreamPriorityRange(&min_p, &max_p), orSuccess);
  EXPECT_EQ(min_p, 0);
  EXPECT_EQ(max_p, 1);
}

} // namespace
