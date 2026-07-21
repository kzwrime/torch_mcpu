#include "include/openreg.h"

#include <atomic>
#include <cstdlib>
#include <iostream>

namespace {

void setDemoTimeout() {
#if defined(_WIN32)
  _putenv_s("TORCH_MCPU_STREAM_SYNC_TIMEOUT_MS", "1000");
#else
  setenv("TORCH_MCPU_STREAM_SYNC_TIMEOUT_MS", "1000", 1);
#endif
}

} // namespace

int main() {
  setDemoTimeout();

  orStream_t stream = nullptr;
  if (orStreamCreate(&stream) != orSuccess) {
    std::cerr << "failed to create stream\n";
    return 1;
  }

  std::atomic<bool> release{false};
  const auto launch_status = openreg::addNamedTaskToStream(
      stream, "example::intentional_stall", [&release] {
        while (!release.load(std::memory_order_acquire)) {
          openreg::cpu_relax();
        }
      });
  if (launch_status != orSuccess) {
    std::cerr << "failed to launch intentional stall task\n";
    orStreamDestroy(stream);
    return 1;
  }

  std::cout << "Waiting for the intentional stall. A timeout diagnostic "
               "should follow."
            << std::endl;
  const auto sync_status = orStreamSynchronize(stream);
  const bool observed_timeout = sync_status == orErrorTimeout;

  release.store(true, std::memory_order_release);
  const auto recovery_status = orStreamSynchronize(stream);
  const auto destroy_status = orStreamDestroy(stream);

  if (!observed_timeout) {
    std::cerr << "expected orErrorTimeout, got " << sync_status << '\n';
    return 1;
  }
  if (recovery_status != orSuccess || destroy_status != orSuccess) {
    std::cerr << "stream did not recover after releasing the task\n";
    return 1;
  }

  std::cout << "Observed the expected timeout and released the task safely.\n";
  return 0;
}
