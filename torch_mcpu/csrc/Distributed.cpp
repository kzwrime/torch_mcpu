#include "Distributed.h"

#include <ATen/ATen.h>
#include <ATen/ThreadLocalState.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <csrc/runtime/OpenRegEvent.h>
#include <pybind11/stl.h>
#include <runtime/McpuKernelLaunch.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/utils/pybind.h>

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

bool is_mcpu_tensor(const at::Tensor& tensor) {
  return tensor.defined() &&
      tensor.device().type() == c10::DeviceType::PrivateUse1;
}

at::Tensor cpu_view_from_mcpu_tensor(const at::Tensor& tensor) {
  if (!is_mcpu_tensor(tensor)) {
    return tensor;
  }
  if (tensor.numel() == 0) {
    return at::empty(
        tensor.sizes(), tensor.options().device(c10::DeviceType::CPU));
  }
  return at::from_blob(
      tensor.data_ptr(),
      tensor.sizes(),
      tensor.strides(),
      /*deleter=*/[base = tensor](void*) {},
      tensor.options().device(c10::DeviceType::CPU));
}

std::vector<at::Tensor> cpu_views_from_tensors(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> cpu_tensors;
  cpu_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    cpu_tensors.push_back(cpu_view_from_mcpu_tensor(tensor));
  }
  return cpu_tensors;
}

std::vector<std::vector<at::Tensor>> cpu_views_from_nested_tensors(
    const std::vector<std::vector<at::Tensor>>& tensors) {
  std::vector<std::vector<at::Tensor>> cpu_tensors;
  cpu_tensors.reserve(tensors.size());
  for (const auto& tensor_list : tensors) {
    cpu_tensors.push_back(cpu_views_from_tensors(tensor_list));
  }
  return cpu_tensors;
}

const at::Tensor& find_mcpu_stream_tensor(
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    if (is_mcpu_tensor(tensor)) {
      return tensor;
    }
  }
  TORCH_CHECK(false, "MCPU ProcessGroup backend expects an mcpu tensor");
}

const at::Tensor& find_mcpu_stream_tensor(
    const std::vector<std::vector<at::Tensor>>& tensors) {
  for (const auto& tensor_list : tensors) {
    for (const auto& tensor : tensor_list) {
      if (is_mcpu_tensor(tensor)) {
        return tensor;
      }
    }
  }
  TORCH_CHECK(false, "MCPU ProcessGroup backend expects an mcpu tensor");
}

class McpuTensorMemoryGuard {
 public:
  explicit McpuTensorMemoryGuard(const std::vector<at::Tensor>& tensors) {
    unprotect(tensors);
  }

  explicit McpuTensorMemoryGuard(
      const std::vector<std::vector<at::Tensor>>& nested_tensors) {
    unprotect(nested_tensors);
  }

  McpuTensorMemoryGuard(
      const std::vector<at::Tensor>& tensors,
      const std::vector<std::vector<at::Tensor>>& nested_tensors) {
    unprotect(tensors);
    unprotect(nested_tensors);
  }

  ~McpuTensorMemoryGuard() noexcept {
    for (void* ptr : unprotected_pointers_) {
      at::mcpu::detail::protect_memory(ptr);
    }
  }

  McpuTensorMemoryGuard(const McpuTensorMemoryGuard&) = delete;
  McpuTensorMemoryGuard& operator=(const McpuTensorMemoryGuard&) = delete;

 private:
  void unprotect(const std::vector<at::Tensor>& tensors) {
    for (const auto& tensor : tensors) {
      at::mcpu::detail::unprotect_tensor_memory(tensor, unprotected_pointers_);
    }
  }

  void unprotect(const std::vector<std::vector<at::Tensor>>& nested_tensors) {
    for (const auto& tensor_list : nested_tensors) {
      unprotect(tensor_list);
    }
  }

  std::unordered_set<void*> unprotected_pointers_;
};

class McpuProcessGroupWork : public c10d::Work {
 public:
  McpuProcessGroupWork(
      int rank,
      c10d::OpType op_type,
      c10::DeviceIndex device,
      std::optional<int> source_rank = std::nullopt,
      bool host_wait_required = false)
      : c10d::Work(rank, op_type),
        device_(device),
        source_rank_(source_rank),
        host_wait_required_(host_wait_required) {}

  bool isCompleted() override {
    return exception() || event_.query();
  }

  bool isSuccess() const override {
    return !exception();
  }

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    synchronize();
    if (host_wait_required_ || timeout != kNoTimeout) {
      const auto start = std::chrono::steady_clock::now();
      while (!isCompleted()) {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
        if (timeout != kNoTimeout) {
          TORCH_CHECK(
              elapsed < timeout,
              "MCPU ProcessGroup work timed out after ",
              timeout.count(),
              " milliseconds");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    if (auto work_exception = exception()) {
      std::rethrow_exception(work_exception);
    }
    return true;
  }

  void synchronize() override {
    event_.block(c10::mcpu::getCurrentMcpuStream(device_));
    if (auto work_exception = exception()) {
      std::rethrow_exception(work_exception);
    }
  }

  void blockCurrentStream() override {
    synchronize();
  }

  std::vector<at::Tensor> result() override {
    return {};
  }

  int sourceRank() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(
        source_rank_.has_value(),
        "MCPU ProcessGroup work does not have a source rank");
    return *source_rank_;
  }

  void record(c10::mcpu::McpuStream stream) {
    event_.record(stream);
  }

  void setSourceRank(int source_rank) {
    std::lock_guard<std::mutex> lock(mutex_);
    source_rank_ = source_rank;
  }

  void markFinished(std::exception_ptr exception = nullptr) {
    finish(exception);
  }

 private:
  c10::DeviceIndex device_;
  std::optional<int> source_rank_;
  bool host_wait_required_;
  c10::mcpu::McpuEvent event_{false};
};

class McpuProcessGroupBackend : public c10d::Backend {
 public:
  explicit McpuProcessGroupBackend(
      c10::intrusive_ptr<c10d::Backend> cpu_backend)
      : c10d::Backend(cpu_backend->getRank(), cpu_backend->getSize()),
        cpu_backend_(std::move(cpu_backend)),
        options_(c10::make_intrusive<c10d::Backend::Options>("mcpu")) {}

  const std::string getBackendName() const override {
    return "mcpu";
  }

  c10::intrusive_ptr<Options> getBackendOptions() override {
    return options_;
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    cpu_backend_->setTimeout(timeout);
    options_->timeout = timeout;
  }

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override {
    auto task_tensors = tensors;
    auto task_opts = sync_options(opts);
    return enqueue(
        find_mcpu_stream_tensor(tensors),
        c10d::OpType::BROADCAST,
        [=, this](const c10::intrusive_ptr<McpuProcessGroupWork>&) mutable {
          McpuTensorMemoryGuard guard(task_tensors);
          auto cpu_tensors = cpu_views_from_tensors(task_tensors);
          wait_cpu_work(cpu_backend_->broadcast(cpu_tensors, task_opts));
        });
  }

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override {
    auto task_tensors = tensors;
    auto task_opts = sync_options(opts);
    return enqueue(
        find_mcpu_stream_tensor(tensors),
        c10d::OpType::ALLREDUCE,
        [=, this](const c10::intrusive_ptr<McpuProcessGroupWork>&) mutable {
          McpuTensorMemoryGuard guard(task_tensors);
          auto cpu_tensors = cpu_views_from_tensors(task_tensors);
          wait_cpu_work(cpu_backend_->allreduce(cpu_tensors, task_opts));
        });
  }

  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override {
    auto task_tensors = tensors;
    auto task_opts = sync_options(opts);
    return enqueue(
        find_mcpu_stream_tensor(tensors),
        c10d::OpType::REDUCE,
        [=, this](const c10::intrusive_ptr<McpuProcessGroupWork>&) mutable {
          McpuTensorMemoryGuard guard(task_tensors);
          auto cpu_tensors = cpu_views_from_tensors(task_tensors);
          wait_cpu_work(cpu_backend_->reduce(cpu_tensors, task_opts));
        });
  }

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    auto task_outputs = output_tensors;
    auto task_inputs = input_tensors;
    auto task_opts = sync_options(opts);
    return enqueue(
        find_mcpu_stream_tensor(input_tensors),
        c10d::OpType::ALLGATHER,
        [=, this](const c10::intrusive_ptr<McpuProcessGroupWork>&) mutable {
          McpuTensorMemoryGuard guard(task_inputs, task_outputs);
          auto cpu_outputs = cpu_views_from_nested_tensors(task_outputs);
          auto cpu_inputs = cpu_views_from_tensors(task_inputs);
          wait_cpu_work(
              cpu_backend_->allgather(cpu_outputs, cpu_inputs, task_opts));
        });
  }

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& output_buffer,
      at::Tensor& input_buffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    auto task_output = output_buffer;
    auto task_input = input_buffer;
    auto task_opts = sync_options(opts);
    return enqueue(
        input_buffer,
        c10d::OpType::_ALLGATHER_BASE,
        [=, this](const c10::intrusive_ptr<McpuProcessGroupWork>&) mutable {
          std::vector<at::Tensor> tensors{task_output, task_input};
          McpuTensorMemoryGuard guard(tensors);
          auto cpu_output = cpu_view_from_mcpu_tensor(task_output);
          auto cpu_input = cpu_view_from_mcpu_tensor(task_input);
          wait_cpu_work(
              cpu_backend_->_allgather_base(cpu_output, cpu_input, task_opts));
        });
  }

  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    auto task_outputs = outputs;
    auto task_inputs = inputs;
    auto task_opts = sync_options(opts);
    return enqueue(
        find_mcpu_stream_tensor(inputs),
        c10d::OpType::ALLGATHER_COALESCED,
        [=, this](const c10::intrusive_ptr<McpuProcessGroupWork>&) mutable {
          std::vector<at::Tensor> tensors = task_outputs;
          tensors.insert(tensors.end(), task_inputs.begin(), task_inputs.end());
          McpuTensorMemoryGuard guard(tensors);
          auto cpu_outputs = cpu_views_from_tensors(task_outputs);
          auto cpu_inputs = cpu_views_from_tensors(task_inputs);
          wait_cpu_work(cpu_backend_->allgather_into_tensor_coalesced(
              cpu_outputs, cpu_inputs, task_opts));
        });
  }

  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override {
    auto task_outputs = output_tensors;
    auto task_inputs = input_tensors;
    auto task_opts = sync_options(opts);
    return enqueue(
        find_mcpu_stream_tensor(input_tensors),
        c10d::OpType::GATHER,
        [=, this](const c10::intrusive_ptr<McpuProcessGroupWork>&) mutable {
          McpuTensorMemoryGuard guard(task_inputs, task_outputs);
          auto cpu_outputs = cpu_views_from_nested_tensors(task_outputs);
          auto cpu_inputs = cpu_views_from_tensors(task_inputs);
          wait_cpu_work(
              cpu_backend_->gather(cpu_outputs, cpu_inputs, task_opts));
        });
  }

  c10::intrusive_ptr<c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dst_rank,
      int tag) override {
    auto task_tensors = tensors;
    return enqueue(
        find_mcpu_stream_tensor(tensors),
        c10d::OpType::SEND,
        [=, this](const c10::intrusive_ptr<McpuProcessGroupWork>&) mutable {
          McpuTensorMemoryGuard guard(task_tensors);
          auto cpu_tensors = cpu_views_from_tensors(task_tensors);
          wait_cpu_work(cpu_backend_->send(cpu_tensors, dst_rank, tag));
        });
  }

  c10::intrusive_ptr<c10d::Work> recv(
      std::vector<at::Tensor>& tensors,
      int src_rank,
      int tag) override {
    auto task_tensors = tensors;
    return enqueue(
        find_mcpu_stream_tensor(tensors),
        c10d::OpType::RECV,
        [=, this](const c10::intrusive_ptr<McpuProcessGroupWork>&) mutable {
          McpuTensorMemoryGuard guard(task_tensors);
          auto cpu_tensors = cpu_views_from_tensors(task_tensors);
          wait_cpu_work(cpu_backend_->recv(cpu_tensors, src_rank, tag));
        },
        src_rank);
  }

  c10::intrusive_ptr<c10d::Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override {
    auto task_tensors = tensors;
    return enqueue(
        find_mcpu_stream_tensor(tensors),
        c10d::OpType::RECV,
        [=,
         this](const c10::intrusive_ptr<McpuProcessGroupWork>& work) mutable {
          McpuTensorMemoryGuard guard(task_tensors);
          auto cpu_tensors = cpu_views_from_tensors(task_tensors);
          auto cpu_work = cpu_backend_->recvAnysource(cpu_tensors, tag);
          wait_cpu_work(cpu_work);
          if (cpu_work) {
            work->setSourceRank(cpu_work->sourceRank());
          }
        },
        std::nullopt,
        /*host_wait_required=*/true);
  }

 private:
  template <typename OptionsT>
  static OptionsT sync_options(OptionsT opts) {
    opts.asyncOp = false;
    return opts;
  }

  static void wait_cpu_work(const c10::intrusive_ptr<c10d::Work>& work) {
    if (work) {
      work->wait();
    }
  }

  c10::intrusive_ptr<c10d::Work> enqueue(
      const at::Tensor& stream_tensor,
      c10d::OpType op_type,
      std::function<void(const c10::intrusive_ptr<McpuProcessGroupWork>&)> task,
      std::optional<int> source_rank = std::nullopt,
      bool host_wait_required = false) {
    auto device = stream_tensor.device().index();
    auto stream = c10::mcpu::getCurrentMcpuStream(device);
    auto work = c10::make_intrusive<McpuProcessGroupWork>(
        cpu_backend_->getRank(),
        op_type,
        device,
        source_rank,
        host_wait_required);
    auto thread_local_state = at::ThreadLocalState();
    auto record_name =
        std::string("mcpu::distributed::") + c10d::opTypeToString(op_type);
    at::mcpu::detail::launch_kernel_task(
        stream_tensor,
        [work,
         record_name = std::move(record_name),
         thread_local_state = std::move(thread_local_state),
         task = std::move(task)]() mutable {
          at::ThreadLocalStateGuard thread_local_state_guard(
              thread_local_state);
          RECORD_USER_SCOPE(record_name.c_str());
          at::mcpu::KernelTaskScope kernel_task;
          try {
            task(work);
            work->markFinished();
          } catch (...) {
            work->markFinished(std::current_exception());
          }
        });
    work->record(stream);
    return work;
  }

  c10::intrusive_ptr<c10d::Backend> cpu_backend_;
  c10::intrusive_ptr<c10d::Backend::Options> options_;
};

} // namespace

void bindMcpuDistributed(pybind11::module& module) {
  module.def(
      "_make_mcpu_process_group_backend",
      [](const c10::intrusive_ptr<c10d::Backend>& cpu_backend)
          -> c10::intrusive_ptr<c10d::Backend> {
        return c10::make_intrusive<McpuProcessGroupBackend>(cpu_backend);
      },
      pybind11::arg("cpu_backend"));
}
