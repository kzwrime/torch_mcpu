#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>

#include <pybind11/chrono.h>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <runtime/McpuKernelLaunch.h>
#include <csrc/runtime/OpenRegException.h>
#include <runtime/OpenRegFunctions.h>

// Adapted from PyTorch sources:
// - ../pytorch/torch/csrc/distributed/c10d/Backend.hpp
// - ../pytorch/torch/csrc/distributed/c10d/Work.hpp
// - ../pytorch/torch/csrc/distributed/c10d/ProcessGroupGloo.cpp
// - ../pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
// - ../pytorch/torch/distributed/distributed_c10d.py
// - ../pytorch/torch/testing/_internal/distributed/fake_pg.py
// - torch_mcpu/csrc/aten/UVAView.cpp
// - torch_mcpu/csrc/runtime/McpuKernelLaunch.cpp
//
// This backend uses ProcessGroupGloo as the transport and stages mcpu tensors
// through CPU tensors. Submission returns immediately by moving the transport
// wait to a detached host thread; that is the important property for P2P
// correctness because both sides can enqueue isend/irecv before either side
// waits. The host thread synchronizes the mcpu device before reading inputs and
// before publishing completion so it does not expose partially copied results.

namespace c10d {
namespace {

static void assert_mcpu_tensors(const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    TORCH_CHECK(
        tensor.device().type() == c10::DeviceType::PrivateUse1,
        "Expected mcpu tensors, got ",
        tensor.device());
  }
}

static std::vector<at::Tensor> flatten_tensor_lists(
    const std::vector<std::vector<at::Tensor>>& tensor_lists) {
  std::vector<at::Tensor> flattened;
  for (const auto& list : tensor_lists) {
    flattened.insert(flattened.end(), list.begin(), list.end());
  }
  return flattened;
}

static void copy_cpu_tensors_to_mcpu(
    std::vector<at::Tensor>& dst_tensors,
    const std::vector<at::Tensor>& src_tensors) {
  TORCH_CHECK(
      dst_tensors.size() == src_tensors.size(),
      "Mismatched tensor counts while copying communication results");
  for (const auto i : c10::irange(dst_tensors.size())) {
    dst_tensors[i].copy_(src_tensors[i]);
  }
}

static void copy_cpu_tensors_to_mcpu(
    std::vector<std::vector<at::Tensor>>& dst_tensors,
    const std::vector<std::vector<at::Tensor>>& src_tensors) {
  TORCH_CHECK(
      dst_tensors.size() == src_tensors.size(),
      "Mismatched nested tensor counts while copying communication results");
  for (const auto i : c10::irange(dst_tensors.size())) {
    TORCH_CHECK(
        dst_tensors[i].size() == src_tensors[i].size(),
        "Mismatched nested tensor counts while copying communication results");
    for (const auto j : c10::irange(dst_tensors[i].size())) {
      dst_tensors[i][j].copy_(src_tensors[i][j]);
    }
  }
}

// Adapted from torch_mcpu/csrc/aten/UVAView.cpp.
static void unprotect_mcpu_tensor_memory(const at::Tensor& mcpu_tensor) {
  TORCH_CHECK(
      mcpu_tensor.device().type() == c10::DeviceType::PrivateUse1,
      "Input tensor must be on mcpu");

  if (!mcpu_tensor.has_storage() || mcpu_tensor.numel() == 0) {
    return;
  }

  orPointerAttributes attr;
  auto attr_status = orPointerGetAttributes(&attr, mcpu_tensor.data_ptr());
  if (attr_status != orSuccess || attr.type != orMemoryTypeDevice) {
    return;
  }

  auto unprotect_status = orMemoryUnprotect(attr.pointer);
  TORCH_CHECK(
      unprotect_status == orSuccess, "Failed to unprotect mcpu tensor memory");
}

static void protect_mcpu_tensor_memory(const at::Tensor& mcpu_tensor) {
  if (!mcpu_tensor.defined() || !mcpu_tensor.has_storage() ||
      mcpu_tensor.numel() == 0) {
    return;
  }
  at::mcpu::detail::protect_memory(mcpu_tensor.data_ptr());
}

static py::object create_gloo_backend(
    py::object store,
    int rank,
    int size,
    std::chrono::milliseconds timeout) {
  py::gil_scoped_acquire gil;
  auto dist_c10d = py::module_::import("torch.distributed.distributed_c10d");
  return dist_c10d.attr("ProcessGroupGloo")(std::move(store), rank, size, timeout);
}

} // namespace

class McpuWork final : public Work {
 public:
  struct WorkState {
    explicit WorkState(c10::Device device) : device(std::move(device)) {}

    c10::Device device;
    std::mutex mutex;
    std::condition_variable cv;
    bool comm_done{false};
    std::optional<int> source_rank;
    std::exception_ptr exception;
  };

  McpuWork(
      c10::intrusive_ptr<Backend> backend,
      int rank,
      OpType op_type,
      std::vector<at::Tensor> tensors,
      std::shared_ptr<WorkState> state)
      : Work(rank, op_type),
        backend_(std::move(backend)),
        tensors_(std::move(tensors)),
        state_(std::move(state)) {}

  bool isCompleted() override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->comm_done;
  }

  bool isSuccess() const override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->comm_done && !state_->exception;
  }

  std::exception_ptr exception() const override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->exception;
  }

  int sourceRank() const override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    TORCH_CHECK(
        state_->source_rank.has_value(),
        "sourceRank() may only be called on recv or recv-from-any work");
    return *state_->source_rank;
  }

  std::vector<at::Tensor> result() override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    TORCH_CHECK(state_->comm_done, "Work must be completed before result()");
    return tensors_;
  }

  void synchronize() override {
    // Host-side synchronization is handled in wait(); this backend avoids
    // blocking a mcpu worker thread on completion.
  }

  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override {
    const bool has_gil = PyGILState_Check();
    if (has_gil) {
      py::gil_scoped_release release;
      std::unique_lock<std::mutex> lock(state_->mutex);
      if (timeout == kNoTimeout) {
        state_->cv.wait(lock, [&] { return state_->comm_done; });
      } else if (!state_->cv.wait_for(lock, timeout, [&] {
                   return state_->comm_done;
                 })) {
        TORCH_CHECK(false, "Operation timed out!");
      }
    } else {
      std::unique_lock<std::mutex> lock(state_->mutex);
      if (timeout == kNoTimeout) {
        state_->cv.wait(lock, [&] { return state_->comm_done; });
      } else if (!state_->cv.wait_for(lock, timeout, [&] {
                   return state_->comm_done;
                 })) {
        TORCH_CHECK(false, "Operation timed out!");
      }
    }

    std::exception_ptr exception;
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      exception = state_->exception;
    }
    if (exception) {
      std::rethrow_exception(exception);
    }

    return true;
  }

  void blockCurrentStream() override {
    synchronize();
  }

  void abort() override {
    TORCH_CHECK(false, "McpuWork::abort not implemented.");
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
    TORCH_CHECK(false, "McpuWork::getFuture not implemented.");
  }

  c10::intrusive_ptr<c10::ivalue::Future> getFutureResult() override {
    TORCH_CHECK(false, "McpuWork::getFutureResult not implemented.");
  }

 private:
  c10::intrusive_ptr<Backend> backend_;
  std::vector<at::Tensor> tensors_;
  std::shared_ptr<WorkState> state_;
};

class McpuBackend final : public Backend {
 public:
  struct Options final : public Backend::Options {
    explicit Options(std::chrono::milliseconds timeout = kBackendDefaultTimeout)
        : Backend::Options("mcpu", timeout) {}
  };

 public:

  McpuBackend(
      c10::intrusive_ptr<Store> store,
      int64_t rank,
      int64_t size,
      int64_t timeout_ms)
      : Backend(static_cast<int>(rank), static_cast<int>(size)),
        options_(c10::make_intrusive<Options>(std::chrono::milliseconds(timeout_ms))),
        gloo_backend_(create_gloo_backend(
            py::cast(store),
            static_cast<int>(rank),
            static_cast<int>(size),
            std::chrono::milliseconds(timeout_ms))) {}

  std::vector<at::Tensor> toCpuViews(
      const std::vector<at::Tensor>& tensors) const {
    std::vector<at::Tensor> cpu_tensors;
    cpu_tensors.reserve(tensors.size());
    for (const auto& tensor : tensors) {
      cpu_tensors.push_back(tensor.to(c10::DeviceType::CPU));
    }
    return cpu_tensors;
  }

  template <typename OpFn>
  c10::intrusive_ptr<Work> submitOp(
      OpType opType,
      std::vector<at::Tensor> tensors,
      OpFn&& opFn,
      std::optional<int> sourceRank = std::nullopt,
      std::shared_ptr<std::optional<int>> sourceRankResult = nullptr) {
    assert_mcpu_tensors(tensors);
    if (tensors.empty()) {
      tensors.emplace_back(at::empty(
          {1},
          at::TensorOptions()
              .device(c10::DeviceType::PrivateUse1)
              .dtype(at::kByte)));
    }

    auto device = tensors.front().device();
    auto state = std::make_shared<McpuWork::WorkState>(device);
    state->source_rank = sourceRank;
    auto backend_ref =
        c10::intrusive_ptr<Backend>::unsafe_reclaim_from_nonowning(this);
    auto work = c10::make_intrusive<McpuWork>(
        backend_ref,
        getRank(),
        opType,
        tensors,
        state);

    std::thread(
        [state,
         backend_ref = backend_ref,
         device,
         tensors = tensors,
         opFn = std::forward<OpFn>(opFn),
         sourceRankResult]() mutable {
          try {
            c10::mcpu::set_device(device.index());
            MCPU_CHECK(orDeviceSynchronize());

            for (const auto& tensor : tensors) {
              unprotect_mcpu_tensor_memory(tensor);
            }

            opFn();

            // Ensure any mcpu-side copies launched by the transport callback
            // have actually completed before memory is re-protected or the
            // work is marked done. This keeps the host-thread transport path
            // aligned with the async stream contract used elsewhere in the
            // backend.
            MCPU_CHECK(orDeviceSynchronize());

            for (const auto& tensor : tensors) {
              protect_mcpu_tensor_memory(tensor);
            }
          } catch (...) {
            for (const auto& tensor : tensors) {
              protect_mcpu_tensor_memory(tensor);
            }
            std::lock_guard<std::mutex> lock(state->mutex);
            state->exception = std::current_exception();
          }

          {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->comm_done = true;
            if (sourceRankResult && sourceRankResult->has_value()) {
              state->source_rank = **sourceRankResult;
            }
          }
          state->cv.notify_all();
        }).detach();

    return work;
  }

  template <typename OpFn>
  c10::intrusive_ptr<Work> submitPointToPoint(
      OpType opType,
      std::vector<at::Tensor>& tensors,
      int peerRank,
      int tag,
      OpFn&& opFn,
      std::optional<int> sourceRank = std::nullopt,
      std::shared_ptr<std::optional<int>> sourceRankResult = nullptr) {
    (void)peerRank;
    (void)tag;
    TORCH_CHECK(
        tensors.size() == 1, "Point-to-point ops expect a single tensor");
    return submitOp(
        opType,
        tensors,
        std::forward<OpFn>(opFn),
        sourceRank,
        std::move(sourceRankResult));
  }

  const std::string getBackendName() const override {
    return "mcpu";
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return options_;
  }

  void setTimeout(std::chrono::milliseconds timeout) override {
    options_->timeout = timeout;
    py::gil_scoped_acquire gil;
    gloo_backend_.attr("setTimeout")(timeout);
  }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override {
    return submitOp(OpType::BROADCAST, tensors, [this, tensors, opts]() mutable {
      auto cpu_tensors = toCpuViews(tensors);
      py::gil_scoped_acquire gil;
      auto work = gloo_backend_.attr("broadcast")(cpu_tensors, opts);
      work.attr("wait")();
      copy_cpu_tensors_to_mcpu(tensors, cpu_tensors);
    });
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override {
    return submitOp(OpType::ALLREDUCE, tensors, [this, tensors, opts]() mutable {
      auto cpu_tensors = toCpuViews(tensors);
      py::gil_scoped_acquire gil;
      auto work = gloo_backend_.attr("allreduce")(cpu_tensors, opts);
      work.attr("wait")();
      copy_cpu_tensors_to_mcpu(tensors, cpu_tensors);
    });
  }

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts = AllreduceCoalescedOptions())
      override {
    return submitOp(
        OpType::ALLREDUCE_COALESCED, tensors, [this, tensors, opts]() mutable {
          auto cpu_tensors = toCpuViews(tensors);
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("allreduce_coalesced")(cpu_tensors, opts);
          work.attr("wait")();
          copy_cpu_tensors_to_mcpu(tensors, cpu_tensors);
        });
  }

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override {
    return submitOp(OpType::REDUCE, tensors, [this, tensors, opts]() mutable {
      auto cpu_tensors = toCpuViews(tensors);
      py::gil_scoped_acquire gil;
      auto work = gloo_backend_.attr("reduce")(cpu_tensors, opts);
      work.attr("wait")();
      copy_cpu_tensors_to_mcpu(tensors, cpu_tensors);
    });
  }

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    auto tensors = flatten_tensor_lists(outputTensors);
    tensors.insert(tensors.end(), inputTensors.begin(), inputTensors.end());
    return submitOp(
        OpType::ALLGATHER,
        tensors,
        [this, outputTensors, inputTensors, opts]() mutable {
          auto cpu_outputs = outputTensors;
          auto cpu_inputs = inputTensors;
          for (auto& output_list : cpu_outputs) {
            for (auto& tensor : output_list) {
              tensor = tensor.to(c10::DeviceType::CPU);
            }
          }
          for (auto& tensor : cpu_inputs) {
            tensor = tensor.to(c10::DeviceType::CPU);
          }
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("allgather")(cpu_outputs, cpu_inputs, opts);
          work.attr("wait")();
          copy_cpu_tensors_to_mcpu(outputTensors, cpu_outputs);
        });
  }

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    std::vector<at::Tensor> tensors{outputBuffer, inputBuffer};
    return submitOp(
        OpType::_ALLGATHER_BASE,
        tensors,
        [this, outputBuffer, inputBuffer, opts]() mutable {
          auto cpu_output = outputBuffer.to(c10::DeviceType::CPU);
          auto cpu_input = inputBuffer.to(c10::DeviceType::CPU);
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("_allgather_base")(cpu_output, cpu_input, opts);
          work.attr("wait")();
          outputBuffer.copy_(cpu_output);
        });
  }

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override {
    auto tensors = outputs;
    tensors.insert(tensors.end(), inputs.begin(), inputs.end());
    return submitOp(
        OpType::ALLGATHER_COALESCED,
        tensors,
        [this, outputs, inputs, opts]() mutable {
          auto cpu_outputs = outputs;
          auto cpu_inputs = inputs;
          for (auto& tensor : cpu_outputs) {
            tensor = tensor.to(c10::DeviceType::CPU);
          }
          for (auto& tensor : cpu_inputs) {
            tensor = tensor.to(c10::DeviceType::CPU);
          }
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("allgather_into_tensor_coalesced")(
              cpu_outputs, cpu_inputs, opts);
          work.attr("wait")();
          copy_cpu_tensors_to_mcpu(outputs, cpu_outputs);
        });
  }

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override {
    auto tensors = flatten_tensor_lists(outputTensors);
    tensors.insert(tensors.end(), inputTensors.begin(), inputTensors.end());
    return submitOp(
        OpType::GATHER,
        tensors,
        [this, outputTensors, inputTensors, opts]() mutable {
          auto cpu_outputs = outputTensors;
          auto cpu_inputs = inputTensors;
          for (auto& output_list : cpu_outputs) {
            for (auto& tensor : output_list) {
              tensor = tensor.to(c10::DeviceType::CPU);
            }
          }
          for (auto& tensor : cpu_inputs) {
            tensor = tensor.to(c10::DeviceType::CPU);
          }
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("gather")(cpu_outputs, cpu_inputs, opts);
          work.attr("wait")();
          copy_cpu_tensors_to_mcpu(outputTensors, cpu_outputs);
        });
  }

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override {
    auto tensors = outputTensors;
    auto flat_inputs = flatten_tensor_lists(inputTensors);
    tensors.insert(tensors.end(), flat_inputs.begin(), flat_inputs.end());
    return submitOp(
        OpType::SCATTER,
        tensors,
        [this, outputTensors, inputTensors, opts]() mutable {
          auto cpu_outputs = outputTensors;
          auto cpu_inputs = inputTensors;
          for (auto& tensor : cpu_outputs) {
            tensor = tensor.to(c10::DeviceType::CPU);
          }
          for (auto& input_list : cpu_inputs) {
            for (auto& tensor : input_list) {
              tensor = tensor.to(c10::DeviceType::CPU);
            }
          }
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("scatter")(cpu_outputs, cpu_inputs, opts);
          work.attr("wait")();
          copy_cpu_tensors_to_mcpu(outputTensors, cpu_outputs);
        });
  }

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    auto tensors = outputTensors;
    auto flat_inputs = flatten_tensor_lists(inputTensors);
    tensors.insert(tensors.end(), flat_inputs.begin(), flat_inputs.end());
    return submitOp(
        OpType::REDUCE_SCATTER,
        tensors,
        [this, outputTensors, inputTensors, opts]() mutable {
          auto cpu_outputs = outputTensors;
          auto cpu_inputs = inputTensors;
          for (auto& tensor : cpu_outputs) {
            tensor = tensor.to(c10::DeviceType::CPU);
          }
          for (auto& input_list : cpu_inputs) {
            for (auto& tensor : input_list) {
              tensor = tensor.to(c10::DeviceType::CPU);
            }
          }
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("reduce_scatter")(cpu_outputs, cpu_inputs, opts);
          work.attr("wait")();
          copy_cpu_tensors_to_mcpu(outputTensors, cpu_outputs);
        });
  }

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override {
    std::vector<at::Tensor> tensors{outputBuffer, inputBuffer};
    return submitOp(
        OpType::_REDUCE_SCATTER_BASE,
        tensors,
        [this, outputBuffer, inputBuffer, opts]() mutable {
          auto cpu_output = outputBuffer.to(c10::DeviceType::CPU);
          auto cpu_input = inputBuffer.to(c10::DeviceType::CPU);
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("_reduce_scatter_base")(
              cpu_output, cpu_input, opts);
          work.attr("wait")();
          outputBuffer.copy_(cpu_output);
        });
  }

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    auto tensors = outputTensors;
    tensors.insert(tensors.end(), inputTensors.begin(), inputTensors.end());
    return submitOp(
        OpType::ALLTOALL,
        tensors,
        [this, outputTensors, inputTensors, opts]() mutable {
          auto cpu_outputs = outputTensors;
          auto cpu_inputs = inputTensors;
          for (auto& tensor : cpu_outputs) {
            tensor = tensor.to(c10::DeviceType::CPU);
          }
          for (auto& tensor : cpu_inputs) {
            tensor = tensor.to(c10::DeviceType::CPU);
          }
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("alltoall")(cpu_outputs, cpu_inputs, opts);
          work.attr("wait")();
          copy_cpu_tensors_to_mcpu(outputTensors, cpu_outputs);
        });
  }

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override {
    std::vector<at::Tensor> tensors{outputBuffer, inputBuffer};
    return submitOp(
        OpType::ALLTOALL_BASE,
        tensors,
        [this,
         outputBuffer,
         inputBuffer,
         outputSplitSizes,
         inputSplitSizes,
         opts]() mutable {
          auto cpu_output = outputBuffer.to(c10::DeviceType::CPU);
          auto cpu_input = inputBuffer.to(c10::DeviceType::CPU);
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("alltoall_base")(
              cpu_output, cpu_input, outputSplitSizes, inputSplitSizes, opts);
          work.attr("wait")();
          outputBuffer.copy_(cpu_output);
        });
  }

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override {
    return submitPointToPoint(
        OpType::SEND,
        tensors,
        dstRank,
        tag,
        [this, tensors, dstRank, tag]() mutable {
          auto cpu_tensors = tensors;
          for (auto& tensor : cpu_tensors) {
            tensor = tensor.to(c10::DeviceType::CPU);
          }
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("send")(cpu_tensors, dstRank, tag);
          work.attr("wait")();
        });
  }

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override {
    return submitPointToPoint(
        OpType::RECV,
        tensors,
        srcRank,
        tag,
        [this, tensors, srcRank, tag]() mutable {
          auto cpu_tensors = tensors;
          for (auto& tensor : cpu_tensors) {
            tensor = tensor.to(c10::DeviceType::CPU);
          }
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("recv")(cpu_tensors, srcRank, tag);
          work.attr("wait")();
          copy_cpu_tensors_to_mcpu(tensors, cpu_tensors);
        },
        std::optional<int>(srcRank));
  }

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int tag) override {
    auto source_rank_result = std::make_shared<std::optional<int>>();
    return submitPointToPoint(
        OpType::RECVANYSOURCE,
        tensors,
        -1,
        tag,
        [this, tensors, tag, source_rank_result]() mutable {
          auto cpu_tensors = tensors;
          for (auto& tensor : cpu_tensors) {
            tensor = tensor.to(c10::DeviceType::CPU);
          }
          py::gil_scoped_acquire gil;
          auto work = gloo_backend_.attr("recv_anysource")(cpu_tensors, tag);
          work.attr("wait")();
          *source_rank_result = work.attr("_source_rank")().cast<int>();
          copy_cpu_tensors_to_mcpu(tensors, cpu_tensors);
        },
        std::nullopt,
        source_rank_result);
  }

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override {
    std::vector<at::Tensor> tensors{at::empty(
      {1},
      at::TensorOptions()
          .device(c10::DeviceType::PrivateUse1)
          .dtype(at::kByte))};
    return submitOp(OpType::BARRIER, tensors, [this, tensors, opts]() mutable {
      auto cpu_tensors = tensors;
      for (auto& tensor : cpu_tensors) {
        tensor = tensor.to(c10::DeviceType::CPU);
      }
      py::gil_scoped_acquire gil;
      auto work = gloo_backend_.attr("barrier")(opts);
      work.attr("wait")();
    });
  }

  c10::intrusive_ptr<Options> options_;
  py::object gloo_backend_;
};

void registerMcpuDistributedBindings(PyObject* module) {
  py::module_ m = py::reinterpret_borrow<py::module_>(module);
  py::class_<McpuBackend, c10::intrusive_ptr<McpuBackend>, Backend>(
      m, "McpuBackend");
}

PyObject* _create_process_group_mcpu(
    PyObject* /*unused*/,
    PyObject* args) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      PyTuple_Check(args) && PyTuple_GET_SIZE(args) == 4,
      "_create_process_group_mcpu expects (store, rank, size, timeout)");

  py::object store_obj = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(args, 0));

  auto* rank_obj = PyTuple_GET_ITEM(args, 1);
  auto* size_obj = PyTuple_GET_ITEM(args, 2);
  TORCH_CHECK(
      THPUtils_checkLong(rank_obj) && THPUtils_checkLong(size_obj),
      "rank and size must be integers");
  int rank = static_cast<int>(THPUtils_unpackLong(rank_obj));
  int size = static_cast<int>(THPUtils_unpackLong(size_obj));

  py::object timeout_obj = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(args, 3));
  auto timeout = timeout_obj.cast<std::chrono::milliseconds>();

  auto backend = c10::make_intrusive<McpuBackend>(
      std::move(store_obj.cast<c10::intrusive_ptr<Store>>()),
      static_cast<int64_t>(rank),
      static_cast<int64_t>(size),
      static_cast<int64_t>(timeout.count()));
  py::object py_backend = py::cast(std::move(backend));
  return py_backend.release().ptr();
  END_HANDLE_TH_ERRORS
}

} // namespace c10d
