#include <ATen/Context.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>

#include <c10/core/CachingDeviceAllocator.h>
#include <runtime/OpenRegFunctions.h>

// Forward-declare allocator management functions from libtorch_mcpu.so.
// (DeviceCachingAllocator.h transitively includes openreg.h which is not on
//  torch_bindings include path, so we use explicit declarations instead.)
namespace c10::mcpu {
void emptyCache(c10::MempoolId_t mempool_id = {0, 0});
c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device);
void resetPeakStats(c10::DeviceIndex device);
void resetAccumulatedStats(c10::DeviceIndex device);
} // namespace c10::mcpu

static PyObject* _initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  at::globalContext().lazyInitDevice(c10::DeviceType::PrivateUse1);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE MCPU GET DEFAULT GENERATOR
static PyObject* _getDefaultGenerator(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_get_default_generator expects an int, but got ",
      THPUtils_typename(arg));
  auto idx = static_cast<int>(THPUtils_unpackLong(arg));

  return THPGenerator_initDefaultGenerator(
      at::globalContext().defaultGenerator(
          c10::Device(c10::DeviceType::PrivateUse1, idx)));

  END_HANDLE_TH_ERRORS
}
// LITERALINCLUDE MCPU GET DEFAULT GENERATOR

// LITERALINCLUDE START: MODULE SET DEVICE HELPER

PyObject* _setDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  auto device = THPUtils_unpackDeviceIndex(arg);
  torch::utils::device_lazy_init(at::kPrivateUse1);
  c10::mcpu::set_device(device);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE END: MODULE SET DEVICE HELPER

PyObject* _exchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device = c10::mcpu::ExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDevice(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto device = static_cast<int32_t>(c10::mcpu::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* _getDeviceCount(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(c10::mcpu::device_count());
  END_HANDLE_TH_ERRORS
}

static PyObject* _emptyCache(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::mcpu::emptyCache();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _memoryStats(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "_memory_stats expects a device index");
  auto device = THPUtils_unpackDeviceIndex(arg);
  auto stats = c10::mcpu::getDeviceStats(device);

  PyObject* dict = PyDict_New();
  auto insert = [&](const char* key, size_t val) {
    PyObject* v = PyLong_FromSize_t(val);
    PyDict_SetItemString(dict, key, v);
    Py_DECREF(v);
  };
  auto insert_stat = [&](const char* prefix, const c10::CachingDeviceAllocator::Stat& s) {
    std::string cur = std::string(prefix) + ".current";
    std::string peak = std::string(prefix) + ".peak";
    std::string alloc = std::string(prefix) + ".allocated";
    std::string freed = std::string(prefix) + ".freed";
    insert(cur.c_str(), static_cast<size_t>(s.current));
    insert(peak.c_str(), static_cast<size_t>(s.peak));
    insert(alloc.c_str(), static_cast<size_t>(s.allocated));
    insert(freed.c_str(), static_cast<size_t>(s.freed));
  };
  insert_stat("allocated_bytes.all", stats.allocated_bytes[0]);
  insert_stat("reserved_bytes.all", stats.reserved_bytes[0]);
  insert_stat("active_bytes.all", stats.active_bytes[0]);
  insert("num_alloc_retries", static_cast<size_t>(stats.num_alloc_retries));
  return dict;
  END_HANDLE_TH_ERRORS
}

static PyObject* _resetPeakMemoryStats(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "_reset_peak_memory_stats expects a device index");
  auto device = THPUtils_unpackDeviceIndex(arg);
  c10::mcpu::resetPeakStats(device);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _resetAccumulatedMemoryStats(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "_reset_accumulated_memory_stats expects a device index");
  auto device = THPUtils_unpackDeviceIndex(arg);
  c10::mcpu::resetAccumulatedStats(device);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* _getStreamPriorityRange(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  int least = 0, greatest = 0;
  c10::mcpu::getStreamPriorityRange(&least, &greatest);
  return Py_BuildValue("(ii)", least, greatest);
  END_HANDLE_TH_ERRORS
}

// LITERALINCLUDE MCPU MODULE METHODS
static PyMethodDef methods[] = {
    {"_init", _initExtension, METH_NOARGS, nullptr},
    {"_get_default_generator", _getDefaultGenerator, METH_O, nullptr},
    {"_get_device", _getDevice, METH_NOARGS, nullptr},
    {"_set_device", _setDevice, METH_O, nullptr},
    {"_exchangeDevice", _exchangeDevice, METH_O, nullptr},
    {"_get_device_count", _getDeviceCount, METH_NOARGS, nullptr},
    {"_empty_cache", _emptyCache, METH_NOARGS, nullptr},
    {"_memory_stats", _memoryStats, METH_O, nullptr},
    {"_reset_peak_memory_stats", _resetPeakMemoryStats, METH_O, nullptr},
    {"_reset_accumulated_memory_stats", _resetAccumulatedMemoryStats, METH_O, nullptr},
    {"_get_stream_priority_range", _getStreamPriorityRange, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};
// LITERALINCLUDE MCPU MODULE METHODS
/*
 * When ASAN is enabled, PyTorch modifies the dlopen flag during import,
 * causing all global and weak symbols in _C.so and its dependent libraries
 * to be exposed to the global symbol scope, which in turn causes
 * subsequent symbols with the same name in other libraries to be intercepted.
 * Therefore, it cannot be named initModule here, otherwise initModule
 * in torch/csrc/Module.cpp will be called, resulting in failure.
 */
extern "C" MCPU_EXPORT PyObject* initMcpuModule(void) {
  static struct PyModuleDef mcpu_C_module = {
      PyModuleDef_HEAD_INIT, "torch_mcpu._C", nullptr, -1, methods};
  PyObject* mod = PyModule_Create(&mcpu_C_module);

  return mod;
}
