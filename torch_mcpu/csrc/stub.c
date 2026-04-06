#include <Python.h>

#ifdef _WIN32
#define MCPU_EXPORT __declspec(dllexport)
#else
#define MCPU_EXPORT __attribute__((visibility("default")))
#endif

extern MCPU_EXPORT PyObject* initMcpuModule(void);

#ifdef __cplusplus
extern "C"
#endif

    MCPU_EXPORT PyObject*
    PyInit__C(void);

PyMODINIT_FUNC PyInit__C(void) {
  return initMcpuModule();
}
