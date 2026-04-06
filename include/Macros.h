#pragma once

#ifdef _WIN32
#define MCPU_EXPORT __declspec(dllexport)
#else
#define MCPU_EXPORT __attribute__((visibility("default")))
#endif
