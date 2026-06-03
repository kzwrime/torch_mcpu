cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(02_external_mcpu_kernel_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" "-c"
          "import pathlib, torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" "-c"
          "import importlib.util, pathlib; spec = importlib.util.find_spec('torch_mcpu'); print(pathlib.Path(spec.origin).parent)"
  OUTPUT_VARIABLE TORCH_MCPU_PACKAGE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

add_library(external_mcpu_axpby SHARED external_axpby.cpp)

target_include_directories(external_mcpu_axpby PRIVATE
  "${TORCH_MCPU_PACKAGE_DIR}/include")
target_link_directories(external_mcpu_axpby PRIVATE
  "${TORCH_MCPU_PACKAGE_DIR}/lib")
target_link_libraries(external_mcpu_axpby PRIVATE
  "${TORCH_LIBRARIES}"
  torch_mcpu)

set_target_properties(external_mcpu_axpby PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  INSTALL_RPATH "$ORIGIN;${TORCH_MCPU_PACKAGE_DIR}/lib"
  BUILD_WITH_INSTALL_RPATH TRUE)
