# Multi-Library Integration Guide

## The Problem

EasyGPU is included as a git submodule. If you depend on multiple libraries that each vendor EasyGPU, CMake will error with "duplicate target" when both try to call `add_subdirectory(EasyGPU)`.

## The Solution

HashEncoder guards its `add_subdirectory` call:

```cmake
if(NOT TARGET EasyGPU)
    add_subdirectory(EasyGPU)
endif()
```

The first library to add EasyGPU wins. Subsequent libraries reuse the existing target.

## Requirements for Safe Coexistence

For this guard to work safely, all libraries in your project must:

1. **Use compatible EasyGPU versions**. If library A vendors EasyGPU at commit `abc` and library B vendors EasyGPU at commit `def` with breaking API changes, the build may fail or produce subtly broken binaries.

2. **Not rely on EasyGPU build options from subdirectory scope**. Cache variables like `EASYGPU_BUILD_EXAMPLES` will only take effect from the first `add_subdirectory` call.

## Recommended Project Layout

```
YourProject/
  CMakeLists.txt          # top-level: add_subdirectory(libs/*)
  libs/
    HashEncoder/          # submodule (vendors EasyGPU)
    YourOtherLib/         # submodule (may also vendor EasyGPU)
  src/
    main.cpp
```

Top-level `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.21)
project(YourProject)

# EasyGPU backend choice — set BEFORE adding subdirectories
set(EASYGPU_BACKEND "OpenGL" CACHE STRING "GPU backend")

add_subdirectory(libs/HashEncoder)
add_subdirectory(libs/YourOtherLib)

add_executable(my_app src/main.cpp)
target_link_libraries(my_app PRIVATE HashEncoder YourOtherLib)
```

## Verifying Compatibility

After configuring, check that only one EasyGPU target exists:

```bash
cmake -B build -S .
# Should see "EasyGPU" added only once in the output
```

## Version Pinning

To ensure all libraries use the same EasyGPU version, consider using a top-level EasyGPU submodule and pointing all libraries at it, or pinning submodule commits to known-compatible versions.
