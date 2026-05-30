# Building HashEncoder

## Prerequisites

- **C++20** compiler: MSVC 2022+, GCC 12+, or Clang 16+
- **CMake** 3.21+
- **GPU driver** with OpenGL 4.3+ support (or Vulkan 1.3+ for Vulkan backend)

## Quick Build

```bash
git clone --recursive https://github.com/EasyGPU/HashEncoder.git
cd HashEncoder
cmake -B build -S .
cmake --build build --config Release
```

## Backend Selection

HashEncoder delegates GPU work to EasyGPU, which supports two backends.

### OpenGL (default)

```bash
cmake -B build -S . -DEASYGPU_BACKEND=OpenGL
cmake --build build --config Release
```

No additional SDK required beyond your GPU driver.

### Vulkan

Requires the [Vulkan SDK](https://vulkan.lunarg.com/) (tested with 1.3.280.0).

```bash
cmake -B build_vk -S . -DEASYGPU_BACKEND=Vulkan
cmake --build build_vk --config Release
```

If the Vulkan SDK is not found, set `VULKAN_SDK` manually:

```bash
cmake -B build_vk -S . -DEASYGPU_BACKEND=Vulkan -DVULKAN_SDK="C:/VulkanSDK/1.3.280.0"
```

Both backends produce identical numerical results and pass the same test suite.

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `HASHENCODER_BUILD_EXAMPLES` | `ON` | Build example programs |
| `HASHENCODER_BUILD_TESTS` | `ON` | Build and register tests with CTest |
| `EASYGPU_BACKEND` | `OpenGL` | EasyGPU backend: `OpenGL` or `Vulkan` |

Example with all options:

```bash
cmake -B build -S . \
    -DHASHENCODER_BUILD_EXAMPLES=ON \
    -DHASHENCODER_BUILD_TESTS=ON \
    -DEASYGPU_BACKEND=OpenGL
```

## Running Tests

```bash
# Build and run all tests
cmake --build build --config Release
cd build && ctest --config Release

# Or run individual test executables
./build/Release/TestHashGridEncoding.exe   # CPU core tests
./build/Release/TestHashGridStress.exe     # Stress / edge-case tests
./build/Release/TestMinimalGPU.exe         # GPU correctness tests
```

## Multi-Library Projects

If your project uses multiple EasyGPU-based libraries, see [Integration Guide](INTEGRATION.md) for details on the `if(NOT TARGET EasyGPU)` guard and version compatibility.

## Troubleshooting

### "EasyGPU target not found"

Ensure submodules are initialized:
```bash
git submodule update --init --recursive
```

### GPU context initialization fails

- Ensure your GPU driver supports OpenGL 4.3+
- On headless servers, you may need a software renderer or Vulkan with `VK_EXT_headless_surface`
- GPU tests automatically skip (exit 0) if no GPU context is available

### MSVC compiler warnings C4819

These are suppressed by default with `/wd4819`. The project uses `/utf-8` to handle source encoding.
