# HashEncoder

Multi-resolution hash grid encoding library (Instant-NGP style) built on [EasyGPU](https://github.com/EasyGPU/EasyGPU).

## Overview

HashEncoder implements the multi-resolution hash grid encoding described in Instant-NGP (Muller et al., 2022). It maps 2D or 3D spatial coordinates to trainable feature vectors by:

1. Computing L levels of progressively finer grids
2. Hashing integer corner coordinates to a shared trainable hash table
3. Bilinear (2D) or trilinear (3D) interpolation of corner features
4. Concatenating features across all levels

This encoding is the foundational building block for neural radiance fields (NeRF), neural SDFs, radiance caches, visibility caches, path guiding, and general neural rendering applications.

## Dependencies

- **EasyGPU** (included as a git submodule) — GPU compute infrastructure
- **C++20** compiler (MSVC 2022+, GCC 12+, Clang 16+)
- **OpenGL 4.3+** for GPU encode (or Vulkan via EasyGPU backend)

## Building

```bash
git clone --recursive https://github.com/your-org/HashEncoder.git
cd HashEncoder
mkdir build && cd build
cmake .. -DHASHENCODER_BUILD_EXAMPLES=ON -DHASHENCODER_BUILD_TESTS=ON
cmake --build .
```

CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `HASHENCODER_BUILD_EXAMPLES` | ON | Build example programs |
| `HASHENCODER_BUILD_TESTS` | ON | Build test suite |

## Quick Start

```cpp
#include <HashEncoder/HashEncoder.h>

// 1. Configure the encoder
HashEncoder::HashGridConfig config;
config.NumLevels        = 16;
config.FeaturesPerLevel = 2;
config.BaseResolution   = 16;
config.PerLevelScale    = 1.5f;
config.Log2HashmapSize  = 19;

// 2. Create a 3D encoder
HashEncoder::HashGridEncoding3D encoder(config);
encoder.Initialize(42);  // seed for reproducibility

// 3. Encode positions on CPU
std::vector<float> positions = {0.5f, 0.5f, 0.5f};
std::vector<float> features(encoder.GetOutputDims());
encoder.EncodeCPU(positions.data(), features.data(), 1);

// 4. Or encode on GPU (requires initialized EasyGPU context)
#include <Runtime/Buffer.h>
GPU::Runtime::Buffer<float> posBuffer(positions, GPU::Runtime::BufferMode::Read);
GPU::Runtime::Buffer<float> featBuffer(encoder.GetOutputDims());
encoder.EncodeGPU(posBuffer, featBuffer, 1);
```

## Input / Output

**Input**: Positions in `[0, 1]^D`.

- 2D: `float positions[N * 2]` packed as `x0,y0, x1,y1, ...`
- 3D: `float positions[N * 3]` packed as `x0,y0,z0, x1,y1,z1, ...`

Inputs outside `[0, 1]` are clamped by default (controlled by `ClampInput`).

**Output**: Feature vectors of length `outputDims = NumLevels * FeaturesPerLevel`.

- `float features[N * outputDims]` with level 0 features first, then level 1, etc.

## Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `Dimensions` | int | 3 | Input dimensionality (2 or 3) |
| `NumLevels` | int | 16 | Number of resolution levels |
| `FeaturesPerLevel` | int | 2 | Feature channels per level |
| `BaseResolution` | int | 16 | Coarsest grid resolution |
| `PerLevelScale` | float | 1.5 | Resolution growth factor per level |
| `Log2HashmapSize` | int | 19 | log2 of hash table entries per level |
| `ClampInput` | bool | true | Clamp positions to [0, 1] |

Level `l` resolution: `BaseResolution * PerLevelScale^l`

Hash table size per level: `2^Log2HashmapSize` entries

Total parameters: `NumLevels * 2^Log2HashmapSize * FeaturesPerLevel` floats

## Hash Function

Uses stable integer-coordinate hashing (consistent CPU / GPU):

- 2D: `hash = uint(x) * 73856093u ^ uint(y) * 19349663u`
- 3D: `hash = uint(x) * 73856093u ^ uint(y) * 19349663u ^ uint(z) * 83492791u`

Hash table index: `hash & (tableSize - 1)` (requires power-of-two table size)

## API Reference

### HashGridEncoding<Dim>

| Method | Description |
|--------|-------------|
| `HashGridEncoding(config)` | Construct with config |
| `Initialize(seed)` | Random initialize hash table (uniform [-1e-4, 1e-4]) |
| `InitializeZero()` | Zero initialize hash table |
| **Forward** | |
| `EncodeCPU(positions, features, count)` | CPU reference encode |
| `EncodeGPU(posBuffer, featBuffer, count)` | GPU batch encode |
| **Backward** | |
| `EncodeCPUBackward(positions, featureGrads, tableGrads, posGrads, count)` | CPU backward (accumulates into tableGrads in-place) |
| `EncodeGPUBackward(posBuffer, gradInBuffer, posGradBuffer, count)` | GPU backward with atomic scatter-add |
| `ZeroTableGradients()` | Zero GPU table gradient buffer before accumulation |
| `GetTableGradients()` | Download accumulated GPU table gradients |
| **Accessors** | |
| `GetOutputDims()` | Returns `NumLevels * FeaturesPerLevel` |
| `GetLevelResolution(level)` | Resolution for a given level |
| `GetTableSize()` | Hash table entries per level |
| `GetHashTableSize()` | Total hash table floats |
| `GetHashTable()` | Const ref to hash table data |
| `SetHashTable(data)` | Set hash table from external data |
| `Save(path)` | Save config + table to binary file |
| `Load(path)` | Load config + table from binary file |

### Backward Pass Usage

```cpp
// CPU backward: pre-zero table gradients, then accumulate
std::vector<float> tableGrads(encoder.GetHashTableSize(), 0.0f);
std::vector<float> posGrads(count * 3, 0.0f);
encoder.EncodeCPUBackward(positions, featureGrads, tableGrads.data(), posGrads.data(), count);
// tableGrads now holds dL/dTable, posGrads holds dL/dPositions

// GPU backward: zero internal buffer, accumulate, download
encoder.ZeroTableGradients();
encoder.EncodeGPUBackward(posBuffer, gradInBuffer, &posGradBuffer, count);
std::vector<float> gpuTableGrads = encoder.GetTableGradients();
```

### Type Aliases

```cpp
using HashGridEncoding2D = HashGridEncoding<2>;  // Bilinear interpolation
using HashGridEncoding3D = HashGridEncoding<3>;  // Trilinear interpolation
```

## Limitations

1. **Float32 only**: No half-precision (float16) support yet.
2. **OpenGL primarily tested**: Vulkan backend should work via EasyGPU but needs additional testing.
3. **No MLP integration**: The encoding module is standalone; it does not include a fully-connected network decoder.
4. **Power-of-two hash table**: The hash table size must be a power of two (enforced by `Log2HashmapSize`).

## File Format

Binary file format (little-endian, native):

```
[uint32] version (= 1)
[int]    dimensions
[int]    numLevels
[int]    featuresPerLevel
[int]    baseResolution
[float]  perLevelScale
[int]    log2HashmapSize
[bool]   clampInput
[uint64] hashTableCount
[float]  hashTable[hashTableCount]
```

## Planned Features

- **Half precision (fp16)**: Reduced memory footprint for large hash tables
- **MLP integration helpers**: Convenience wrappers for encoding + FC network pipelines
- **Multi-GPU support**: Encoding distribution across multiple GPUs
- **Performance optimizations**: Occupancy tuning, shared memory caching, L1 cache hints
- **More encoding types**: Spherical harmonics encoding, frequency encoding, dense grid encoding

## License

This project is licensed under the same terms as EasyGPU. See [LICENSE](EasyGPU/LICENSE) for details.

## References

- Muller, T., Evans, A., Schied, C., & Keller, A. (2022). Instant Neural Graphics Primitives with a Multiresolution Hash Encoding. *ACM Transactions on Graphics (SIGGRAPH)*.
