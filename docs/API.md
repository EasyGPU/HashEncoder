# API Reference

## HashGridConfig

Configuration struct for the hash grid encoding.

```cpp
struct HashGridConfig {
    int   Dimensions        = 3;     // Input dimensionality (2 or 3)
    int   NumLevels         = 16;    // Number of resolution levels
    int   FeaturesPerLevel  = 2;     // Feature channels per level
    int   BaseResolution    = 16;    // Coarsest grid resolution
    float PerLevelScale     = 1.5f;  // Resolution growth factor
    int   Log2HashmapSize   = 19;    // log2(hash table entries per level)
    bool  ClampInput        = true;  // Clamp positions to [0, 1]
};
```

## HashGridEncoding\<Dim\>

Template class parameterized on input dimensionality (2 or 3).

### Type Aliases

```cpp
using HashGridEncoding2D = HashGridEncoding<2>;
using HashGridEncoding3D = HashGridEncoding<3>;
```

### Constructor

```cpp
explicit HashGridEncoding(const HashGridConfig &config);
```

Throws `std::invalid_argument` if config is invalid (e.g., `Dimensions` not 2 or 3).

### Initialization

```cpp
void Initialize(uint64_t seed);
void InitializeZero();
```

`Initialize` fills the hash table with uniform random values in `[-1e-4, 1e-4]`.
`InitializeZero` sets all table entries to zero.

### Forward Pass

```cpp
// CPU: positions[count * Dim] → features[count * OutputDims]
void EncodeCPU(const float *positions, float *features, int count) const;

// GPU: positions/features passed as GPU::Runtime::Buffer<float>
void EncodeGPU(Buffer<float> &posBuffer, Buffer<float> &featBuffer, int count);
```

### Backward Pass

```cpp
// CPU: accumulates into tableGrads (caller must pre-zero)
void EncodeCPUBackward(const float *positions, const float *featureGrads,
                       float *tableGrads, float *posGrads, int count);

// GPU: manages table gradient buffer internally
void EncodeGPUBackward(Buffer<float> &posBuffer, Buffer<float> &gradInBuffer,
                       Buffer<float> *posGradBuffer, int count);
void ZeroTableGradients();
std::vector<float> GetTableGradients();
```

`posGrads` / `posGradBuffer` may be `nullptr` to skip position gradient computation.

### Accessors

```cpp
unsigned    GetOutputDims() const;         // NumLevels * FeaturesPerLevel
float       GetLevelResolution(int l) const;
int         GetTableSize() const;          // 2^Log2HashmapSize
size_t      GetHashTableSize() const;      // Total floats in table
const std::vector<float>& GetHashTable() const;
void        SetHashTable(const std::vector<float> &data);
void        MarkTableDirty();              // Force GPU re-upload
```

### Serialization

```cpp
bool Save(const std::string &path) const;
bool Load(const std::string &path);
```

Binary format: version tag → config fields → hash table data. See README for exact layout.

`TrainStep` runs forward → MSE loss → backward → parameter update, returning the loss value before the update.
