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

## MiniMLP

A small CPU multilayer perceptron for use with hash grid features.

### Config

```cpp
struct Config {
    int InputDim        = 16;
    int HiddenDim       = 64;
    int OutputDim       = 3;
    int NumHiddenLayers = 2;
};
```

### Construction

```cpp
MiniMLP();                          // Default, uninitialized
explicit MiniMLP(const Config &c);  // Initialize with Xavier-uniform weights
void Init(const Config &c);         // Re-initialize
```

### Forward / Backward

```cpp
void Forward(const float *input, float *output,
             std::vector<std::vector<float>> &activations) const;

void Backward(const float *lossGrad,
              const std::vector<std::vector<float>> &activations,
              float *inputGrad);
```

`Forward` saves per-layer activations (including pre-ReLU values) into `activations`.
`Backward` uses these activations to compute ReLU gradients and accumulates weight/bias gradients internally.

### Parameter Update

```cpp
void ZeroGradients();
void UpdateParameters(float learningRate);  // SGD step
```

## NeuralHashGrid\<Dim\>

End-to-end trainable model combining HashGridEncoding + MiniMLP.

```cpp
template <int Dim>
class NeuralHashGrid {
public:
    NeuralHashGrid(const HashGridConfig &encCfg, const MiniMLP::Config &mlpCfg);

    void Initialize(uint64_t seed);

    // Forward: positions → features → MLP → output
    void Forward(const float *positions, float *output, int count) const;

    // Backward: loss gradient → MLP grads + encoder grads
    void Backward(const float *positions, const float *lossGrad,
                  float *tableGrads, float *posGrads, int count);

    // Full training step with MSE loss
    float TrainStep(const float *positions, const float *targets,
                    int count, float learningRate);

    // Access underlying components
    HashGridEncoding<Dim>& GetEncoder();
    MiniMLP&               GetMLP();
};
```

`TrainStep` runs forward → MSE loss → backward → parameter update, returning the loss value before the update.
