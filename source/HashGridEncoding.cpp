/**
 * @file HashGridEncoding.cpp
 * @brief GPU forward/backward via EasyGPU Kernel1D DSL, serialization, lifecycle.
 *
 * Uses EasyGPU's C++ DSL (Kernel1D, BufferSlot, Uniform, AtomicAdd, Callable)
 * instead of raw GLSL strings. The GPU state is PIMPL'd behind HashGridGPUState
 * so that kernel lambdas capture a stable raw pointer, surviving moves of the
 * owning HashGridEncoding object.
 */

#include <HashEncoder/HashGridConfig.h>
#include <HashEncoder/HashGridEncoding.h>
#include <HashEncoder/HashGridEncoding.inl>

#include <GPU.h>

#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>

namespace HashEncoder {

// ============================================================================
// GPU state helpers (non-template, to avoid template ABI issues)
// ============================================================================

namespace {
	std::unique_ptr<Buffer<float>> CreateTableBuffer(const std::vector<float> &data) {
		return std::make_unique<Buffer<float>>(data, BufferMode::Read);
	}

	std::unique_ptr<Buffer<float>> CreateEmptyBuffer(size_t count) {
		return std::make_unique<Buffer<float>>(count, BufferMode::ReadWrite);
	}
} // anonymous namespace

// ============================================================================
// GPU state (PIMPL)
// ============================================================================

struct HashGridGPUState {
	// Dynamic buffer slots — attached per-call
	BufferSlot<float> posSlot;
	BufferSlot<float> featSlot;
	BufferSlot<float> gradInSlot;
	BufferSlot<float> posGradSlot;

	// Owned GPU buffers
	std::unique_ptr<Buffer<float>> tableBuffer;
	std::unique_ptr<Buffer<float>> tableGradBuffer;

	// Uniforms (set before dispatch)
	Uniform<int> uNumLevels;
	Uniform<int> uFeaturesPerLevel;
	Uniform<int> uBaseResolution;
	Uniform<float> uPerLevelScale;
	Uniform<int> uLog2HashmapSize;
	Uniform<int> uTableSize;
	Uniform<int> uCount;
	Uniform<int> uComputePosGrad;

	// Callables (must outlive kernel objects — Callable::operator() captures [this])
	std::unique_ptr<Callable<Int(Int, Int)>> hash2D;
	std::unique_ptr<Callable<Int(Int, Int, Int)>> hash3D;

	// Cached config for re-upload detection
	std::vector<float> cachedTable;

	// Kernels (lazy-created)
	std::unique_ptr<Kernel1D> forwardKernel;
	std::unique_ptr<Kernel1D> backwardKernel;

	bool forwardCompiled  = false;
	bool backwardCompiled = false;
	bool tableDirty       = true;
};

// ============================================================================
// EnsureGPUProgram — lazy-create forward Kernel1D
// ============================================================================

template <int Dim>
void HashGridEncoding<Dim>::EnsureGPUProgram() {
	if (_gpu && _gpu->forwardCompiled) {
		return;
	}

	if (!_gpu) {
		_gpu = std::make_unique<HashGridGPUState>();
	}

	AutoInitContext();
	Context::GetInstance().MakeCurrent();

	// Upload table
	_gpu->tableBuffer = CreateTableBuffer(_hashTable);
	_gpu->cachedTable = _hashTable;
	_gpu->tableDirty  = false;

	// Stable raw pointer for kernel lambda capture (survives HashGridEncoding moves)
	HashGridGPUState *gpu = _gpu.get();

	// Hash callables stored in GPUState to outlive kernel construction
	// (Callable::operator() captures [this] — must stay alive until Dispatch)
	gpu->hash2D = std::make_unique<Callable<Int(Int, Int)>>([](Int &x, Int &y) {
		Int r = (x * MakeInt(73856093)) ^ (y * MakeInt(19349663));
		Return(r);
	});

	gpu->hash3D = std::make_unique<Callable<Int(Int, Int, Int)>>([](Int &x, Int &y, Int &z) {
		Int r = (x * MakeInt(73856093)) ^ (y * MakeInt(19349663)) ^ (z * MakeInt(83492791));
		Return(r);
	});

	if constexpr (Dim == 2) {
		gpu->forwardKernel = std::make_unique<Kernel1D>([gpu](Int i) {
			auto pos   = gpu->posSlot.Bind();
			auto table = gpu->tableBuffer->Bind();
			auto out   = gpu->featSlot.Bind();

			auto numLevels    = gpu->uNumLevels.Load();
			auto featPerLevel = gpu->uFeaturesPerLevel.Load();
			auto baseRes      = gpu->uBaseResolution.Load();
			auto scale        = gpu->uPerLevelScale.Load();
			auto tableSize    = gpu->uTableSize.Load();
			auto count        = gpu->uCount.Load();

			If(i >= count, [&]() { Return(); });

			Float px = Clamp(pos[i * 2 + 0], 0.0f, 1.0f);
			Float py = Clamp(pos[i * 2 + 1], 0.0f, 1.0f);

			Int outOffset  = i * numLevels * featPerLevel;
			Int tableMask = tableSize - MakeInt(1);

			For(MakeInt(0), numLevels, [&](Int &level) {
				Float res = ToFloat(baseRes) * Pow(scale, ToFloat(level));

				Float fx = px * res - 0.5f;
				Float fy = py * res - 0.5f;

				Int ix = ToInt(Floor(fx));
				Int iy = ToInt(Floor(fy));

				Float wx = fx - ToFloat(ix);
				Float wy = fy - ToFloat(iy);

				Int tableOffset = level * tableSize * featPerLevel;
				Int featOffset  = outOffset + level * featPerLevel;

				// Zero output for this level
				For(MakeInt(0), featPerLevel, [&](Int &c) {
					out[featOffset + c] = MakeFloat(0.0f);
				});

				// Bilinear interpolation over 4 corners
				For(MakeInt(0), MakeInt(2), [&](Int &dy) {
					For(MakeInt(0), MakeInt(2), [&](Int &dx) {
						Int cx = ix + dx;
						Int cy = iy + dy;

						Int h     = (*gpu->hash2D)(cx, cy);
						Int entry = h & tableMask;

						Float wxCoef = Select(dx != MakeInt(0), wx, MakeFloat(1.0f) - wx);
						Float wyCoef = Select(dy != MakeInt(0), wy, MakeFloat(1.0f) - wy);
						Float w      = wxCoef * wyCoef;

						Int tableAddr = tableOffset + entry * featPerLevel;

						For(MakeInt(0), featPerLevel, [&](Int &c) {
							out[featOffset + c] = out[featOffset + c] + w * table[tableAddr + c];
						});
					});
				});
			});
		});
	} else if constexpr (Dim == 3) {
		gpu->forwardKernel = std::make_unique<Kernel1D>([gpu](Int i) {
			auto pos   = gpu->posSlot.Bind();
			auto table = gpu->tableBuffer->Bind();
			auto out   = gpu->featSlot.Bind();

			auto numLevels    = gpu->uNumLevels.Load();
			auto featPerLevel = gpu->uFeaturesPerLevel.Load();
			auto baseRes      = gpu->uBaseResolution.Load();
			auto scale        = gpu->uPerLevelScale.Load();
			auto tableSize    = gpu->uTableSize.Load();
			auto count        = gpu->uCount.Load();

			If(i >= count, [&]() { Return(); });

			Float px = Clamp(pos[i * 3 + 0], 0.0f, 1.0f);
			Float py = Clamp(pos[i * 3 + 1], 0.0f, 1.0f);
			Float pz = Clamp(pos[i * 3 + 2], 0.0f, 1.0f);

			Int outOffset = i * numLevels * featPerLevel;
			Int tableMask = tableSize - MakeInt(1);

			For(MakeInt(0), numLevels, [&](Int &level) {
				Float res = ToFloat(baseRes) * Pow(scale, ToFloat(level));

				Float fx = px * res - 0.5f;
				Float fy = py * res - 0.5f;
				Float fz = pz * res - 0.5f;

				Int ix = ToInt(Floor(fx));
				Int iy = ToInt(Floor(fy));
				Int iz = ToInt(Floor(fz));

				Float wx = fx - ToFloat(ix);
				Float wy = fy - ToFloat(iy);
				Float wz = fz - ToFloat(iz);

				Int tableOffset = level * tableSize * featPerLevel;
				Int featOffset  = outOffset + level * featPerLevel;

				For(MakeInt(0), featPerLevel, [&](Int &c) {
					out[featOffset + c] = MakeFloat(0.0f);
				});

				// Trilinear interpolation over 8 corners
				For(MakeInt(0), MakeInt(2), [&](Int &dz) {
					For(MakeInt(0), MakeInt(2), [&](Int &dy) {
						For(MakeInt(0), MakeInt(2), [&](Int &dx) {
							Int cx = ix + dx;
							Int cy = iy + dy;
							Int cz = iz + dz;

							Int h     = (*gpu->hash3D)(cx, cy, cz);
							Int entry = h & tableMask;

							Float wxCoef = Select(dx != MakeInt(0), wx, MakeFloat(1.0f) - wx);
							Float wyCoef = Select(dy != MakeInt(0), wy, MakeFloat(1.0f) - wy);
							Float wzCoef = Select(dz != MakeInt(0), wz, MakeFloat(1.0f) - wz);
							Float w      = wxCoef * wyCoef * wzCoef;

							Int tableAddr = tableOffset + entry * featPerLevel;

							For(MakeInt(0), featPerLevel, [&](Int &c) {
								out[featOffset + c] = out[featOffset + c] + w * table[tableAddr + c];
							});
						});
					});
				});
			});
		});
	}

	gpu->forwardCompiled = true;
}

// ============================================================================
// EncodeGPU — attach slots, set uniforms, dispatch
// ============================================================================

template <int Dim>
void HashGridEncoding<Dim>::EncodeGPU(const Buffer<float> &PositionBuffer, Buffer<float> &FeatureBuffer, size_t Count) {
	if (Count == 0) {
		return;
	}

	EnsureGPUProgram();

	if (_gpu->tableDirty) {
		_gpu->tableBuffer->Upload(_hashTable);
		_gpu->cachedTable = _hashTable;
		_gpu->tableDirty  = false;
	}

	_gpu->uNumLevels       = _config.NumLevels;
	_gpu->uFeaturesPerLevel = _config.FeaturesPerLevel;
	_gpu->uBaseResolution  = _config.BaseResolution;
	_gpu->uPerLevelScale   = _config.PerLevelScale;
	_gpu->uLog2HashmapSize = _config.Log2HashmapSize;
	_gpu->uTableSize       = _tableSize;
	_gpu->uCount           = static_cast<int>(Count);

	_gpu->posSlot.Attach(const_cast<Buffer<float> &>(PositionBuffer));
	_gpu->featSlot.Attach(FeatureBuffer);

	uint32_t groups = (static_cast<uint32_t>(Count) + 255) / 256;
	_gpu->forwardKernel->Dispatch(groups, true);
}

// ============================================================================
// EnsureGPUBackwardProgram — lazy-create backward Kernel1D
// ============================================================================

template <int Dim>
void HashGridEncoding<Dim>::EnsureGPUBackwardProgram() {
	if (_gpu && _gpu->backwardCompiled) {
		return;
	}

	// Ensure forward is ready (table buffer, context, callables)
	EnsureGPUProgram();

	HashGridGPUState *gpu = _gpu.get();

	// Allocate gradient buffer BEFORE kernel construction — kernel binds it during IR build
	gpu->tableGradBuffer = CreateEmptyBuffer(_hashTable.size());
	ZeroTableGradients();

	if constexpr (Dim == 2) {
		gpu->backwardKernel = std::make_unique<Kernel1D>([gpu](Int i) {
			auto pos       = gpu->posSlot.Bind();
			auto table     = gpu->tableBuffer->Bind();
			auto gradIn    = gpu->gradInSlot.Bind();
			auto tableGrad = gpu->tableGradBuffer->Bind();

			auto numLevels      = gpu->uNumLevels.Load();
			auto featPerLevel   = gpu->uFeaturesPerLevel.Load();
			auto baseRes        = gpu->uBaseResolution.Load();
			auto scale          = gpu->uPerLevelScale.Load();
			auto tableSize      = gpu->uTableSize.Load();
			auto count          = gpu->uCount.Load();
			auto computePosGrad = gpu->uComputePosGrad.Load();

			If(i >= count, [&]() { Return(); });

			Float px = Clamp(pos[i * 2 + 0], 0.0f, 1.0f);
			Float py = Clamp(pos[i * 2 + 1], 0.0f, 1.0f);

			Int gradInOffset = i * numLevels * featPerLevel;
			Int tableMask    = tableSize - MakeInt(1);

			Float posGradX = MakeFloat(0.0f);
			Float posGradY = MakeFloat(0.0f);

			For(MakeInt(0), numLevels, [&](Int &level) {
				Float res = ToFloat(baseRes) * Pow(scale, ToFloat(level));

				Float fx = px * res - 0.5f;
				Float fy = py * res - 0.5f;

				Int ix = ToInt(Floor(fx));
				Int iy = ToInt(Floor(fy));

				Float wx = fx - ToFloat(ix);
				Float wy = fy - ToFloat(iy);

				Int tableOffset = level * tableSize * featPerLevel;
				Int featOffset  = gradInOffset + level * featPerLevel;

				For(MakeInt(0), MakeInt(2), [&](Int &dy) {
					For(MakeInt(0), MakeInt(2), [&](Int &dx) {
						Int cx = ix + dx;
						Int cy = iy + dy;

						Int h     = (*gpu->hash2D)(cx, cy);
						Int entry = h & tableMask;

						Float wxCoef = Select(dx != MakeInt(0), wx, MakeFloat(1.0f) - wx);
						Float wyCoef = Select(dy != MakeInt(0), wy, MakeFloat(1.0f) - wy);
						Float w      = wxCoef * wyCoef;

						Int tableAddr = tableOffset + entry * featPerLevel;

						For(MakeInt(0), featPerLevel, [&](Int &c) {
							Float gradVal = w * gradIn[featOffset + c];
							ExprBase::NotUse(AtomicAdd(tableGrad[tableAddr + c], gradVal));

							If(computePosGrad != MakeInt(0), [&]() {
								Float tableVal = table[tableAddr + c];
								Float dWx      = Select(dx != MakeInt(0), res, -res);
								Float dWy      = Select(dy != MakeInt(0), res, -res);
								Float wxTerm   = gradIn[featOffset + c] * tableVal;
								posGradX = posGradX + wxTerm * dWx * wyCoef;
								posGradY = posGradY + wxTerm * dWy * wxCoef;
							});
						});
					});
				});
			});

			If(computePosGrad != MakeInt(0), [&]() {
				auto posGrad      = gpu->posGradSlot.Bind();
				Int poff          = i * 2;
				posGrad[poff + 0] = posGradX;
				posGrad[poff + 1] = posGradY;
			});
		});
	} else if constexpr (Dim == 3) {
		gpu->backwardKernel = std::make_unique<Kernel1D>([gpu](Int i) {
			auto pos       = gpu->posSlot.Bind();
			auto table     = gpu->tableBuffer->Bind();
			auto gradIn    = gpu->gradInSlot.Bind();
			auto tableGrad = gpu->tableGradBuffer->Bind();

			auto numLevels      = gpu->uNumLevels.Load();
			auto featPerLevel   = gpu->uFeaturesPerLevel.Load();
			auto baseRes        = gpu->uBaseResolution.Load();
			auto scale          = gpu->uPerLevelScale.Load();
			auto tableSize      = gpu->uTableSize.Load();
			auto count          = gpu->uCount.Load();
			auto computePosGrad = gpu->uComputePosGrad.Load();

			If(i >= count, [&]() { Return(); });

			Float px = Clamp(pos[i * 3 + 0], 0.0f, 1.0f);
			Float py = Clamp(pos[i * 3 + 1], 0.0f, 1.0f);
			Float pz = Clamp(pos[i * 3 + 2], 0.0f, 1.0f);

			Int gradInOffset = i * numLevels * featPerLevel;
			Int tableMask    = tableSize - MakeInt(1);

			Float posGradX = MakeFloat(0.0f);
			Float posGradY = MakeFloat(0.0f);
			Float posGradZ = MakeFloat(0.0f);

			For(MakeInt(0), numLevels, [&](Int &level) {
				Float res = ToFloat(baseRes) * Pow(scale, ToFloat(level));

				Float fx = px * res - 0.5f;
				Float fy = py * res - 0.5f;
				Float fz = pz * res - 0.5f;

				Int ix = ToInt(Floor(fx));
				Int iy = ToInt(Floor(fy));
				Int iz = ToInt(Floor(fz));

				Float wx = fx - ToFloat(ix);
				Float wy = fy - ToFloat(iy);
				Float wz = fz - ToFloat(iz);

				Int tableOffset = level * tableSize * featPerLevel;
				Int featOffset  = gradInOffset + level * featPerLevel;

				For(MakeInt(0), MakeInt(2), [&](Int &dz) {
					For(MakeInt(0), MakeInt(2), [&](Int &dy) {
						For(MakeInt(0), MakeInt(2), [&](Int &dx) {
							Int cx = ix + dx;
							Int cy = iy + dy;
							Int cz = iz + dz;

							Int h     = (*gpu->hash3D)(cx, cy, cz);
							Int entry = h & tableMask;

							Float wxCoef = Select(dx != MakeInt(0), wx, MakeFloat(1.0f) - wx);
							Float wyCoef = Select(dy != MakeInt(0), wy, MakeFloat(1.0f) - wy);
							Float wzCoef = Select(dz != MakeInt(0), wz, MakeFloat(1.0f) - wz);
							Float w      = wxCoef * wyCoef * wzCoef;

							Int tableAddr = tableOffset + entry * featPerLevel;

							For(MakeInt(0), featPerLevel, [&](Int &c) {
								Float gradVal = w * gradIn[featOffset + c];
								ExprBase::NotUse(AtomicAdd(tableGrad[tableAddr + c], gradVal));

								If(computePosGrad != MakeInt(0), [&]() {
									Float tableVal = table[tableAddr + c];
									Float dWx      = Select(dx != MakeInt(0), res, -res);
									Float dWy      = Select(dy != MakeInt(0), res, -res);
									Float dWz      = Select(dz != MakeInt(0), res, -res);
									Float wxTerm   = gradIn[featOffset + c] * tableVal;
									posGradX = posGradX + wxTerm * dWx * wyCoef * wzCoef;
									posGradY = posGradY + wxTerm * dWy * wxCoef * wzCoef;
									posGradZ = posGradZ + wxTerm * dWz * wxCoef * wyCoef;
								});
							});
						});
					});
				});
			});

			If(computePosGrad != MakeInt(0), [&]() {
				auto posGrad      = gpu->posGradSlot.Bind();
				Int poff          = i * 3;
				posGrad[poff + 0] = posGradX;
				posGrad[poff + 1] = posGradY;
				posGrad[poff + 2] = posGradZ;
			});
		});
	}

	gpu->backwardCompiled = true;
}

// ============================================================================
// EncodeGPUBackward
// ============================================================================

template <int Dim>
void HashGridEncoding<Dim>::EncodeGPUBackward(const Buffer<float> &PositionBuffer,
											   const Buffer<float> &FeatureGradBuffer,
											   Buffer<float> *PositionGradBuffer, size_t Count) {
	if (Count == 0) {
		return;
	}

	EnsureGPUBackwardProgram();

	if (_gpu->tableDirty) {
		_gpu->tableBuffer->Upload(_hashTable);
		_gpu->cachedTable = _hashTable;
		_gpu->tableDirty  = false;
	}

	_gpu->uNumLevels       = _config.NumLevels;
	_gpu->uFeaturesPerLevel = _config.FeaturesPerLevel;
	_gpu->uBaseResolution  = _config.BaseResolution;
	_gpu->uPerLevelScale   = _config.PerLevelScale;
	_gpu->uLog2HashmapSize = _config.Log2HashmapSize;
	_gpu->uTableSize       = _tableSize;
	_gpu->uCount           = static_cast<int>(Count);
	_gpu->uComputePosGrad  = (PositionGradBuffer != nullptr) ? 1 : 0;

	_gpu->posSlot.Attach(const_cast<Buffer<float> &>(PositionBuffer));
	_gpu->gradInSlot.Attach(const_cast<Buffer<float> &>(FeatureGradBuffer));

	if (PositionGradBuffer) {
		_gpu->posGradSlot.Attach(*PositionGradBuffer);
	}

	uint32_t groups = (static_cast<uint32_t>(Count) + 255) / 256;
	_gpu->backwardKernel->Dispatch(groups, true);
}

// ============================================================================
// GPU Gradients
// ============================================================================

template <int Dim>
void HashGridEncoding<Dim>::ZeroTableGradients() {
	if (!_gpu || !_gpu->tableGradBuffer) {
		return;
	}
	std::vector<float> zeros(_hashTable.size(), 0.0f);
	_gpu->tableGradBuffer->Upload(zeros);
}

template <int Dim>
std::vector<float> HashGridEncoding<Dim>::GetTableGradients() const {
	if (!_gpu || !_gpu->tableGradBuffer) {
		throw std::runtime_error("HashGridEncoding::GetTableGradients: backward pass not yet compiled");
	}
	std::vector<float> grads(_hashTable.size());
	_gpu->tableGradBuffer->Download(grads.data(), grads.size());
	return grads;
}

// ============================================================================
// Serialization
// ============================================================================

static constexpr uint32_t kHashGridFileVersion = 1;

template <int Dim>
void HashGridEncoding<Dim>::Save(const std::string &FilePath) const {
	std::ofstream file(FilePath, std::ios::binary);
	if (!file) {
		throw std::runtime_error(std::format("HashGridEncoding::Save: cannot open file '{}'", FilePath));
	}

	uint32_t version = kHashGridFileVersion;
	file.write(reinterpret_cast<const char *>(&version), sizeof(version));

	file.write(reinterpret_cast<const char *>(&_config.Dimensions), sizeof(_config.Dimensions));
	file.write(reinterpret_cast<const char *>(&_config.NumLevels), sizeof(_config.NumLevels));
	file.write(reinterpret_cast<const char *>(&_config.FeaturesPerLevel), sizeof(_config.FeaturesPerLevel));
	file.write(reinterpret_cast<const char *>(&_config.BaseResolution), sizeof(_config.BaseResolution));
	file.write(reinterpret_cast<const char *>(&_config.PerLevelScale), sizeof(_config.PerLevelScale));
	file.write(reinterpret_cast<const char *>(&_config.Log2HashmapSize), sizeof(_config.Log2HashmapSize));
	file.write(reinterpret_cast<const char *>(&_config.ClampInput), sizeof(_config.ClampInput));

	uint64_t tableCount = static_cast<uint64_t>(_hashTable.size());
	file.write(reinterpret_cast<const char *>(&tableCount), sizeof(tableCount));
	file.write(reinterpret_cast<const char *>(_hashTable.data()), _hashTable.size() * sizeof(float));

	if (!file) {
		throw std::runtime_error(std::format("HashGridEncoding::Save: write failed for '{}'", FilePath));
	}
}

template <int Dim>
void HashGridEncoding<Dim>::Load(const std::string &FilePath) {
	std::ifstream file(FilePath, std::ios::binary);
	if (!file) {
		throw std::runtime_error(std::format("HashGridEncoding::Load: cannot open file '{}'", FilePath));
	}

	uint32_t version = 0;
	file.read(reinterpret_cast<char *>(&version), sizeof(version));
	if (version != kHashGridFileVersion) {
		throw std::runtime_error(
			std::format("HashGridEncoding::Load: unsupported file version {} (expected {})", version, kHashGridFileVersion));
	}

	HashGridConfig loadedConfig;
	file.read(reinterpret_cast<char *>(&loadedConfig.Dimensions), sizeof(loadedConfig.Dimensions));
	file.read(reinterpret_cast<char *>(&loadedConfig.NumLevels), sizeof(loadedConfig.NumLevels));
	file.read(reinterpret_cast<char *>(&loadedConfig.FeaturesPerLevel), sizeof(loadedConfig.FeaturesPerLevel));
	file.read(reinterpret_cast<char *>(&loadedConfig.BaseResolution), sizeof(loadedConfig.BaseResolution));
	file.read(reinterpret_cast<char *>(&loadedConfig.PerLevelScale), sizeof(loadedConfig.PerLevelScale));
	file.read(reinterpret_cast<char *>(&loadedConfig.Log2HashmapSize), sizeof(loadedConfig.Log2HashmapSize));
	file.read(reinterpret_cast<char *>(&loadedConfig.ClampInput), sizeof(loadedConfig.ClampInput));

	if (!file) {
		throw std::runtime_error("HashGridEncoding::Load: failed to read config from file");
	}

	uint64_t tableCount = 0;
	file.read(reinterpret_cast<char *>(&tableCount), sizeof(tableCount));

	std::vector<float> loadedTable(tableCount);
	file.read(reinterpret_cast<char *>(loadedTable.data()), tableCount * sizeof(float));

	if (!file) {
		throw std::runtime_error("HashGridEncoding::Load: failed to read hash table from file");
	}

	if (loadedConfig.Dimensions != Dim || loadedConfig.NumLevels != _config.NumLevels ||
		loadedConfig.FeaturesPerLevel != _config.FeaturesPerLevel ||
		loadedConfig.Log2HashmapSize != _config.Log2HashmapSize) {
		throw std::runtime_error("HashGridEncoding::Load: config mismatch - file config differs from current config");
	}

	_config    = loadedConfig;
	_hashTable = std::move(loadedTable);
	MarkTableDirty();
}

// ============================================================================
// Lifecycle
// ============================================================================

template <int Dim>
HashGridEncoding<Dim>::HashGridEncoding(HashGridEncoding &&Other) noexcept
	: _config(Other._config), _hashTable(std::move(Other._hashTable)), _numLevels(Other._numLevels),
	  _featuresPerLevel(Other._featuresPerLevel), _tableSize(Other._tableSize), _outputDims(Other._outputDims),
	  _gpu(std::move(Other._gpu)) {}

template <int Dim>
HashGridEncoding<Dim> &HashGridEncoding<Dim>::operator=(HashGridEncoding &&Other) noexcept {
	if (this != &Other) {
		_config           = Other._config;
		_hashTable        = std::move(Other._hashTable);
		_numLevels        = Other._numLevels;
		_featuresPerLevel = Other._featuresPerLevel;
		_tableSize        = Other._tableSize;
		_outputDims       = Other._outputDims;
		_gpu              = std::move(Other._gpu);
	}
	return *this;
}

template <int Dim>
HashGridEncoding<Dim>::~HashGridEncoding() = default;

// ============================================================================
// Explicit template instantiation
// ============================================================================

template class HashGridEncoding<2>;
template class HashGridEncoding<3>;

} // namespace HashEncoder
