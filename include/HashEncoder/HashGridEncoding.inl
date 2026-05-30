#pragma once

/**
 * @file HashGridEncoding.inl
 * @brief Template implementation for HashGridEncoding<Dim>.
 */

#ifndef EASYGPU_HASHGRIDENCODING_INL
#define EASYGPU_HASHGRIDENCODING_INL

#include <HashEncoder/HashGridEncoding.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace HashEncoder {

// ============================================================================
// Construction
// ============================================================================

template <int Dim>
HashGridEncoding<Dim>::HashGridEncoding(const HashGridConfig &Config)
	: _config(Config), _numLevels(Config.NumLevels), _featuresPerLevel(Config.FeaturesPerLevel),
	  _tableSize(1 << Config.Log2HashmapSize), _outputDims(Config.NumLevels * Config.FeaturesPerLevel) {
	if (Config.NumLevels <= 0) {
		throw std::invalid_argument("HashGridConfig::NumLevels must be positive");
	}
	if (Config.FeaturesPerLevel <= 0) {
		throw std::invalid_argument("HashGridConfig::FeaturesPerLevel must be positive");
	}
	if (Config.BaseResolution <= 0) {
		throw std::invalid_argument("HashGridConfig::BaseResolution must be positive");
	}
	if (Config.Log2HashmapSize <= 0 || Config.Log2HashmapSize > 30) {
		throw std::invalid_argument("HashGridConfig::Log2HashmapSize must be in [1, 30]");
	}
	if (Config.Dimensions != Dim) {
		throw std::invalid_argument("HashGridConfig::Dimensions does not match template parameter Dim");
	}
	if (Config.PerLevelScale <= 0.0f) {
		throw std::invalid_argument("HashGridConfig::PerLevelScale must be positive");
	}

	size_t tableSize = static_cast<size_t>(_numLevels) * _tableSize * _featuresPerLevel;
	_hashTable.assign(tableSize, 0.0f);
}

// ============================================================================
// Initialization
// ============================================================================

template <int Dim>
void HashGridEncoding<Dim>::Initialize(uint32_t Seed) {
	std::mt19937 rng(Seed);
	std::uniform_real_distribution<float> dist(-1e-4f, 1e-4f);

	for (float &v : _hashTable) {
		v = dist(rng);
	}
	MarkTableDirty();
}

template <int Dim>
void HashGridEncoding<Dim>::InitializeZero() {
	std::fill(_hashTable.begin(), _hashTable.end(), 0.0f);
	MarkTableDirty();
}

// ============================================================================
// CPU Encode
// ============================================================================

template <int Dim>
void HashGridEncoding<Dim>::EncodeCPU(const float *Positions, float *Features, size_t Count) const {
	if (Positions == nullptr || Features == nullptr) {
		throw std::invalid_argument("EncodeCPU: Positions and Features must not be null");
	}
	if (Count == 0) {
		return;
	}

	const float *tableData = _hashTable.data();
	const int	 outputDim = _outputDims;

	for (size_t i = 0; i < Count; i++) {
		const float *pos	= Positions + i * Dim;
		float *		featOut = Features + i * outputDim;

		// Clamp input to [0, 1] per axis
		float clampedPos[3] = {};
		for (int d = 0; d < Dim; d++) {
			clampedPos[d] = _config.ClampInput ? std::clamp(pos[d], 0.0f, 1.0f) : pos[d];
		}

		for (int level = 0; level < _numLevels; level++) {
			float resolution = GetLevelResolution(level);
			int	  tableOffset = level * _tableSize * _featuresPerLevel;

			if constexpr (Dim == 2) {
				// Bilinear interpolation
				float fx = clampedPos[0] * resolution - 0.5f;
				float fy = clampedPos[1] * resolution - 0.5f;

				int ix = static_cast<int>(std::floor(fx));
				int iy = static_cast<int>(std::floor(fy));

				float wx = fx - static_cast<float>(ix);
				float wy = fy - static_cast<float>(iy);

				float *levelFeat = featOut + level * _featuresPerLevel;
				std::memset(levelFeat, 0, _featuresPerLevel * sizeof(float));

				for (int dy = 0; dy <= 1; dy++) {
					for (int dx = 0; dx <= 1; dx++) {
						int	  cx	   = ix + dx;
						int	  cy	   = iy + dy;
						float weight  = (dx == 0 ? (1.0f - wx) : wx) * (dy == 0 ? (1.0f - wy) : wy);
						uint32_t hash = Hash2D(cx, cy);
						int		entry = static_cast<int>(hash & (static_cast<uint32_t>(_tableSize) - 1));
						int		addr  = tableOffset + entry * _featuresPerLevel;

						for (int c = 0; c < _featuresPerLevel; c++) {
							levelFeat[c] += weight * tableData[addr + c];
						}
					}
				}
			} else if constexpr (Dim == 3) {
				// Trilinear interpolation
				float fx = clampedPos[0] * resolution - 0.5f;
				float fy = clampedPos[1] * resolution - 0.5f;
				float fz = clampedPos[2] * resolution - 0.5f;

				int ix = static_cast<int>(std::floor(fx));
				int iy = static_cast<int>(std::floor(fy));
				int iz = static_cast<int>(std::floor(fz));

				float wx = fx - static_cast<float>(ix);
				float wy = fy - static_cast<float>(iy);
				float wz = fz - static_cast<float>(iz);

				float *levelFeat = featOut + level * _featuresPerLevel;
				std::memset(levelFeat, 0, _featuresPerLevel * sizeof(float));

				for (int dz = 0; dz <= 1; dz++) {
					for (int dy = 0; dy <= 1; dy++) {
						for (int dx = 0; dx <= 1; dx++) {
							int	  cx	   = ix + dx;
							int	  cy	   = iy + dy;
							int	  cz	   = iz + dz;
							float weight  = (dx == 0 ? (1.0f - wx) : wx) * (dy == 0 ? (1.0f - wy) : wy) *
										   (dz == 0 ? (1.0f - wz) : wz);
							uint32_t hash	= Hash3D(cx, cy, cz);
							int		 entry	= static_cast<int>(hash & (static_cast<uint32_t>(_tableSize) - 1));
							int		 addr	= tableOffset + entry * _featuresPerLevel;

							for (int c = 0; c < _featuresPerLevel; c++) {
								levelFeat[c] += weight * tableData[addr + c];
							}
						}
					}
				}
			}
		}
	}
}

// ============================================================================
// CPU Backward
// ============================================================================

template <int Dim>
void HashGridEncoding<Dim>::EncodeCPUBackward(const float *Positions, const float *FeatureGrads, float *TableGrads,
											  float *PositionGrads, size_t Count) const {
	if (Positions == nullptr || FeatureGrads == nullptr || TableGrads == nullptr) {
		throw std::invalid_argument("EncodeCPUBackward: Positions, FeatureGrads, and TableGrads must not be null");
	}
	if (Count == 0) {
		return;
	}

	const float *tableData = _hashTable.data();
	const int	 outputDim = _outputDims;

	for (size_t i = 0; i < Count; i++) {
		const float *pos	  = Positions + i * Dim;
		const float *gradIn	  = FeatureGrads + i * outputDim;
		float *		 posGrad  = PositionGrads ? (PositionGrads + i * Dim) : nullptr;

		// Clamp input to [0, 1] per axis
		float clampedPos[3] = {};
		for (int d = 0; d < Dim; d++) {
			clampedPos[d] = _config.ClampInput ? std::clamp(pos[d], 0.0f, 1.0f) : pos[d];
		}

		// Zero position gradients if requested
		if (posGrad) {
			for (int d = 0; d < Dim; d++) {
				posGrad[d] = 0.0f;
			}
		}

		for (int level = 0; level < _numLevels; level++) {
			float resolution = GetLevelResolution(level);
			int	  tableOffset = level * _tableSize * _featuresPerLevel;
			const float *levelGradIn = gradIn + level * _featuresPerLevel;

			if constexpr (Dim == 2) {
				float fx = clampedPos[0] * resolution - 0.5f;
				float fy = clampedPos[1] * resolution - 0.5f;

				int ix = static_cast<int>(std::floor(fx));
				int iy = static_cast<int>(std::floor(fy));

				float wx = fx - static_cast<float>(ix);
				float wy = fy - static_cast<float>(iy);

				for (int dy = 0; dy <= 1; dy++) {
					for (int dx = 0; dx <= 1; dx++) {
						int		 cx		= ix + dx;
						int		 cy		= iy + dy;
						float	 wxCoef = (dx == 0) ? (1.0f - wx) : wx;
						float	 wyCoef = (dy == 0) ? (1.0f - wy) : wy;
						float	 weight = wxCoef * wyCoef;
						uint32_t hash	= Hash2D(cx, cy);
						int		 entry	= static_cast<int>(hash & (static_cast<uint32_t>(_tableSize) - 1));
						int		 addr	= tableOffset + entry * _featuresPerLevel;

						for (int c = 0; c < _featuresPerLevel; c++) {
							float gradVal = weight * levelGradIn[c];
							TableGrads[addr + c] += gradVal;

							if (posGrad) {
								float tableVal = tableData[addr + c];
								float dWx	   = (dx == 0) ? -resolution : resolution;
								float dWy	   = (dy == 0) ? -resolution : resolution;
								float wxTerm   = levelGradIn[c] * tableVal;
								posGrad[0] += wxTerm * dWx * wyCoef;
								posGrad[1] += wxTerm * dWy * wxCoef;
							}
						}
					}
				}
			} else if constexpr (Dim == 3) {
				float fx = clampedPos[0] * resolution - 0.5f;
				float fy = clampedPos[1] * resolution - 0.5f;
				float fz = clampedPos[2] * resolution - 0.5f;

				int ix = static_cast<int>(std::floor(fx));
				int iy = static_cast<int>(std::floor(fy));
				int iz = static_cast<int>(std::floor(fz));

				float wx = fx - static_cast<float>(ix);
				float wy = fy - static_cast<float>(iy);
				float wz = fz - static_cast<float>(iz);

				for (int dz = 0; dz <= 1; dz++) {
					for (int dy = 0; dy <= 1; dy++) {
						for (int dx = 0; dx <= 1; dx++) {
							int		 cx		= ix + dx;
							int		 cy		= iy + dy;
							int		 cz		= iz + dz;
							float	 wxCoef = (dx == 0) ? (1.0f - wx) : wx;
							float	 wyCoef = (dy == 0) ? (1.0f - wy) : wy;
							float	 wzCoef = (dz == 0) ? (1.0f - wz) : wz;
							float	 weight = wxCoef * wyCoef * wzCoef;
							uint32_t hash	= Hash3D(cx, cy, cz);
							int		 entry	= static_cast<int>(hash & (static_cast<uint32_t>(_tableSize) - 1));
							int		 addr	= tableOffset + entry * _featuresPerLevel;

							for (int c = 0; c < _featuresPerLevel; c++) {
								float gradVal = weight * levelGradIn[c];
								TableGrads[addr + c] += gradVal;

								if (posGrad) {
									float tableVal = tableData[addr + c];
									float dWx	   = (dx == 0) ? -resolution : resolution;
									float dWy	   = (dy == 0) ? -resolution : resolution;
									float dWz	   = (dz == 0) ? -resolution : resolution;
									float wxTerm   = levelGradIn[c] * tableVal;
									posGrad[0] += wxTerm * dWx * wyCoef * wzCoef;
									posGrad[1] += wxTerm * dWy * wxCoef * wzCoef;
									posGrad[2] += wxTerm * dWz * wxCoef * wyCoef;
								}
							}
						}
					}
				}
			}
		}
	}
}

// ============================================================================
// Accessors
// ============================================================================

template <int Dim>
int HashGridEncoding<Dim>::GetOutputDims() const {
	return _outputDims;
}

template <int Dim>
float HashGridEncoding<Dim>::GetLevelResolution(int Level) const {
	return static_cast<float>(_config.BaseResolution) * std::pow(_config.PerLevelScale, static_cast<float>(Level));
}

template <int Dim>
int HashGridEncoding<Dim>::GetTableSize() const {
	return _tableSize;
}

template <int Dim>
size_t HashGridEncoding<Dim>::GetHashTableSize() const {
	return _hashTable.size();
}

template <int Dim>
const std::vector<float> &HashGridEncoding<Dim>::GetHashTable() const {
	return _hashTable;
}

template <int Dim>
void HashGridEncoding<Dim>::SetHashTable(const std::vector<float> &Data) {
	if (Data.size() != _hashTable.size()) {
		throw std::invalid_argument("SetHashTable: data size does not match hash table size");
	}
	_hashTable = Data;
	MarkTableDirty();
}

template <int Dim>
const HashGridConfig &HashGridEncoding<Dim>::GetConfig() const {
	return _config;
}

// ============================================================================
// GPU helpers
// ============================================================================

template <int Dim>
void HashGridEncoding<Dim>::MarkTableDirty() {
	if (_gpu) {
		_gpu->tableDirty = true;
	}
}

} // namespace HashEncoder

#endif // EASYGPU_HASHGRIDENCODING_INL
