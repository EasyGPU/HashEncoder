#pragma once

/**
 * @file HashGridEncoding.h
 * @brief Multi-resolution hash grid encoding for 2D and 3D spatial coordinates.
 *
 * Implements the hash grid encoding described in Instant-NGP (Muller et al.).
 * Given an input position in [0,1]^D, the encoder maps it through L levels
 * of trainable hash tables with bilinear (2D) or trilinear (3D) interpolation,
 * producing an output feature vector of length NumLevels * FeaturesPerLevel.
 *
 * GPU path uses EasyGPU's Kernel1D C++ DSL (not raw GLSL).
 */

#ifndef EASYGPU_HASHGRIDENCODING_H
#define EASYGPU_HASHGRIDENCODING_H

#include <HashEncoder/HashGridConfig.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward declarations for EasyGPU types used in the GPU path.
namespace GPU::Runtime {
template <typename T>
class Buffer;
} // namespace GPU::Runtime

namespace HashEncoder {

// ============================================================================
// Hash prime constants (from Instant-NGP)
// ============================================================================

/** @brief Hash prime for X coordinate. */
constexpr uint32_t kHashPrime1 = 73856093u;
/** @brief Hash prime for Y coordinate. */
constexpr uint32_t kHashPrime2 = 19349663u;
/** @brief Hash prime for Z coordinate (3D only). */
constexpr uint32_t kHashPrime3 = 83492791u;

// ============================================================================
// Hash function (CPU)
// ============================================================================

/**
 * @brief Compute integer-coordinate hash for 2D grid lookups.
 */
inline uint32_t Hash2D(int X, int Y) {
	return static_cast<uint32_t>(X) * kHashPrime1 ^ static_cast<uint32_t>(Y) * kHashPrime2;
}

/**
 * @brief Compute integer-coordinate hash for 3D grid lookups.
 */
inline uint32_t Hash3D(int X, int Y, int Z) {
	return static_cast<uint32_t>(X) * kHashPrime1 ^ static_cast<uint32_t>(Y) * kHashPrime2 ^
		   static_cast<uint32_t>(Z) * kHashPrime3;
}

// ============================================================================
// GPU state (PIMPL — defined in HashGridEncoding.cpp)
// ============================================================================

struct HashGridGPUState;

// ============================================================================
// HashGridEncoding
// ============================================================================

/**
 * @brief Multi-resolution hash grid encoding.
 * @tparam Dim Dimensionality: 2 for 2D bilinear, 3 for 3D trilinear.
 *
 * Encodes input positions in [0,1]^Dim into feature vectors by:
 *   1. Computing level-dependent grid resolution
 *   2. Hashing integer corner coordinates to a trainable feature table
 *   3. Bilinearly (2D) or trilinearly (3D) interpolating corner features
 *   4. Concatenating features across all levels
 *
 * GPU path uses EasyGPU Kernel1D DSL.
 */
template <int Dim>
class HashGridEncoding {
public:
	explicit HashGridEncoding(const HashGridConfig &Config);

	HashGridEncoding(const HashGridEncoding &)			  = delete;
	HashGridEncoding &operator=(const HashGridEncoding &) = delete;

	/** @brief Move constructor. GPU state pointer moves — kernel captures remain valid. */
	HashGridEncoding(HashGridEncoding &&) noexcept;

	/** @brief Move assignment. Old GPU resources released, then pointer move. */
	HashGridEncoding &operator=(HashGridEncoding &&) noexcept;

	/** @brief Destructor. GPU resources released via _gpu deleter. */
	~HashGridEncoding();

	// ========================================================================
	// Initialization
	// ========================================================================

	void Initialize(uint32_t Seed);
	void InitializeZero();

	// ========================================================================
	// CPU Encode
	// ========================================================================

	void EncodeCPU(const float *Positions, float *Features, size_t Count) const;

	// ========================================================================
	// GPU Encode (EasyGPU Kernel1D DSL)
	// ========================================================================

	/**
	 * @brief Encode positions on GPU using Kernel1D DSL.
	 * @param PositionBuffer GPU buffer of positions (Count * Dim floats).
	 * @param FeatureBuffer GPU buffer for output features (Count * GetOutputDims() floats).
	 * @param Count Number of positions.
	 */
	void EncodeGPU(const GPU::Runtime::Buffer<float> &PositionBuffer, GPU::Runtime::Buffer<float> &FeatureBuffer,
				   size_t Count);

	// ========================================================================
	// Accessors
	// ========================================================================

	[[nodiscard]] int GetOutputDims() const;
	[[nodiscard]] float GetLevelResolution(int Level) const;
	[[nodiscard]] int GetTableSize() const;
	[[nodiscard]] size_t GetHashTableSize() const;
	[[nodiscard]] const std::vector<float> &GetHashTable() const;
	void SetHashTable(const std::vector<float> &Data);
	[[nodiscard]] const HashGridConfig &GetConfig() const;

	// ========================================================================
	// Serialization
	// ========================================================================

	void Save(const std::string &FilePath) const;
	void Load(const std::string &FilePath);

	// ========================================================================
	// CPU Backward
	// ========================================================================

	/**
	 * @brief Compute gradients on CPU (reference implementation).
	 *
	 * Accumulates dL/dTable into TableGrads (caller must zero-init before first use).
	 * Optionally computes dL/dPosition.
	 */
	void EncodeCPUBackward(const float *Positions, const float *FeatureGrads, float *TableGrads, float *PositionGrads,
						   size_t Count) const;

	// ========================================================================
	// GPU Backward (EasyGPU Kernel1D DSL + AtomicAdd)
	// ========================================================================

	/**
	 * @brief Compute gradients on GPU using Kernel1D DSL with AtomicAdd.
	 *
	 * Accumulates dL/dTable into an internal gradient buffer via AtomicAdd.
	 * Optionally computes dL/dPosition. Call ZeroTableGradients() before
	 * accumulating a new batch, and GetTableGradients() to download results.
	 */
	void EncodeGPUBackward(const GPU::Runtime::Buffer<float> &PositionBuffer,
						   const GPU::Runtime::Buffer<float> &FeatureGradBuffer,
						   GPU::Runtime::Buffer<float> *PositionGradBuffer, size_t Count);

	void ZeroTableGradients();
	[[nodiscard]] std::vector<float> GetTableGradients() const;

private:
	// ========================================================================
	// GPU helpers
	// ========================================================================

	void EnsureGPUProgram();
	void EnsureGPUBackwardProgram();
	void MarkTableDirty();

	// ========================================================================
	// Data
	// ========================================================================

	HashGridConfig _config;
	std::vector<float> _hashTable;
	int _numLevels;
	int _featuresPerLevel;
	int _tableSize;
	int _outputDims;

	// GPU state (PIMPL — all EasyGPU objects live here, pointer stable across moves)
	std::unique_ptr<HashGridGPUState> _gpu;
};

// ============================================================================
// Convenience type aliases
// ============================================================================

using HashGridEncoding2D = HashGridEncoding<2>;
using HashGridEncoding3D = HashGridEncoding<3>;

} // namespace HashEncoder

#endif // EASYGPU_HASHGRIDENCODING_H
