#pragma once

/**
 * @file HashGridConfig.h
 * @brief Configuration struct for multi-resolution hash grid encoding.
 *
 * Controls the parameters of a HashGridEncoding instance, including
 * dimensionality, number of levels, feature count per level, base resolution,
 * hash table size, and input clamping behavior.
 */

#ifndef EASYGPU_HASHGRIDCONFIG_H
#define EASYGPU_HASHGRIDCONFIG_H

#include <cstdint>

namespace HashEncoder {

/**
 * @brief Configuration for multi-resolution hash grid encoding.
 *
 * Default values follow the Instant-NGP convention:
 *   - 16 levels
 *   - 2 features per level
 *   - base resolution of 16
 *   - per-level scale of 1.5
 *   - hash table size of 2^19 entries per level
 */
struct HashGridConfig {
	int		Dimensions		= 3;
	int		NumLevels		= 16;
	int		FeaturesPerLevel = 2;
	int		BaseResolution	= 16;
	float	PerLevelScale	= 1.5f;
	int		Log2HashmapSize = 19;
	bool	ClampInput		= true;
};

} // namespace HashEncoder

#endif // EASYGPU_HASHGRIDCONFIG_H
