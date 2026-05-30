#pragma once

/**
 * @file HashEncoder.h
 * @brief Umbrella header for the HashEncoder library.
 *
 * Multi-resolution hash grid encoding for spatial coordinates,
 * compatible with the EasyGPU compute ecosystem.
 *
 * Quick start:
 * @code
 * #include <HashEncoder/HashEncoder.h>
 *
 * HashEncoder::HashGridConfig config;
 * config.NumLevels = 16;
 * config.FeaturesPerLevel = 2;
 *
 * HashEncoder::HashGridEncoding3D encoder(config);
 * encoder.Initialize(42);
 *
 * std::vector<float> positions = {0.5f, 0.5f, 0.5f};
 * std::vector<float> features(encoder.GetOutputDims());
 * encoder.EncodeCPU(positions.data(), features.data(), 1);
 * @endcode
 */

#ifndef EASYGPU_HASHENCODER_H
#define EASYGPU_HASHENCODER_H

#include <HashEncoder/HashGridConfig.h>
#include <HashEncoder/HashGridEncoding.h>

#endif // EASYGPU_HASHENCODER_H
