/**
 * @file main.cpp
 * @brief Minimal demo of HashEncoder: 2D and 3D hash grid encoding.
 *
 * Demonstrates:
 *   - Configuration
 *   - Parameter initialization
 *   - CPU encode
 *   - Output dimension inspection
 *   - Serialization round-trip
 *   - GPU encode (if GPU context is available)
 */

#include <HashEncoder/HashEncoder.h>

#include <Runtime/Buffer.h>
#include <Runtime/Context.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
	printf("=== HashEncoder Demo ===\n\n");

	// ========================================================================
	// 1. Configuration
	// ========================================================================
	HashEncoder::HashGridConfig config;
	config.NumLevels		 = 8;
	config.FeaturesPerLevel	 = 2;
	config.BaseResolution	 = 16;
	config.PerLevelScale	 = 1.5f;
	config.Log2HashmapSize	 = 14;
	config.ClampInput		 = true;

	printf("Configuration:\n");
	printf("  Dimensions: %d\n", config.Dimensions);
	printf("  NumLevels: %d\n", config.NumLevels);
	printf("  FeaturesPerLevel: %d\n", config.FeaturesPerLevel);
	printf("  BaseResolution: %d\n", config.BaseResolution);
	printf("  PerLevelScale: %.2f\n", config.PerLevelScale);
	printf("  Log2HashmapSize: %d\n", config.Log2HashmapSize);
	printf("  OutputDims: %d\n\n", config.NumLevels * config.FeaturesPerLevel);

	// ========================================================================
	// 2. Create 3D encoder
	// ========================================================================
	printf("--- 3D Hash Grid Encoding ---\n");

	HashEncoder::HashGridEncoding3D encoder3D(config);
	encoder3D.Initialize(42);

	printf("Table size per level: %d entries\n", encoder3D.GetTableSize());
	printf("Total hash table size: %zu floats\n", encoder3D.GetHashTableSize());
	printf("Output dims: %d\n\n", encoder3D.GetOutputDims());

	// ========================================================================
	// 3. CPU encode - single position
	// ========================================================================
	{
		float	 position[3] = {0.5f, 0.5f, 0.5f};
		unsigned outputDim	  = encoder3D.GetOutputDims();
		auto	 features	  = std::vector<float>(outputDim);

		encoder3D.EncodeCPU(position, features.data(), 1);

		printf("Single position (0.5, 0.5, 0.5):\n");
		printf("  Encoded features (first 8): ");
		for (int i = 0; i < 8 && i < outputDim; i++) {
			printf("%.6f ", static_cast<double>(features[i]));
		}
		printf("\n");

		// Verify no NaN
		bool hasNaN = false;
		for (int i = 0; i < outputDim; i++) {
			if (std::isnan(features[i])) {
				hasNaN = true;
				break;
			}
		}
		printf("  Contains NaN: %s\n\n", hasNaN ? "YES (FAIL)" : "no (ok)");
	}

	// ========================================================================
	// 4. CPU encode - batch
	// ========================================================================
	{
		const int	   batchSize  = 1024;
		unsigned	   outputDim  = encoder3D.GetOutputDims();
		auto		   positions  = std::vector<float>(batchSize * 3);
		auto		   features   = std::vector<float>(batchSize * outputDim);

		std::mt19937						  rng(123);
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		for (int i = 0; i < batchSize * 3; i++) {
			positions[i] = dist(rng);
		}

		encoder3D.EncodeCPU(positions.data(), features.data(), batchSize);

		// Check for NaN
		bool hasNaN = false;
		for (size_t i = 0; i < features.size(); i++) {
			if (std::isnan(features[i])) {
				hasNaN = true;
				break;
			}
		}
		printf("Batch encode (%d positions):\n", batchSize);
		printf("  First feature: %.6f\n", static_cast<double>(features[0]));
		printf("  Contains NaN: %s\n\n", hasNaN ? "YES (FAIL)" : "no (ok)");
	}

	// ========================================================================
	// 5. Serialization round-trip
	// ========================================================================
	{
		const int	batchSize = 4;
		unsigned	outputDim = encoder3D.GetOutputDims();
		auto		positions = std::vector<float>(batchSize * 3);
		std::mt19937 rng(456);
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		for (int i = 0; i < batchSize * 3; i++) {
			positions[i] = dist(rng);
		}

		// Encode before save
		auto featuresBefore = std::vector<float>(batchSize * outputDim);
		encoder3D.EncodeCPU(positions.data(), featuresBefore.data(), batchSize);

		// Save and reload
		encoder3D.Save("hash_grid_demo_temp.bin");

		HashEncoder::HashGridEncoding3D loadedEncoder(config);
		loadedEncoder.Load("hash_grid_demo_temp.bin");

		// Encode after load
		auto featuresAfter = std::vector<float>(batchSize * outputDim);
		loadedEncoder.EncodeCPU(positions.data(), featuresAfter.data(), batchSize);

		// Compare
		bool match = true;
		for (size_t i = 0; i < featuresBefore.size(); i++) {
			if (featuresBefore[i] != featuresAfter[i]) {
				match = false;
				printf("  Mismatch at index %zu: %f vs %f\n", i, static_cast<double>(featuresBefore[i]),
					   static_cast<double>(featuresAfter[i]));
				break;
			}
		}
		printf("Serialization round-trip: %s\n", match ? "PASS (outputs match)" : "FAIL (outputs differ)");

		// Clean up temp file
		std::remove("hash_grid_demo_temp.bin");
	}

	// ========================================================================
	// 6. 2D encoder quick test
	// ========================================================================
	printf("\n--- 2D Hash Grid Encoding ---\n");

	HashEncoder::HashGridConfig config2D;
	config2D.Dimensions		 = 2;
	config2D.NumLevels		 = 8;
	config2D.FeaturesPerLevel = 2;
	config2D.BaseResolution	 = 16;
	config2D.PerLevelScale	 = 1.5f;
	config2D.Log2HashmapSize = 14;

	HashEncoder::HashGridEncoding2D encoder2D(config2D);
	encoder2D.Initialize(123);

	{
		float	 position[2] = {0.25f, 0.75f};
		unsigned outputDim	  = encoder2D.GetOutputDims();
		auto	 features	  = std::vector<float>(outputDim);

		encoder2D.EncodeCPU(position, features.data(), 1);

		printf("2D position (0.25, 0.75):\n");
		printf("  Output dims: %d\n", outputDim);
		printf("  Features: ");
		for (int i = 0; i < outputDim; i++) {
			printf("%.6f ", static_cast<double>(features[i]));
		}
		printf("\n");
	}

	// ========================================================================
	// 7. GPU encode (if available)
	// ========================================================================
	printf("\n--- GPU Encode ---\n");

	try {
		GPU::Runtime::AutoInitContext();

		const int	batchSize = 256;
		unsigned	outputDim = encoder3D.GetOutputDims();
		auto		positions = std::vector<float>(batchSize * 3);
		std::mt19937 rng(789);
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		for (int i = 0; i < batchSize * 3; i++) {
			positions[i] = dist(rng);
		}

		// CPU encode
		auto cpuFeatures = std::vector<float>(batchSize * outputDim);
		encoder3D.EncodeCPU(positions.data(), cpuFeatures.data(), batchSize);

		// GPU encode
		GPU::Runtime::Buffer<float> posBuffer(positions, GPU::Runtime::BufferMode::Read);
		GPU::Runtime::Buffer<float> featBuffer(batchSize * outputDim, GPU::Runtime::BufferMode::ReadWrite);

		encoder3D.EncodeGPU(posBuffer, featBuffer, batchSize);

		// Download GPU results
		auto gpuFeatures = std::vector<float>(batchSize * outputDim);
		featBuffer.Download(gpuFeatures.data(), gpuFeatures.size());

		// Compare CPU vs GPU
		double maxDiff = 0.0;
		for (size_t i = 0; i < cpuFeatures.size(); i++) {
			double diff = std::abs(static_cast<double>(cpuFeatures[i]) - static_cast<double>(gpuFeatures[i]));
			if (diff > maxDiff) {
				maxDiff = diff;
			}
		}
		printf("CPU vs GPU max difference: %.10f\n", maxDiff);
		printf("GPU consistency: %s (threshold 1e-4)\n", maxDiff < 1e-4 ? "PASS" : "FAIL (difference too large)");

	} catch (const std::exception &ex) {
		printf("GPU encode skipped: %s\n", ex.what());
	}

	// ========================================================================
	// 8. CPU Backward pass demo
	// ========================================================================
	printf("\n--- CPU Backward Pass ---\n");
	{
		// Use a small encoder for clarity
		HashEncoder::HashGridConfig bwConfig;
		bwConfig.NumLevels       = 4;
		bwConfig.FeaturesPerLevel = 2;
		bwConfig.BaseResolution  = 16;
		bwConfig.Log2HashmapSize = 10;

		HashEncoder::HashGridEncoding3D bwEncoder(bwConfig);
		bwEncoder.Initialize(123);

		unsigned outputDim = bwEncoder.GetOutputDims();

		// Two positions with random upstream gradients
		const int  count    = 2;
		float      positions[6] = {0.5f, 0.5f, 0.5f, 0.1f, 0.9f, 0.2f};
		auto       featureGrads = std::vector<float>(count * outputDim);
		auto       tableGrads   = std::vector<float>(bwEncoder.GetHashTableSize(), 0.0f);
		auto       posGrads     = std::vector<float>(count * 3, 0.0f);

		// Upstream gradient: identity-ish (one-hot per position for simplicity)
		for (int n = 0; n < count; n++) {
			featureGrads[n * outputDim + 0] = 1.0f;  // dL/d(feature_0) = 1
		}

		bwEncoder.EncodeCPUBackward(positions, featureGrads.data(), tableGrads.data(),
		                             posGrads.data(), count);

		// Show position gradients
		for (int n = 0; n < count; n++) {
			printf("Position %d (%.2f, %.2f, %.2f):\n", n,
			       static_cast<double>(positions[n * 3 + 0]),
			       static_cast<double>(positions[n * 3 + 1]),
			       static_cast<double>(positions[n * 3 + 2]));
			printf("  dL/dPos = (%.6f, %.6f, %.6f)\n",
			       static_cast<double>(posGrads[n * 3 + 0]),
			       static_cast<double>(posGrads[n * 3 + 1]),
			       static_cast<double>(posGrads[n * 3 + 2]));
		}

		// Count non-zero table gradients
		size_t nonZeroGrads = 0;
		for (size_t i = 0; i < tableGrads.size(); i++) {
			if (tableGrads[i] != 0.0f) nonZeroGrads++;
		}
		printf("Non-zero table gradients: %zu / %zu\n", nonZeroGrads, tableGrads.size());

		// Verify no NaN
		bool hasNaN = false;
		for (size_t i = 0; i < tableGrads.size(); i++) {
			if (std::isnan(tableGrads[i])) { hasNaN = true; break; }
		}
		printf("Table gradients NaN check: %s\n", hasNaN ? "FAIL" : "ok");
	}

	// ========================================================================
	// 9. GPU Backward pass demo
	// ========================================================================
	printf("\n--- GPU Backward Pass ---\n");

	try {
		GPU::Runtime::AutoInitContext();

		HashEncoder::HashGridConfig bwConfig;
		bwConfig.NumLevels       = 4;
		bwConfig.FeaturesPerLevel = 2;
		bwConfig.BaseResolution  = 16;
		bwConfig.Log2HashmapSize = 10;

		HashEncoder::HashGridEncoding3D bwEncoder(bwConfig);
		bwEncoder.Initialize(123);

		const int  count    = 32;
		unsigned   outputDim = bwEncoder.GetOutputDims();
		auto       positions = std::vector<float>(count * 3);
		auto       featureGrads = std::vector<float>(count * outputDim);

		std::mt19937                            rng(555);
		std::uniform_real_distribution<float>   posDist(0.0f, 1.0f);
		std::uniform_real_distribution<float>   gradDist(-1.0f, 1.0f);
		for (int i = 0; i < count * 3; i++) positions[i] = posDist(rng);
		for (size_t i = 0; i < featureGrads.size(); i++) featureGrads[i] = gradDist(rng);

		// CPU backward
		auto cpuTableGrads = std::vector<float>(bwEncoder.GetHashTableSize(), 0.0f);
		bwEncoder.EncodeCPUBackward(positions.data(), featureGrads.data(), cpuTableGrads.data(),
		                             nullptr, count);

		// GPU backward (no position gradient for simplicity)
		GPU::Runtime::Buffer<float> posBuffer(positions, GPU::Runtime::BufferMode::Read);
		GPU::Runtime::Buffer<float> gradInBuffer(featureGrads, GPU::Runtime::BufferMode::Read);

		bwEncoder.ZeroTableGradients();
		bwEncoder.EncodeGPUBackward(posBuffer, gradInBuffer, nullptr, count);
		auto gpuTableGrads = bwEncoder.GetTableGradients();

		// Compare
		double maxDiff = 0.0;
		for (size_t i = 0; i < cpuTableGrads.size(); i++) {
			double diff = std::abs(static_cast<double>(cpuTableGrads[i]) - static_cast<double>(gpuTableGrads[i]));
			if (diff > maxDiff) maxDiff = diff;
		}
		printf("CPU vs GPU table gradient max difference: %.10f\n", maxDiff);
		printf("GPU backward consistency: %s (threshold 1e-4)\n", maxDiff < 1e-4 ? "PASS" : "FAIL");

	} catch (const std::exception &ex) {
		printf("GPU backward skipped: %s\n", ex.what());
	}

	printf("\n=== Demo Complete ===\n");
	return 0;
}
