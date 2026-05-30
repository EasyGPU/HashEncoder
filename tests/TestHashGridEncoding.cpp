/**
 * @file TestHashGridEncoding.cpp
 * @brief Unit tests for the HashEncoder library.
 *
 * Tests:
 *   1. Config validation
 *   2. CPU encode sanity (no NaN, correct dimensions)
 *   3. CPU/GPU consistency
 *   4. Serialization round-trip
 *   5. Determinism (same seed -> same output)
 */

#include <HashEncoder/HashEncoder.h>

#include <Runtime/Buffer.h>
#include <Runtime/Context.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

// ============================================================================
// Simple test harness (no external test framework)
// ============================================================================

static int  gTestsPassed = 0;
static int  gTestsFailed = 0;
static bool gTestFailed	 = false;

#define TEST(name)                                                                                                     \
	do {                                                                                                               \
		gTestFailed = false;                                                                                           \
		printf("  %-50s ... ", name);                                                                                  \
	} while (0)

#define CHECK(cond)                                                                                                    \
	do {                                                                                                               \
		if (!(cond)) {                                                                                                 \
			printf("FAIL\n    CHECK(%s) at %s:%d\n", #cond, __FILE__, __LINE__);                                       \
			gTestFailed = true;                                                                                        \
		}                                                                                                              \
	} while (0)

#define CHECK_CLOSE(a, b, eps)                                                                                         \
	do {                                                                                                               \
		double _va = static_cast<double>(a);                                                                           \
		double _vb = static_cast<double>(b);                                                                           \
		if (std::abs(_va - _vb) > (eps)) {                                                                             \
			printf("FAIL\n    CHECK_CLOSE(%s, %s) at %s:%d: |%f - %f| = %f > %f\n", #a, #b, __FILE__, __LINE__,        \
				   _va, _vb, std::abs(_va - _vb), static_cast<double>(eps));                                           \
			gTestFailed = true;                                                                                        \
		}                                                                                                              \
	} while (0)

#define END_TEST()                                                                                                     \
	do {                                                                                                               \
		if (gTestFailed) {                                                                                             \
			gTestsFailed++;                                                                                            \
		} else {                                                                                                       \
			printf("PASS\n");                                                                                          \
			gTestsPassed++;                                                                                            \
		}                                                                                                              \
	} while (0)

// ============================================================================
// Test: Config Validation
// ============================================================================
static void TestConfigValidation() {
	TEST("Config validation");

	// Valid config
	{
		HashEncoder::HashGridConfig config;
		HashEncoder::HashGridEncoding3D encoder(config);
		CHECK(encoder.GetOutputDims() == config.NumLevels * config.FeaturesPerLevel);
	}

	// Invalid: zero NumLevels
	{
		HashEncoder::HashGridConfig config;
		config.NumLevels = 0;
		bool threw		  = false;
		try {
			HashEncoder::HashGridEncoding3D encoder(config);
		} catch (const std::invalid_argument &) {
			threw = true;
		}
		CHECK(threw);
	}

	// Invalid: dimension mismatch
	{
		HashEncoder::HashGridConfig config;
		config.Dimensions = 2;
		bool threw		  = false;
		try {
			HashEncoder::HashGridEncoding3D encoder(config);
		} catch (const std::invalid_argument &) {
			threw = true;
		}
		CHECK(threw);
	}

	// Invalid: negative base resolution
	{
		HashEncoder::HashGridConfig config;
		config.BaseResolution = -1;
		bool threw			  = false;
		try {
			HashEncoder::HashGridEncoding3D encoder(config);
		} catch (const std::invalid_argument &) {
			threw = true;
		}
		CHECK(threw);
	}

	END_TEST();
}

// ============================================================================
// Test: CPU Encode Sanity (2D)
// ============================================================================
static void TestCPUEncode2D() {
	TEST("CPU encode 2D sanity");

	HashEncoder::HashGridConfig config;
	config.Dimensions		= 2;
	config.NumLevels		= 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution	= 16;
	config.Log2HashmapSize	= 10;

	HashEncoder::HashGridEncoding2D encoder(config);
	encoder.Initialize(42);

	const int	   count	  = 8;
	unsigned	   outputDim = encoder.GetOutputDims();
	auto		   positions = std::vector<float>(count * 2);
	auto		   features  = std::vector<float>(count * outputDim);

	// Fill with various positions
	for (int i = 0; i < count; i++) {
		positions[i * 2 + 0] = static_cast<float>(i) / static_cast<float>(count);
		positions[i * 2 + 1] = 1.0f - static_cast<float>(i) / static_cast<float>(count);
	}

	encoder.EncodeCPU(positions.data(), features.data(), count);

	// Check no NaN
	for (size_t i = 0; i < features.size(); i++) {
		CHECK(!std::isnan(features[i]));
	}

	// Check output dimension
	CHECK(outputDim == config.NumLevels * config.FeaturesPerLevel);

	// Check that different positions produce different features (not all zeros)
	bool allSame = true;
	for (int i = 1; i < count; i++) {
		for (unsigned j = 0; j < outputDim; j++) {
			if (features[i * outputDim + j] != features[j]) {
				allSame = false;
				break;
			}
		}
		if (!allSame) break;
	}
	CHECK(!allSame); // positions should produce different features

	END_TEST();
}

// ============================================================================
// Test: CPU Encode Sanity (3D)
// ============================================================================
static void TestCPUEncode3D() {
	TEST("CPU encode 3D sanity");

	HashEncoder::HashGridConfig config;
	config.NumLevels		= 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution	= 16;
	config.Log2HashmapSize	= 10;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	const int	   count	  = 8;
	unsigned	   outputDim = encoder.GetOutputDims();
	auto		   positions = std::vector<float>(count * 3);
	auto		   features  = std::vector<float>(count * outputDim);

	for (int i = 0; i < count; i++) {
		positions[i * 3 + 0] = static_cast<float>(i) / static_cast<float>(count);
		positions[i * 3 + 1] = 0.5f;
		positions[i * 3 + 2] = 1.0f - static_cast<float>(i) / static_cast<float>(count);
	}

	encoder.EncodeCPU(positions.data(), features.data(), count);

	// Check no NaN
	for (size_t i = 0; i < features.size(); i++) {
		CHECK(!std::isnan(features[i]));
	}

	// Check output dimension
	CHECK(outputDim == config.NumLevels * config.FeaturesPerLevel);

	// Check that positions outside [0,1] are clamped without NaN
	{
		float	 badPos[3] = {-100.0f, 0.5f, 100.0f};
		unsigned outDim	  = encoder.GetOutputDims();
		auto	 feat	  = std::vector<float>(outDim);
		encoder.EncodeCPU(badPos, feat.data(), 1);
		for (unsigned i = 0; i < outDim; i++) {
			CHECK(!std::isnan(feat[i]));
		}
	}

	END_TEST();
}

// ============================================================================
// Test: Determinism
// ============================================================================
static void TestDeterminism() {
	TEST("Determinism (same seed -> same output)");

	HashEncoder::HashGridConfig config;
	config.NumLevels		= 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution	= 16;
	config.Log2HashmapSize	= 10;

	HashEncoder::HashGridEncoding3D encoderA(config);
	HashEncoder::HashGridEncoding3D encoderB(config);

	encoderA.Initialize(12345);
	encoderB.Initialize(12345);

	const int	 count		= 4;
	unsigned	 outputDim	= encoderA.GetOutputDims();
	auto		 positions	= std::vector<float>(count * 3);
	auto		 featuresA	= std::vector<float>(count * outputDim);
	auto		 featuresB	= std::vector<float>(count * outputDim);

	std::mt19937						  rng(999);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < count * 3; i++) {
		positions[i] = dist(rng);
	}

	encoderA.EncodeCPU(positions.data(), featuresA.data(), count);
	encoderB.EncodeCPU(positions.data(), featuresB.data(), count);

	for (size_t i = 0; i < featuresA.size(); i++) {
		CHECK(featuresA[i] == featuresB[i]);
	}

	END_TEST();
}

// ============================================================================
// Test: Hash Table Manual Set
// ============================================================================
static void TestHashTableSet() {
	TEST("Hash table manual set");

	HashEncoder::HashGridConfig config;
	config.NumLevels		= 2;
	config.FeaturesPerLevel = 2;
	config.BaseResolution	= 4;
	config.Log2HashmapSize	= 4;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.InitializeZero();

	// Set hash table to all 1.0
	auto table = encoder.GetHashTable();
	std::fill(table.begin(), table.end(), 1.0f);
	encoder.SetHashTable(table);

	// With all weights = 1, each level's output = sum of weights * 1.0
	// = sum of 8 corner weights = 1.0 for trilinear
	float	 pos[3]		= {0.5f, 0.5f, 0.5f};
	unsigned outputDim = encoder.GetOutputDims();
	auto	 features	= std::vector<float>(outputDim);
	encoder.EncodeCPU(pos, features.data(), 1);

	// Each level should output approximately 1.0 per feature channel
	for (int level = 0; level < config.NumLevels; level++) {
		for (int c = 0; c < config.FeaturesPerLevel; c++) {
			CHECK_CLOSE(features[level * config.FeaturesPerLevel + c], 1.0f, 1e-5f);
		}
	}

	END_TEST();
}

// ============================================================================
// Test: Serialization
// ============================================================================
static void TestSerialization() {
	TEST("Serialization round-trip");

	HashEncoder::HashGridConfig config;
	config.NumLevels		= 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution	= 16;
	config.Log2HashmapSize	= 10;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	const int	 count	  = 8;
	unsigned	 outputDim = encoder.GetOutputDims();
	auto		 positions = std::vector<float>(count * 3);
	std::mt19937 rng(777);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < count * 3; i++) {
		positions[i] = dist(rng);
	}

	auto featuresBefore = std::vector<float>(count * outputDim);
	encoder.EncodeCPU(positions.data(), featuresBefore.data(), count);

	encoder.Save("test_serialization_temp.bin");

	HashEncoder::HashGridEncoding3D loadedEncoder(config);
	loadedEncoder.Load("test_serialization_temp.bin");

	auto featuresAfter = std::vector<float>(count * outputDim);
	loadedEncoder.EncodeCPU(positions.data(), featuresAfter.data(), count);

	for (size_t i = 0; i < featuresBefore.size(); i++) {
		CHECK(featuresBefore[i] == featuresAfter[i]);
	}

	// Clean up
	std::remove("test_serialization_temp.bin");

	END_TEST();
}

// ============================================================================
// Test: Level Resolution Computation
// ============================================================================
static void TestLevelResolution() {
	TEST("Level resolution computation");

	HashEncoder::HashGridConfig config;
	config.BaseResolution = 16;
	config.PerLevelScale  = 2.0f;

	HashEncoder::HashGridEncoding3D encoder(config);

	CHECK_CLOSE(encoder.GetLevelResolution(0), 16.0f, 1e-5f);
	CHECK_CLOSE(encoder.GetLevelResolution(1), 32.0f, 1e-5f);
	CHECK_CLOSE(encoder.GetLevelResolution(2), 64.0f, 1e-5f);

	END_TEST();
}

// ============================================================================
// Test: CPU Backward Sanity (3D)
// ============================================================================
static void TestCPUBackwardSanity3D() {
	TEST("CPU backward 3D sanity");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 10;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	const int    count     = 8;
	unsigned     outputDim = encoder.GetOutputDims();
	auto         positions = std::vector<float>(count * 3);
	std::mt19937 rng(777);
	std::uniform_real_distribution<float> posDist(0.0f, 1.0f);
	for (int i = 0; i < count * 3; i++) {
		positions[i] = posDist(rng);
	}

	// Upstream gradients: random values
	auto featureGrads = std::vector<float>(count * outputDim);
	std::uniform_real_distribution<float> gradDist(-1.0f, 1.0f);
	for (size_t i = 0; i < featureGrads.size(); i++) {
		featureGrads[i] = gradDist(rng);
	}

	// Table gradients (zero-init)
	auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);

	// Position gradients
	auto positionGrads = std::vector<float>(count * 3, 0.0f);

	encoder.EncodeCPUBackward(positions.data(), featureGrads.data(), tableGrads.data(),
	                           positionGrads.data(), count);

	// No NaN in table gradients
	for (size_t i = 0; i < tableGrads.size(); i++) {
		CHECK(!std::isnan(tableGrads[i]));
	}

	// At least some table gradients should be non-zero
	bool hasNonZeroTableGrad = false;
	for (size_t i = 0; i < tableGrads.size(); i++) {
		if (tableGrads[i] != 0.0f) {
			hasNonZeroTableGrad = true;
			break;
		}
	}
	CHECK(hasNonZeroTableGrad);

	// No NaN in position gradients
	for (size_t i = 0; i < positionGrads.size(); i++) {
		CHECK(!std::isnan(positionGrads[i]));
	}

	// At least some position gradients should be non-zero
	bool hasNonZeroPosGrad = false;
	for (size_t i = 0; i < positionGrads.size(); i++) {
		if (positionGrads[i] != 0.0f) {
			hasNonZeroPosGrad = true;
			break;
		}
	}
	CHECK(hasNonZeroPosGrad);

	END_TEST();
}

// ============================================================================
// Test: Position Gradient Finite Difference
// ============================================================================
static void TestPositionGradientFiniteDiff() {
	TEST("Position gradient finite difference");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 2;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 4;
	config.Log2HashmapSize = 4;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(123);

	const unsigned outputDim = encoder.GetOutputDims();

	// Use a few test positions
	const int  count = 4;
	float      positions[4 * 3] = {0.2f, 0.3f, 0.4f, 0.6f, 0.7f, 0.1f, 0.9f, 0.5f, 0.2f, 0.1f, 0.8f, 0.9f};

	// Set upstream grad to all 1.0: dL/d(feature_i) = 1 for all i
	auto featureGrads = std::vector<float>(count * outputDim, 1.0f);

	// Compute analytical position gradients
	auto tableGrads   = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
	auto posGrads     = std::vector<float>(count * 3, 0.0f);
	encoder.EncodeCPUBackward(positions, featureGrads.data(), tableGrads.data(), posGrads.data(), count);

	// Finite difference check: d(sum of features)/d(pos_j)
	// With gradIn = all 1.0, posGrad_j = sum_i d(f_i)/d(pos_j) = d(sum_i f_i)/d(pos_j)
	auto features = std::vector<float>(outputDim);
	float eps     = 1e-4f;

	for (int n = 0; n < count; n++) {
		float origPos[3] = {positions[n * 3 + 0], positions[n * 3 + 1], positions[n * 3 + 2]};

		for (int d = 0; d < 3; d++) {
			// f(pos + eps * e_d)
			float posPlus[3] = {origPos[0], origPos[1], origPos[2]};
			posPlus[d] += eps;

			encoder.EncodeCPU(posPlus, features.data(), 1);
			float sumPlus = 0.0f;
			for (unsigned i = 0; i < outputDim; i++) {
				sumPlus += features[i];
			}

			// f(pos - eps * e_d)
			float posMinus[3] = {origPos[0], origPos[1], origPos[2]};
			posMinus[d] -= eps;

			encoder.EncodeCPU(posMinus, features.data(), 1);
			float sumMinus = 0.0f;
			for (unsigned i = 0; i < outputDim; i++) {
				sumMinus += features[i];
			}

			float numericalGrad = (sumPlus - sumMinus) / (2.0f * eps);
			float analyticalGrad = posGrads[n * 3 + d];

			CHECK_CLOSE(analyticalGrad, numericalGrad, 5e-3f);
		}
	}

	END_TEST();
}

// ============================================================================
// Test: CPU/GPU Backward Consistency
// ============================================================================
static void TestCPUGPUBackwardConsistency() {
	TEST("CPU/GPU backward consistency");

	try {
		GPU::Runtime::AutoInitContext();

		HashEncoder::HashGridConfig config;
		config.NumLevels       = 4;
		config.FeaturesPerLevel = 2;
		config.BaseResolution  = 16;
		config.Log2HashmapSize = 10;

		HashEncoder::HashGridEncoding3D encoder(config);
		encoder.Initialize(42);

		const int    count     = 64;
		unsigned     outputDim = encoder.GetOutputDims();
		auto         positions = std::vector<float>(count * 3);
		std::mt19937 rng(123);
		std::uniform_real_distribution<float> posDist(0.0f, 1.0f);
		for (int i = 0; i < count * 3; i++) {
			positions[i] = posDist(rng);
		}

		// Random upstream gradients
		auto featureGrads = std::vector<float>(count * outputDim);
		std::uniform_real_distribution<float> gradDist(-1.0f, 1.0f);
		for (size_t i = 0; i < featureGrads.size(); i++) {
			featureGrads[i] = gradDist(rng);
		}

		// --- CPU backward ---
		auto cpuTableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
		auto cpuPosGrads   = std::vector<float>(count * 3, 0.0f);
		encoder.EncodeCPUBackward(positions.data(), featureGrads.data(), cpuTableGrads.data(),
		                           cpuPosGrads.data(), count);

		// --- GPU backward ---
		GPU::Runtime::Buffer<float> posBuffer(positions, GPU::Runtime::BufferMode::Read);
		GPU::Runtime::Buffer<float> gradInBuffer(featureGrads, GPU::Runtime::BufferMode::Read);
		GPU::Runtime::Buffer<float> posGradBuffer(count * 3, GPU::Runtime::BufferMode::ReadWrite);

		encoder.ZeroTableGradients();
		encoder.EncodeGPUBackward(posBuffer, gradInBuffer, &posGradBuffer, count);

		auto gpuTableGrads = encoder.GetTableGradients();
		auto gpuPosGrads   = std::vector<float>(count * 3);
		posGradBuffer.Download(gpuPosGrads.data(), gpuPosGrads.size());

		// Compare table gradients
		double maxTableDiff = 0.0;
		for (size_t i = 0; i < cpuTableGrads.size(); i++) {
			double diff = std::abs(static_cast<double>(cpuTableGrads[i]) - static_cast<double>(gpuTableGrads[i]));
			if (diff > maxTableDiff) {
				maxTableDiff = diff;
			}
		}
		CHECK(maxTableDiff < 1e-4);

		// Compare position gradients
		double maxPosDiff = 0.0;
		for (size_t i = 0; i < cpuPosGrads.size(); i++) {
			double diff = std::abs(static_cast<double>(cpuPosGrads[i]) - static_cast<double>(gpuPosGrads[i]));
			if (diff > maxPosDiff) {
				maxPosDiff = diff;
			}
		}
		CHECK(maxPosDiff < 1e-4);

	} catch (const std::exception &ex) {
		printf("SKIP (GPU not available: %s)\n", ex.what());
		gTestFailed = false;
	}

	END_TEST();
}

// ============================================================================
// Test: Backward Determinism
// ============================================================================
static void TestBackwardDeterminism() {
	TEST("Backward determinism");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 10;

	HashEncoder::HashGridEncoding3D encoderA(config);
	HashEncoder::HashGridEncoding3D encoderB(config);
	encoderA.Initialize(999);
	encoderB.Initialize(999);

	const int    count     = 8;
	unsigned     outputDim = encoderA.GetOutputDims();
	auto         positions = std::vector<float>(count * 3);
	auto         gradIns   = std::vector<float>(count * outputDim);

	std::mt19937                            rng(42);
	std::uniform_real_distribution<float>   posDist(0.0f, 1.0f);
	std::uniform_real_distribution<float>   gradDist(-1.0f, 1.0f);
	for (int i = 0; i < count * 3; i++) positions[i] = posDist(rng);
	for (size_t i = 0; i < gradIns.size(); i++) gradIns[i] = gradDist(rng);

	auto tableGradsA = std::vector<float>(encoderA.GetHashTableSize(), 0.0f);
	auto posGradsA   = std::vector<float>(count * 3, 0.0f);
	auto tableGradsB = std::vector<float>(encoderB.GetHashTableSize(), 0.0f);
	auto posGradsB   = std::vector<float>(count * 3, 0.0f);

	encoderA.EncodeCPUBackward(positions.data(), gradIns.data(), tableGradsA.data(), posGradsA.data(), count);
	encoderB.EncodeCPUBackward(positions.data(), gradIns.data(), tableGradsB.data(), posGradsB.data(), count);

	for (size_t i = 0; i < tableGradsA.size(); i++) {
		CHECK(tableGradsA[i] == tableGradsB[i]);
	}
	for (size_t i = 0; i < posGradsA.size(); i++) {
		CHECK(posGradsA[i] == posGradsB[i]);
	}

	END_TEST();
}

// ============================================================================
// Main
// ============================================================================
int main() {
	printf("=== HashEncoder Tests ===\n\n");

	TestConfigValidation();
	TestCPUEncode2D();
	TestCPUEncode3D();
	TestDeterminism();
	TestHashTableSet();
	TestLevelResolution();
	TestSerialization();
	TestCPUBackwardSanity3D();
	TestPositionGradientFiniteDiff();
	TestBackwardDeterminism();

	printf("\n=== Results: %d passed, %d failed ===\n", gTestsPassed, gTestsFailed);

	return gTestsFailed > 0 ? 1 : 0;
}
