/**
 * @file TestHashGridStress.cpp
 * @brief Stress / edge-case tests for HashEncoder library.
 *
 * Covers config extremes, position boundaries, gradient extremes,
 * serialization stress, and random config testing.
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
// Simple test harness
// ============================================================================

static int  gTestsPassed = 0;
static int  gTestsFailed = 0;
static bool gTestFailed  = false;

#define TEST(name)                                                                                                     \
	do {                                                                                                               \
		gTestFailed = false;                                                                                           \
		printf("  %-55s ... ", name);                                                                                  \
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
// Config Extremes
// ============================================================================

static void TestMaxLevelsConfig() {
	TEST("Max levels config (16 levels * 8 feat * 2^20 table)");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 16;
	config.FeaturesPerLevel = 8;
	config.Log2HashmapSize = 20;
	config.BaseResolution  = 16;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	CHECK(encoder.GetOutputDims() == 16 * 8);

	// Single encode should not crash or produce NaN
	float pos[3] = {0.5f, 0.5f, 0.5f};
	std::vector<float> features(encoder.GetOutputDims());
	encoder.EncodeCPU(pos, features.data(), 1);

	for (size_t i = 0; i < features.size(); i++) {
		CHECK(!std::isnan(features[i]));
		CHECK(!std::isinf(features[i]));
	}

	END_TEST();
}

static void TestMinConfig() {
	TEST("Min config (1 level, 1 feat, 2^1 table, base res 1)");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 1;
	config.FeaturesPerLevel = 1;
	config.Log2HashmapSize = 1;
	config.BaseResolution  = 1;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	CHECK(encoder.GetOutputDims() == 1);

	float pos[3] = {0.25f, 0.75f, 0.5f};
	float feat;
	encoder.EncodeCPU(pos, &feat, 1);
	CHECK(!std::isnan(feat));
	CHECK(!std::isinf(feat));

	END_TEST();
}

static void TestHighPerLevelScale() {
	TEST("High PerLevelScale (4.0)");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 8;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 4;
	config.PerLevelScale   = 4.0f;
	config.Log2HashmapSize = 14;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(99);

	// Level 7 resolution = 4 * 4^7 = 65536 — very fine grid
	float res7 = encoder.GetLevelResolution(7);
	CHECK_CLOSE(res7, 4.0f * std::pow(4.0f, 7.0f), 1e-3f);

	const int count = 16;
	std::vector<float> positions(count * 3);
	std::vector<float> features(count * encoder.GetOutputDims());

	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < count * 3; i++) positions[i] = dist(rng);

	encoder.EncodeCPU(positions.data(), features.data(), count);
	for (size_t i = 0; i < features.size(); i++) {
		CHECK(!std::isnan(features[i]));
		CHECK(!std::isinf(features[i]));
	}

	END_TEST();
}

static void TestPerLevelScaleOne() {
	TEST("PerLevelScale = 1.0 (same resolution all levels)");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 32;
	config.PerLevelScale   = 1.0f;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(123);

	for (int lvl = 0; lvl < 4; lvl++) {
		CHECK_CLOSE(encoder.GetLevelResolution(lvl), 32.0f, 1e-5f);
	}

	END_TEST();
}

// ============================================================================
// Invalid Config Edge Cases
// ============================================================================

static void TestInvalidConfigs() {
	TEST("Invalid config edge cases");

	// Log2HashmapSize = 0
	{
		HashEncoder::HashGridConfig config;
		config.Log2HashmapSize = 0;
		bool threw = false;
		try { HashEncoder::HashGridEncoding3D e(config); }
		catch (const std::invalid_argument &) { threw = true; }
		CHECK(threw);
	}

	// Log2HashmapSize = 31 (too large)
	{
		HashEncoder::HashGridConfig config;
		config.Log2HashmapSize = 31;
		bool threw = false;
		try { HashEncoder::HashGridEncoding3D e(config); }
		catch (const std::invalid_argument &) { threw = true; }
		CHECK(threw);
	}

	// BaseResolution = 0
	{
		HashEncoder::HashGridConfig config;
		config.BaseResolution = 0;
		bool threw = false;
		try { HashEncoder::HashGridEncoding3D e(config); }
		catch (const std::invalid_argument &) { threw = true; }
		CHECK(threw);
	}

	// FeaturesPerLevel = 0
	{
		HashEncoder::HashGridConfig config;
		config.FeaturesPerLevel = 0;
		bool threw = false;
		try { HashEncoder::HashGridEncoding3D e(config); }
		catch (const std::invalid_argument &) { threw = true; }
		CHECK(threw);
	}

	// PerLevelScale = 0
	{
		HashEncoder::HashGridConfig config;
		config.PerLevelScale = 0.0f;
		bool threw = false;
		try { HashEncoder::HashGridEncoding3D e(config); }
		catch (const std::invalid_argument &) { threw = true; }
		CHECK(threw);
	}

	// Negative NumLevels
	{
		HashEncoder::HashGridConfig config;
		config.NumLevels = -1;
		bool threw = false;
		try { HashEncoder::HashGridEncoding3D e(config); }
		catch (const std::invalid_argument &) { threw = true; }
		CHECK(threw);
	}

	END_TEST();
}

// ============================================================================
// Position Extremes
// ============================================================================

static void TestGridCorners() {
	TEST("Position at exact grid corners");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	float corners[8][3] = {
		{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
		{1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f},
	};

	int outDim = encoder.GetOutputDims();
	std::vector<float> features(8 * outDim);
	encoder.EncodeCPU(&corners[0][0], features.data(), 8);

	for (size_t i = 0; i < features.size(); i++) {
		CHECK(!std::isnan(features[i]));
		CHECK(!std::isinf(features[i]));
	}

	END_TEST();
}

static void TestPositionsFarOutside() {
	TEST("Positions far outside [0,1] with clamping");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	float extremePos[5][3] = {
		{-1e6f, -1e6f, -1e6f},
		{ 1e6f,  1e6f,  1e6f},
		{-1e6f,  0.5f,  1e6f},
		{ 0.5f, -1e6f,  0.5f},
		{ 1e6f,  1e6f, -1e6f},
	};

	int outDim = encoder.GetOutputDims();
	std::vector<float> features(5 * outDim);
	encoder.EncodeCPU(&extremePos[0][0], features.data(), 5);

	for (size_t i = 0; i < features.size(); i++) {
		CHECK(!std::isnan(features[i]));
		CHECK(!std::isinf(features[i]));
	}

	END_TEST();
}

static void TestClampInputFalse() {
	TEST("ClampInput=false with extreme positions");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;
	config.ClampInput      = false;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	// Values outside [0,1] should not crash — may produce large but finite results
	float extremePos[4][3] = {
		{-10.0f, -10.0f, -10.0f},
		{ 10.0f,  10.0f,  10.0f},
		{-10.0f,   0.5f,  10.0f},
		{  0.5f,  10.0f,  -5.0f},
	};

	int outDim = encoder.GetOutputDims();
	std::vector<float> features(4 * outDim);
	encoder.EncodeCPU(&extremePos[0][0], features.data(), 4);

	for (size_t i = 0; i < features.size(); i++) {
		CHECK(!std::isnan(features[i]));
		CHECK(!std::isinf(features[i]));
	}

	// ClampInput=false with backward should also not crash
	auto gradIn  = std::vector<float>(4 * outDim, 1.0f);
	auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
	auto posGrads   = std::vector<float>(4 * 3, 0.0f);
	encoder.EncodeCPUBackward(&extremePos[0][0], gradIn.data(), tableGrads.data(), posGrads.data(), 4);

	for (size_t i = 0; i < tableGrads.size(); i++) {
		CHECK(!std::isnan(tableGrads[i]));
	}
	for (size_t i = 0; i < posGrads.size(); i++) {
		CHECK(!std::isnan(posGrads[i]));
	}

	END_TEST();
}

static void TestTinyPerturbedPositions() {
	TEST("Tiny perturbed positions (eps=1e-7)");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	float eps = 1e-7f;
	float positions[4][3] = {
		{eps, eps, eps},
		{eps, 1.0f - eps, 0.5f},
		{1.0f - eps, eps, 1.0f - eps},
		{0.5f, 0.5f, eps},
	};

	int outDim = encoder.GetOutputDims();
	std::vector<float> features(4 * outDim);
	encoder.EncodeCPU(&positions[0][0], features.data(), 4);

	for (size_t i = 0; i < features.size(); i++) {
		CHECK(!std::isnan(features[i]));
		CHECK(!std::isinf(features[i]));
	}

	END_TEST();
}

static void TestAllIdenticalPositions() {
	TEST("All identical positions");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	const int count = 100;
	int outDim = encoder.GetOutputDims();
	std::vector<float> positions(count * 3, 0.0f);
	for (int i = 0; i < count; i++) {
		positions[i * 3 + 0] = 0.5f;
		positions[i * 3 + 1] = 0.5f;
		positions[i * 3 + 2] = 0.5f;
	}

	std::vector<float> features(count * outDim);
	encoder.EncodeCPU(positions.data(), features.data(), count);

	// All outputs should be identical
	for (int i = 1; i < count; i++) {
		for (int j = 0; j < outDim; j++) {
			CHECK(features[i * outDim + j] == features[j]);
		}
	}

	// Backward: accumulate from all identical points
	auto gradIn  = std::vector<float>(count * outDim, 1.0f);
	auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
	auto posGrads   = std::vector<float>(count * 3, 0.0f);
	encoder.EncodeCPUBackward(positions.data(), gradIn.data(), tableGrads.data(), posGrads.data(), count);

	for (size_t i = 0; i < tableGrads.size(); i++) {
		CHECK(!std::isnan(tableGrads[i]));
	}
	for (size_t i = 0; i < posGrads.size(); i++) {
		CHECK(!std::isnan(posGrads[i]));
	}

	// Position gradients should be identical across all identical points
	for (int i = 1; i < count; i++) {
		for (int d = 0; d < 3; d++) {
			CHECK(posGrads[i * 3 + d] == posGrads[d]);
		}
	}

	END_TEST();
}

static void TestSinglePosition() {
	TEST("Single position (count=1)");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 10;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	float pos[3] = {0.314f, 0.159f, 0.265f};
	int outDim = encoder.GetOutputDims();
	std::vector<float> features(outDim);
	encoder.EncodeCPU(pos, features.data(), 1);

	for (int i = 0; i < outDim; i++) {
		CHECK(!std::isnan(features[i]));
	}

	// Backward with count=1
	auto gradIn = std::vector<float>(outDim, 1.0f);
	auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
	float posGrad[3] = {};
	encoder.EncodeCPUBackward(pos, gradIn.data(), tableGrads.data(), posGrad, 1);

	for (size_t i = 0; i < tableGrads.size(); i++) {
		CHECK(!std::isnan(tableGrads[i]));
	}
	for (int d = 0; d < 3; d++) {
		CHECK(!std::isnan(posGrad[d]));
	}

	END_TEST();
}

static void TestZeroCount() {
	TEST("Zero count (no-op)");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	float dummy = 0;
	encoder.EncodeCPU(&dummy, &dummy, 0);

	auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
	encoder.EncodeCPUBackward(&dummy, &dummy, tableGrads.data(), nullptr, 0);

	// Table gradients should remain zero
	for (size_t i = 0; i < tableGrads.size(); i++) {
		CHECK(tableGrads[i] == 0.0f);
	}

	END_TEST();
}

// ============================================================================
// Gradient Extremes
// ============================================================================

static void TestZeroUpstreamGradients() {
	TEST("Zero upstream gradients");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	const int count = 16;
	int outDim = encoder.GetOutputDims();

	std::vector<float> positions(count * 3);
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < count * 3; i++) positions[i] = dist(rng);

	auto gradIn  = std::vector<float>(count * outDim, 0.0f);
	auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), -999.0f);
	auto posGrads   = std::vector<float>(count * 3, -999.0f);

	encoder.EncodeCPUBackward(positions.data(), gradIn.data(), tableGrads.data(), posGrads.data(), count);

	// All table gradients should still be -999 (nothing accumulated)
	for (size_t i = 0; i < tableGrads.size(); i++) {
		CHECK(tableGrads[i] == -999.0f);
	}

	END_TEST();
}

static void TestLargeUpstreamGradients() {
	TEST("Very large upstream gradients (1e6)");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	float pos[3] = {0.5f, 0.5f, 0.5f};
	int outDim = encoder.GetOutputDims();

	auto gradIn  = std::vector<float>(outDim, 1e6f);
	auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
	float posGrad[3] = {};

	encoder.EncodeCPUBackward(pos, gradIn.data(), tableGrads.data(), posGrad, 1);

	for (size_t i = 0; i < tableGrads.size(); i++) {
		CHECK(!std::isnan(tableGrads[i]));
		CHECK(!std::isinf(tableGrads[i]));
	}
	for (int d = 0; d < 3; d++) {
		CHECK(!std::isnan(posGrad[d]));
		CHECK(!std::isinf(posGrad[d]));
	}

	// At least some gradients should be non-zero
	bool hasNonZero = false;
	for (size_t i = 0; i < tableGrads.size(); i++) {
		if (tableGrads[i] != 0.0f) { hasNonZero = true; break; }
	}
	CHECK(hasNonZero);

	END_TEST();
}

static void TestGradientAccumulation() {
	TEST("Gradient accumulation correctness");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	float pos[3] = {0.5f, 0.5f, 0.5f};
	int outDim = encoder.GetOutputDims();
	auto gradIn = std::vector<float>(outDim, 1.0f);

	// First backward pass
	auto tableGrads1 = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
	encoder.EncodeCPUBackward(pos, gradIn.data(), tableGrads1.data(), nullptr, 1);

	// Second backward pass (accumulate into same buffer)
	encoder.EncodeCPUBackward(pos, gradIn.data(), tableGrads1.data(), nullptr, 1);

	// Single pass with 2x gradients should match
	auto tableGrads2 = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
	auto gradIn2x = std::vector<float>(outDim, 2.0f);
	encoder.EncodeCPUBackward(pos, gradIn2x.data(), tableGrads2.data(), nullptr, 1);

	for (size_t i = 0; i < tableGrads1.size(); i++) {
		CHECK_CLOSE(tableGrads1[i], tableGrads2[i], 1e-5f);
	}

	END_TEST();
}

static void TestNullPositionGrad() {
	TEST("Backward with nullptr position gradients");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	const int count = 32;
	int outDim = encoder.GetOutputDims();

	std::vector<float> positions(count * 3);
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < count * 3; i++) positions[i] = dist(rng);

	auto gradIn  = std::vector<float>(count * outDim, 1.0f);
	auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);

	encoder.EncodeCPUBackward(positions.data(), gradIn.data(), tableGrads.data(), nullptr, count);

	for (size_t i = 0; i < tableGrads.size(); i++) {
		CHECK(!std::isnan(tableGrads[i]));
	}

	// Should have non-zero table gradients (backward still computes table grads)
	bool hasNonZero = false;
	for (size_t i = 0; i < tableGrads.size(); i++) {
		if (tableGrads[i] != 0.0f) { hasNonZero = true; break; }
	}
	CHECK(hasNonZero);

	END_TEST();
}

// ============================================================================
// 2D Backward
// ============================================================================

static void Test2DBackwardFiniteDiff() {
	TEST("2D backward finite difference");

	HashEncoder::HashGridConfig config;
	config.Dimensions      = 2;
	config.NumLevels       = 2;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 4;
	config.Log2HashmapSize = 4;

	HashEncoder::HashGridEncoding2D encoder(config);
	encoder.Initialize(123);

	const int outDim = encoder.GetOutputDims();
	float pos[2] = {0.35f, 0.65f};

	auto gradIn = std::vector<float>(outDim, 1.0f);
	auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
	float posGrad[2] = {};

	encoder.EncodeCPUBackward(pos, gradIn.data(), tableGrads.data(), posGrad, 1);

	// Finite difference check
	float eps = 1e-4f;
	auto features = std::vector<float>(outDim);

	for (int d = 0; d < 2; d++) {
		float posPlus[2] = {pos[0], pos[1]};
		posPlus[d] += eps;
		encoder.EncodeCPU(posPlus, features.data(), 1);
		float sumPlus = 0.0f;
		for (int j = 0; j < outDim; j++) sumPlus += features[j];

		float posMinus[2] = {pos[0], pos[1]};
		posMinus[d] -= eps;
		encoder.EncodeCPU(posMinus, features.data(), 1);
		float sumMinus = 0.0f;
		for (int j = 0; j < outDim; j++) sumMinus += features[j];

		float numericalGrad = (sumPlus - sumMinus) / (2.0f * eps);
		CHECK_CLOSE(posGrad[d], numericalGrad, 5e-3f);
	}

	END_TEST();
}

static void Test2DCPUEncodeSanity() {
	TEST("2D CPU encode extended sanity");

	HashEncoder::HashGridConfig config;
	config.Dimensions      = 2;
	config.NumLevels       = 8;
	config.FeaturesPerLevel = 4;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 12;

	HashEncoder::HashGridEncoding2D encoder(config);
	encoder.Initialize(777);

	const int count = 256;
	int outDim = encoder.GetOutputDims();

	std::vector<float> positions(count * 2);
	std::vector<float> features(count * outDim);

	std::mt19937 rng(123);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < count * 2; i++) positions[i] = dist(rng);

	encoder.EncodeCPU(positions.data(), features.data(), count);

	for (size_t i = 0; i < features.size(); i++) {
		CHECK(!std::isnan(features[i]));
		CHECK(!std::isinf(features[i]));
	}

	// All features should be small (initialized near zero)
	for (size_t i = 0; i < features.size(); i++) {
		CHECK(std::abs(features[i]) < 1.0f);
	}

	END_TEST();
}

// ============================================================================
// Large Batch
// ============================================================================

static void TestVeryLargeBatch() {
	TEST("Very large batch (65536 positions)");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 10;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	const int count = 65536;
	int outDim = encoder.GetOutputDims();

	std::vector<float> positions(count * 3);
	std::vector<float> features(count * outDim);

	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < count * 3; i++) positions[i] = dist(rng);

	encoder.EncodeCPU(positions.data(), features.data(), count);

	// Check for NaN/Inf
	for (size_t i = 0; i < features.size(); i++) {
		CHECK(!std::isnan(features[i]));
		CHECK(!std::isinf(features[i]));
	}

	// Backward
	auto gradIn = std::vector<float>(count * outDim, 1.0f);
	auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
	auto posGrads   = std::vector<float>(count * 3, 0.0f);

	encoder.EncodeCPUBackward(positions.data(), gradIn.data(), tableGrads.data(), posGrads.data(), count);

	for (size_t i = 0; i < tableGrads.size(); i++) {
		CHECK(!std::isnan(tableGrads[i]));
		CHECK(!std::isinf(tableGrads[i]));
	}
	for (size_t i = 0; i < posGrads.size(); i++) {
		CHECK(!std::isnan(posGrads[i]));
		CHECK(!std::isinf(posGrads[i]));
	}

	END_TEST();
}

// ============================================================================
// Random Config Stress
// ============================================================================

static void TestRandomConfigStress() {
	TEST("Random config stress (10 random valid configs)");

	std::mt19937 rng(888);
	std::uniform_int_distribution<int> levelsDist(1, 12);
	std::uniform_int_distribution<int> featDist(1, 8);
	std::uniform_int_distribution<int> baseResDist(1, 64);
	std::uniform_real_distribution<float> scaleDist(1.1f, 3.0f);
	std::uniform_int_distribution<int> log2Dist(4, 18);
	std::uniform_int_distribution<int> dimDist(0, 1);

	for (int trial = 0; trial < 10; trial++) {
		HashEncoder::HashGridConfig config;
		config.NumLevels       = levelsDist(rng);
		config.FeaturesPerLevel = featDist(rng);
		config.BaseResolution  = baseResDist(rng);
		config.PerLevelScale   = scaleDist(rng);
		config.Log2HashmapSize = log2Dist(rng);

		int dim = dimDist(rng) == 0 ? 2 : 3;
		config.Dimensions = dim;

		const int count = 64;

		if (dim == 2) {
			HashEncoder::HashGridEncoding2D encoder(config);
			encoder.Initialize(rng());

			int outDim = encoder.GetOutputDims();
			std::vector<float> positions(count * 2);
			std::vector<float> features(count * outDim);
			std::uniform_real_distribution<float> posDist(0.0f, 1.0f);
			for (int i = 0; i < count * 2; i++) positions[i] = posDist(rng);

			encoder.EncodeCPU(positions.data(), features.data(), count);
			for (size_t i = 0; i < features.size(); i++) {
				CHECK(!std::isnan(features[i]));
				CHECK(!std::isinf(features[i]));
			}

			// Backward
			auto gradIn = std::vector<float>(count * outDim);
			std::uniform_real_distribution<float> gradDist(-1.0f, 1.0f);
			for (size_t i = 0; i < gradIn.size(); i++) gradIn[i] = gradDist(rng);

			auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
			auto posGrads   = std::vector<float>(count * 2, 0.0f);
			encoder.EncodeCPUBackward(positions.data(), gradIn.data(), tableGrads.data(), posGrads.data(), count);

			for (size_t i = 0; i < tableGrads.size(); i++) CHECK(!std::isnan(tableGrads[i]));
			for (size_t i = 0; i < posGrads.size(); i++) CHECK(!std::isnan(posGrads[i]));
		} else {
			HashEncoder::HashGridEncoding3D encoder(config);
			encoder.Initialize(rng());

			int outDim = encoder.GetOutputDims();
			std::vector<float> positions(count * 3);
			std::vector<float> features(count * outDim);
			std::uniform_real_distribution<float> posDist(0.0f, 1.0f);
			for (int i = 0; i < count * 3; i++) positions[i] = posDist(rng);

			encoder.EncodeCPU(positions.data(), features.data(), count);
			for (size_t i = 0; i < features.size(); i++) {
				CHECK(!std::isnan(features[i]));
				CHECK(!std::isinf(features[i]));
			}

			auto gradIn = std::vector<float>(count * outDim);
			std::uniform_real_distribution<float> gradDist(-1.0f, 1.0f);
			for (size_t i = 0; i < gradIn.size(); i++) gradIn[i] = gradDist(rng);

			auto tableGrads = std::vector<float>(encoder.GetHashTableSize(), 0.0f);
			auto posGrads   = std::vector<float>(count * 3, 0.0f);
			encoder.EncodeCPUBackward(positions.data(), gradIn.data(), tableGrads.data(), posGrads.data(), count);

			for (size_t i = 0; i < tableGrads.size(); i++) CHECK(!std::isnan(tableGrads[i]));
			for (size_t i = 0; i < posGrads.size(); i++) CHECK(!std::isnan(posGrads[i]));
		}
	}

	END_TEST();
}

// ============================================================================
// Serialization Stress
// ============================================================================

static void TestLargeTableSerialization() {
	TEST("Large table serialization (16L * 2^14 * 8F = 2M floats)");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 16;
	config.FeaturesPerLevel = 8;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 14;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(42);

	const int count = 16;
	int outDim = encoder.GetOutputDims();

	std::vector<float> positions(count * 3);
	std::mt19937 rng(555);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < count * 3; i++) positions[i] = dist(rng);

	auto featuresBefore = std::vector<float>(count * outDim);
	encoder.EncodeCPU(positions.data(), featuresBefore.data(), count);

	encoder.Save("test_stress_serialization.bin");

	HashEncoder::HashGridEncoding3D loadedEncoder(config);
	loadedEncoder.Load("test_stress_serialization.bin");

	auto featuresAfter = std::vector<float>(count * outDim);
	loadedEncoder.EncodeCPU(positions.data(), featuresAfter.data(), count);

	for (size_t i = 0; i < featuresBefore.size(); i++) {
		CHECK(featuresBefore[i] == featuresAfter[i]);
	}

	std::remove("test_stress_serialization.bin");
	END_TEST();
}

static void TestSaveLoadDeterminism() {
	TEST("Save/Load determinism across multiple cycles");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 10;

	HashEncoder::HashGridEncoding3D encoderA(config);
	encoderA.Initialize(999);

	// Save, load into B, save B, load into C — all should produce same output
	encoderA.Save("test_det_a.bin");

	HashEncoder::HashGridEncoding3D encoderB(config);
	encoderB.Load("test_det_a.bin");
	encoderB.Save("test_det_b.bin");

	HashEncoder::HashGridEncoding3D encoderC(config);
	encoderC.Load("test_det_b.bin");

	const int count = 16;
	int outDim = encoderA.GetOutputDims();
	std::vector<float> positions(count * 3);
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < count * 3; i++) positions[i] = dist(rng);

	auto featA = std::vector<float>(count * outDim);
	auto featB = std::vector<float>(count * outDim);
	auto featC = std::vector<float>(count * outDim);

	encoderA.EncodeCPU(positions.data(), featA.data(), count);
	encoderB.EncodeCPU(positions.data(), featB.data(), count);
	encoderC.EncodeCPU(positions.data(), featC.data(), count);

	for (size_t i = 0; i < featA.size(); i++) {
		CHECK(featA[i] == featB[i]);
		CHECK(featB[i] == featC[i]);
	}

	std::remove("test_det_a.bin");
	std::remove("test_det_b.bin");
	END_TEST();
}

// ============================================================================
// Output Dimension Consistency
// ============================================================================

static void TestOutputDimConsistency() {
	TEST("Output dimension consistency across configs");

	for (int nl = 1; nl <= 8; nl++) {
		for (int fp = 1; fp <= 4; fp++) {
			HashEncoder::HashGridConfig config;
			config.NumLevels       = nl;
			config.FeaturesPerLevel = fp;
			config.BaseResolution  = 16;
			config.Log2HashmapSize = 8;

			HashEncoder::HashGridEncoding3D encoder(config);
			CHECK(encoder.GetOutputDims() == nl * fp);

			float pos[3] = {0.5f, 0.5f, 0.5f};
			std::vector<float> features(nl * fp);
			encoder.Initialize(42);
			encoder.EncodeCPU(pos, features.data(), 1);

			CHECK(static_cast<int>(features.size()) == nl * fp);
			for (size_t i = 0; i < features.size(); i++) {
				CHECK(!std::isnan(features[i]));
			}
		}
	}

	END_TEST();
}

// ============================================================================
// InitializeZero
// ============================================================================

static void TestInitializeZero() {
	TEST("InitializeZero gives all-zero output");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.InitializeZero();

	float pos[3] = {0.5f, 0.5f, 0.5f};
	int outDim = encoder.GetOutputDims();
	std::vector<float> features(outDim);
	encoder.EncodeCPU(pos, features.data(), 1);

	for (int i = 0; i < outDim; i++) {
		CHECK(features[i] == 0.0f);
	}

	END_TEST();
}

// ============================================================================
// Random vs Zero Init
// ============================================================================

static void TestRandomInitProducesVariedOutput() {
	TEST("Random init produces non-uniform features");

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 8;

	HashEncoder::HashGridEncoding3D encoder(config);
	encoder.Initialize(12345);

	const int count = 100;
	int outDim = encoder.GetOutputDims();
	std::vector<float> positions(count * 3);
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < count * 3; i++) positions[i] = dist(rng);

	std::vector<float> features(count * outDim);
	encoder.EncodeCPU(positions.data(), features.data(), count);

	// Different feature channels should have different values
	bool allSame = true;
	float firstVal = features[0];
	for (size_t i = 1; i < features.size(); i++) {
		if (features[i] != firstVal) { allSame = false; break; }
	}
	CHECK(!allSame);

	END_TEST();
}

// ============================================================================
// Main
// ============================================================================

int main() {
	printf("=== HashEncoder Stress Tests ===\n\n");

	// Config extremes
	TestMaxLevelsConfig();
	TestMinConfig();
	TestHighPerLevelScale();
	TestPerLevelScaleOne();

	// Invalid configs
	TestInvalidConfigs();

	// Position extremes
	TestGridCorners();
	TestPositionsFarOutside();
	TestClampInputFalse();
	TestTinyPerturbedPositions();
	TestAllIdenticalPositions();
	TestSinglePosition();
	TestZeroCount();

	// Gradient extremes
	TestZeroUpstreamGradients();
	TestLargeUpstreamGradients();
	TestGradientAccumulation();
	TestNullPositionGrad();

	// 2D backward
	Test2DBackwardFiniteDiff();
	Test2DCPUEncodeSanity();

	// Large batch
	TestVeryLargeBatch();

	// Random config
	TestRandomConfigStress();

	// Serialization stress
	TestLargeTableSerialization();
	TestSaveLoadDeterminism();

	// Misc
	TestOutputDimConsistency();
	TestInitializeZero();
	TestRandomInitProducesVariedOutput();

	printf("\n=== Results: %d passed, %d failed ===\n", gTestsPassed, gTestsFailed);

	return gTestsFailed > 0 ? 1 : 0;
}
