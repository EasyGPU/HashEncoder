#include <HashEncoder/HashEncoder.h>
#include <GPU.h>
#include <Runtime/Buffer.h>
#include <Runtime/Context.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CHECK(cond, msg) \
	do { if (!(cond)) { printf("  FAIL: %s\n", msg); return 1; } } while (0)

int main() {
	try {
		GPU::Runtime::AutoInitContext();
		GPU::Runtime::Context::GetInstance().MakeCurrent();
	} catch (const std::exception &ex) {
		printf("SKIP (no GPU): %s\n", ex.what());
		return 0;
	}

	try {

	HashEncoder::HashGridConfig config;
	config.NumLevels       = 4;
	config.FeaturesPerLevel = 2;
	config.BaseResolution  = 16;
	config.Log2HashmapSize = 10;
	HashEncoder::HashGridEncoding3D encoder3D(config);
	encoder3D.Initialize(42);

	HashEncoder::HashGridConfig config2D = config;
	config2D.Dimensions = 2;
	HashEncoder::HashGridEncoding2D encoder2D(config2D);
	encoder2D.Initialize(42);

	// ------------------------------------------------------------------
	// Test 1: 3D GPU forward vs CPU
	// ------------------------------------------------------------------
	printf("Test 1: 3D GPU forward vs CPU... ");
	{
		const int count = 64;
		unsigned outDim = encoder3D.GetOutputDims();
		std::vector<float> positions(count * 3);
		std::mt19937 rng(555);
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		for (int i = 0; i < count * 3; i++) positions[i] = dist(rng);

		auto cpuFeat = std::vector<float>(count * outDim);
		encoder3D.EncodeCPU(positions.data(), cpuFeat.data(), count);

		GPU::Runtime::Buffer<float> posBuf(positions, GPU::Runtime::BufferMode::Read);
		GPU::Runtime::Buffer<float> featBuf(count * outDim, GPU::Runtime::BufferMode::ReadWrite);
		encoder3D.EncodeGPU(posBuf, featBuf, count);

		auto gpuFeat = std::vector<float>(count * outDim);
		featBuf.Download(gpuFeat.data(), gpuFeat.size());

		double maxDiff = 0.0;
		for (size_t i = 0; i < cpuFeat.size(); i++) {
			double d = std::abs(static_cast<double>(cpuFeat[i]) - static_cast<double>(gpuFeat[i]));
			if (d > maxDiff) maxDiff = d;
		}
		printf("diff=%.2e ", maxDiff);
		CHECK(maxDiff < 1e-4, "3D GPU forward mismatch");
		printf("PASS\n");
	}

	// ------------------------------------------------------------------
	// Test 2: 2D GPU forward vs CPU
	// ------------------------------------------------------------------
	printf("Test 2: 2D GPU forward vs CPU... ");
	{
		const int count = 64;
		unsigned outDim = encoder2D.GetOutputDims();
		std::vector<float> positions(count * 2);
		std::mt19937 rng(777);
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		for (int i = 0; i < count * 2; i++) positions[i] = dist(rng);

		auto cpuFeat = std::vector<float>(count * outDim);
		encoder2D.EncodeCPU(positions.data(), cpuFeat.data(), count);

		GPU::Runtime::Buffer<float> posBuf(positions, GPU::Runtime::BufferMode::Read);
		GPU::Runtime::Buffer<float> featBuf(count * outDim, GPU::Runtime::BufferMode::ReadWrite);
		encoder2D.EncodeGPU(posBuf, featBuf, count);

		auto gpuFeat = std::vector<float>(count * outDim);
		featBuf.Download(gpuFeat.data(), gpuFeat.size());

		double maxDiff = 0.0;
		for (size_t i = 0; i < cpuFeat.size(); i++) {
			double d = std::abs(static_cast<double>(cpuFeat[i]) - static_cast<double>(gpuFeat[i]));
			if (d > maxDiff) maxDiff = d;
		}
		printf("diff=%.2e ", maxDiff);
		CHECK(maxDiff < 1e-4, "2D GPU forward mismatch");
		printf("PASS\n");
	}

	// ------------------------------------------------------------------
	// Test 3: 3D GPU backward vs CPU
	// ------------------------------------------------------------------
	printf("Test 3: 3D GPU backward vs CPU... ");
	{
		const int count = 32;
		unsigned outDim = encoder3D.GetOutputDims();
		std::vector<float> positions(count * 3);
		std::mt19937 rng(123);
		std::uniform_real_distribution<float> posDist(0.0f, 1.0f);
		for (int i = 0; i < count * 3; i++) positions[i] = posDist(rng);

		auto gradIn = std::vector<float>(count * outDim);
		std::uniform_real_distribution<float> gradDist(-1.0f, 1.0f);
		for (size_t i = 0; i < gradIn.size(); i++) gradIn[i] = gradDist(rng);

		auto cpuTableGrads = std::vector<float>(encoder3D.GetHashTableSize(), 0.0f);
		auto cpuPosGrads   = std::vector<float>(count * 3, 0.0f);
		encoder3D.EncodeCPUBackward(positions.data(), gradIn.data(), cpuTableGrads.data(),
		                             cpuPosGrads.data(), count);

		GPU::Runtime::Buffer<float> posBuf(positions, GPU::Runtime::BufferMode::Read);
		GPU::Runtime::Buffer<float> gradBuf(gradIn, GPU::Runtime::BufferMode::Read);
		GPU::Runtime::Buffer<float> posGradBuf(count * 3, GPU::Runtime::BufferMode::ReadWrite);

		encoder3D.ZeroTableGradients();
		encoder3D.EncodeGPUBackward(posBuf, gradBuf, &posGradBuf, count);

		auto gpuTableGrads = encoder3D.GetTableGradients();
		auto gpuPosGrads   = std::vector<float>(count * 3);
		posGradBuf.Download(gpuPosGrads.data(), gpuPosGrads.size());

		double maxTableDiff = 0.0;
		for (size_t i = 0; i < cpuTableGrads.size(); i++) {
			double d = std::abs(static_cast<double>(cpuTableGrads[i]) - static_cast<double>(gpuTableGrads[i]));
			if (d > maxTableDiff) maxTableDiff = d;
		}
		double maxPosDiff = 0.0;
		for (size_t i = 0; i < cpuPosGrads.size(); i++) {
			double d = std::abs(static_cast<double>(cpuPosGrads[i]) - static_cast<double>(gpuPosGrads[i]));
			if (d > maxPosDiff) maxPosDiff = d;
		}
		printf("table=%.2e pos=%.2e ", maxTableDiff, maxPosDiff);
		CHECK(maxTableDiff < 1e-4, "3D GPU backward table grad mismatch");
		CHECK(maxPosDiff   < 1e-4, "3D GPU backward pos grad mismatch");
		printf("PASS\n");
	}

	// ------------------------------------------------------------------
	// Test 4: 2D GPU backward vs CPU
	// ------------------------------------------------------------------
	printf("Test 4: 2D GPU backward vs CPU... ");
	{
		const int count = 32;
		unsigned outDim = encoder2D.GetOutputDims();
		std::vector<float> positions(count * 2);
		std::mt19937 rng(456);
		std::uniform_real_distribution<float> posDist(0.0f, 1.0f);
		for (int i = 0; i < count * 2; i++) positions[i] = posDist(rng);

		auto gradIn = std::vector<float>(count * outDim);
		std::uniform_real_distribution<float> gradDist(-1.0f, 1.0f);
		for (size_t i = 0; i < gradIn.size(); i++) gradIn[i] = gradDist(rng);

		auto cpuTableGrads = std::vector<float>(encoder2D.GetHashTableSize(), 0.0f);
		auto cpuPosGrads   = std::vector<float>(count * 2, 0.0f);
		encoder2D.EncodeCPUBackward(positions.data(), gradIn.data(), cpuTableGrads.data(),
		                             cpuPosGrads.data(), count);

		GPU::Runtime::Buffer<float> posBuf(positions, GPU::Runtime::BufferMode::Read);
		GPU::Runtime::Buffer<float> gradBuf(gradIn, GPU::Runtime::BufferMode::Read);
		GPU::Runtime::Buffer<float> posGradBuf(count * 2, GPU::Runtime::BufferMode::ReadWrite);

		encoder2D.ZeroTableGradients();
		encoder2D.EncodeGPUBackward(posBuf, gradBuf, &posGradBuf, count);

		auto gpuTableGrads = encoder2D.GetTableGradients();
		auto gpuPosGrads   = std::vector<float>(count * 2);
		posGradBuf.Download(gpuPosGrads.data(), gpuPosGrads.size());

		double maxTableDiff = 0.0;
		for (size_t i = 0; i < cpuTableGrads.size(); i++) {
			double d = std::abs(static_cast<double>(cpuTableGrads[i]) - static_cast<double>(gpuTableGrads[i]));
			if (d > maxTableDiff) maxTableDiff = d;
		}
		double maxPosDiff = 0.0;
		for (size_t i = 0; i < cpuPosGrads.size(); i++) {
			double d = std::abs(static_cast<double>(cpuPosGrads[i]) - static_cast<double>(gpuPosGrads[i]));
			if (d > maxPosDiff) maxPosDiff = d;
		}
		printf("table=%.2e pos=%.2e ", maxTableDiff, maxPosDiff);
		CHECK(maxTableDiff < 1e-4, "2D GPU backward table grad mismatch");
		CHECK(maxPosDiff   < 1e-4, "2D GPU backward pos grad mismatch");
		printf("PASS\n");
	}

	// ------------------------------------------------------------------
	// Test 5: Cached pipeline reuse
	// ------------------------------------------------------------------
	printf("Test 5: Cached pipeline reuse... ");
	{
		const int count = 16;
		std::vector<float> positions(count * 3, 0.25f);
		unsigned outDim = encoder3D.GetOutputDims();

		GPU::Runtime::Buffer<float> posBuf(positions, GPU::Runtime::BufferMode::Read);
		GPU::Runtime::Buffer<float> featBuf(count * outDim, GPU::Runtime::BufferMode::ReadWrite);
		encoder3D.EncodeGPU(posBuf, featBuf, count);
		printf("PASS\n");
	}

	printf("\nAll GPU tests passed.\n");
	return 0;

	} catch (const std::exception &ex) {
		printf("FAIL: %s\n", ex.what());
		return 1;
	} catch (...) {
		printf("FAIL: unknown exception\n");
		return 1;
	}
}
