/**
 * @file main.cpp
 * @brief End-to-end training: NeuralHashGrid learns a 2D scalar field.
 *
 * Demonstrates the Instant-NGP pattern:
 *   (x, y) -> hash encode -> MLP -> scalar value
 *
 * Target function: f(x, y) = sin(2*pi*x) * cos(2*pi*y), a smooth bump map.
 */

#include <HashEncoder/NeuralHashGrid.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static float TargetFunc(float x, float y) {
	return std::sin(2.0f * 3.14159265f * x) * std::cos(2.0f * 3.14159265f * y);
}

int main() {
	printf("=== NeuralHashGrid 2D Training Demo ===\n\n");

	// ------------------------------------------------------------------
	// 1. Configure a small model for fast training
	// ------------------------------------------------------------------
	HashEncoder::HashGridConfig encCfg;
	encCfg.Dimensions       = 2;
	encCfg.NumLevels        = 4;
	encCfg.FeaturesPerLevel = 2;
	encCfg.BaseResolution   = 8;
	encCfg.PerLevelScale    = 2.0f;
	encCfg.Log2HashmapSize  = 8;

	HashEncoder::MiniMLP::Config mlpCfg;
	mlpCfg.InputDim        = encCfg.NumLevels * encCfg.FeaturesPerLevel;  // 8
	mlpCfg.HiddenDim       = 16;
	mlpCfg.OutputDim       = 1;
	mlpCfg.NumHiddenLayers = 1;

	printf("Configuration:\n");
	printf("  Encoder: %d levels x %d features = %d dims\n",
	       encCfg.NumLevels, encCfg.FeaturesPerLevel, mlpCfg.InputDim);
	printf("  MLP: [%d] -> [%d] -> [%d]\n",
	       mlpCfg.InputDim, mlpCfg.HiddenDim, mlpCfg.OutputDim);
	printf("  Hash table: %d x 2^%d x %d = %d params\n",
	       encCfg.NumLevels, encCfg.Log2HashmapSize, encCfg.FeaturesPerLevel,
	       encCfg.NumLevels * (1 << encCfg.Log2HashmapSize) * encCfg.FeaturesPerLevel);
	printf("  MLP params: %d\n",
	       mlpCfg.InputDim * mlpCfg.HiddenDim + mlpCfg.HiddenDim +
	       mlpCfg.HiddenDim * mlpCfg.OutputDim + mlpCfg.OutputDim);

	// ------------------------------------------------------------------
	// 2. Create model and scale up initial hash table
	// ------------------------------------------------------------------
	HashEncoder::NeuralHashGrid<2> model(encCfg, mlpCfg);
	model.Initialize(42);

	// Boost hash table magnitude so encoder features are meaningful.
	// Standard init U(-1e-4,1e-4) gives features ~1e-5, invisible to Xavier MLP.
	{
		auto table = model.GetEncoder().GetHashTable();
		for (auto &v : table) v *= 100.0f;
		model.GetEncoder().SetHashTable(table);
	}

	// ------------------------------------------------------------------
	// 3. Generate training data
	// ------------------------------------------------------------------
	const int trainCount = 1024;
	const int testCount  = 256;

	std::mt19937 rng(1337);
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	std::vector<float> trainPos(trainCount * 2);
	std::vector<float> trainTarget(trainCount);
	for (int i = 0; i < trainCount; i++) {
		float x = dist(rng);
		float y = dist(rng);
		trainPos[i * 2 + 0] = x;
		trainPos[i * 2 + 1] = y;
		trainTarget[i]       = TargetFunc(x, y);
	}

	std::vector<float> testPos(testCount * 2);
	std::vector<float> testTarget(testCount);
	for (int i = 0; i < testCount; i++) {
		float x = dist(rng);
		float y = dist(rng);
		testPos[i * 2 + 0] = x;
		testPos[i * 2 + 1] = y;
		testTarget[i]       = TargetFunc(x, y);
	}

	printf("\nTraining samples: %d, Test samples: %d\n", trainCount, testCount);

	// ------------------------------------------------------------------
	// 4. Sanity check before training
	// ------------------------------------------------------------------
	{
		auto &enc = model.GetEncoder();
		int featDim = enc.GetOutputDims();
		float p1[2] = {0.2f, 0.3f};
		float p2[2] = {0.8f, 0.7f};
		std::vector<float> f1(featDim), f2(featDim);
		enc.EncodeCPU(p1, f1.data(), 1);
		enc.EncodeCPU(p2, f2.data(), 1);
		float d = 0;
		for (int i = 0; i < featDim; i++) d += (f1[i]-f2[i])*(f1[i]-f2[i]);
		printf("Encoder L2 dist between 2 points: %.4f\n", static_cast<double>(std::sqrt(d)));

		std::vector<float> out1(1), out2(1);
		model.Forward(p1, out1.data(), 1);
		model.Forward(p2, out2.data(), 1);
		printf("MLP output at (0.2,0.3): %.4f  (target: %.4f)\n",
		       static_cast<double>(out1[0]), static_cast<double>(TargetFunc(0.2f, 0.3f)));
		printf("MLP output at (0.8,0.7): %.4f  (target: %.4f)\n",
		       static_cast<double>(out2[0]), static_cast<double>(TargetFunc(0.8f, 0.7f)));
	}

	// ------------------------------------------------------------------
	// 5. Training
	// ------------------------------------------------------------------
	const int   steps = 2000;
	const float lr    = 0.2f;

	printf("\nTraining for %d steps (lr=%.3f, effective=%.6f)...\n",
	       steps, static_cast<double>(lr),
	       static_cast<double>(lr / trainCount));
	printf("  Step  | Train Loss | Test RMSE\n");
	printf("  ------|------------|----------\n");

	float bestLoss = 1e10f;

	for (int step = 0; step < steps; step++) {
		float trainLoss = model.TrainStep(trainPos.data(), trainTarget.data(), trainCount, lr);

		if (trainLoss < bestLoss) bestLoss = trainLoss;

		if (step % 200 == 0 || step == steps - 1) {
			std::vector<float> testOut(testCount);
			model.Forward(testPos.data(), testOut.data(), testCount);

			float rmse = 0.0f;
			for (int i = 0; i < testCount; i++) {
				float e = testOut[i] - testTarget[i];
				rmse += e * e;
			}
			rmse = std::sqrt(rmse / static_cast<float>(testCount));

			printf("  %5d | %10.6f | %9.6f\n",
			       step, static_cast<double>(trainLoss), static_cast<double>(rmse));
		}
	}

	// ------------------------------------------------------------------
	// 6. Final evaluation
	// ------------------------------------------------------------------
	printf("\n--- Final Evaluation ---\n");
	{
		std::vector<float> testOut(testCount);
		model.Forward(testPos.data(), testOut.data(), testCount);

		float mae = 0.0f, rmse = 0.0f;
		for (int i = 0; i < testCount; i++) {
			float e = std::abs(testOut[i] - testTarget[i]);
			mae += e;
			rmse += (testOut[i] - testTarget[i]) * (testOut[i] - testTarget[i]);
		}
		mae  /= static_cast<float>(testCount);
		rmse  = std::sqrt(rmse / static_cast<float>(testCount));

		printf("Test MAE:  %.6f\n", static_cast<double>(mae));
		printf("Test RMSE: %.6f\n", static_cast<double>(rmse));
		printf("Best train loss: %.6f\n", static_cast<double>(bestLoss));

		// Show sample predictions
		printf("\nSample predictions:\n");
		printf("  Position     | Predicted | Target   | Error\n");
		printf("  -------------|-----------|----------|-------\n");
		for (int i = 0; i < 8 && i < testCount; i++) {
			const float *p = testPos.data() + i * 2;
			printf("  (%.2f, %.2f)  | %9.4f | %8.4f | %6.4f\n",
			       static_cast<double>(p[0]), static_cast<double>(p[1]),
			       static_cast<double>(testOut[i]),
			       static_cast<double>(testTarget[i]),
			       static_cast<double>(std::abs(testOut[i] - testTarget[i])));
		}
	}

	// Check for NaN divergence
	if (std::isnan(bestLoss)) {
		printf("\nFAIL: Model diverged to NaN\n");
		return 1;
	}

	printf("\n=== Demo Complete ===\n");
	return 0;
}
