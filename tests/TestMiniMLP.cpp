/**
 * Quick MiniMLP sanity test — learns a simple linear function.
 */
#include <HashEncoder/NeuralHashGrid.h>
#include <cmath>
#include <cstdio>

int main() {
	printf("=== MiniMLP Sanity Test ===\n");

	// Learn f(x) = [2*x, -x, 1-x] from 3D input
	HashEncoder::MiniMLP::Config cfg;
	cfg.InputDim        = 3;
	cfg.HiddenDim       = 32;
	cfg.OutputDim       = 3;
	cfg.NumHiddenLayers = 1;

	HashEncoder::MiniMLP mlp(cfg);

	// Generate data
	const int N = 512;
	std::vector<float> inputs(N * 3);
	std::vector<float> targets(N * 3);
	for (int i = 0; i < N; i++) {
		float x = static_cast<float>(i) / static_cast<float>(N - 1);
		inputs[i * 3 + 0] = x;
		inputs[i * 3 + 1] = 0.5f;
		inputs[i * 3 + 2] = 0.25f;
		targets[i * 3 + 0] = 2.0f * x;
		targets[i * 3 + 1] = -x;
		targets[i * 3 + 2] = 1.0f - x;
	}

	// Manual training loop
	std::vector<std::vector<float>> activations;
	std::vector<float> featGrads(N * 3);
	float bestLoss = 1e10f;

	printf("Training MiniMLP for 2000 steps (lr=0.01/N)...\n");
	printf("  Step  | Loss\n");
	printf("  ------|--------\n");

	for (int step = 0; step < 2000; step++) {
		float loss = 0.0f;
		mlp.ZeroGradients();

		for (int n = 0; n < N; n++) {
			const float *in    = inputs.data() + n * 3;
			const float *tgt   = targets.data() + n * 3;
			float       *grad  = featGrads.data() + n * 3;

			std::vector<float> out(3);
			mlp.Forward(in, out.data(), activations);

			float diff0 = out[0] - tgt[0];
			float diff1 = out[1] - tgt[1];
			float diff2 = out[2] - tgt[2];
			loss += 0.5f * (diff0 * diff0 + diff1 * diff1 + diff2 * diff2);

			float lossGrad[3] = { diff0, diff1, diff2 };
			float inGrad[3];
			mlp.Backward(lossGrad, activations, inGrad);
		}
		loss /= static_cast<float>(N);
		mlp.UpdateParameters(0.01f / static_cast<float>(N));

		if (loss < bestLoss) bestLoss = loss;

		if (step % 200 == 0 || step == 1999) {
			printf("  %5d | %.6f\n", step, static_cast<double>(loss));
			if (std::isnan(loss)) {
				printf("  FAIL: NaN at step %d\n", step);
				return 1;
			}
		}
	}

	// Check final prediction
	std::vector<float> out(3);
	std::vector<std::vector<float>> act;
	float testIn[3] = {0.5f, 0.5f, 0.25f};
	mlp.Forward(testIn, out.data(), act);
	printf("\nFinal prediction for x=0.5: [%.4f, %.4f, %.4f]\n",
	       static_cast<double>(out[0]), static_cast<double>(out[1]), static_cast<double>(out[2]));
	printf("Expected:                [1.0000, -0.5000, 0.5000]\n");

	float err = std::abs(out[0] - 1.0f) + std::abs(out[1] + 0.5f) + std::abs(out[2] - 0.5f);
	printf("Total error: %.6f\n", static_cast<double>(err));
	printf("Best loss: %.6f\n", static_cast<double>(bestLoss));

	if (err < 0.5f && !std::isnan(bestLoss)) {
		printf("\nMiniMLP sanity test PASSED\n");
		return 0;
	} else {
		printf("\nMiniMLP sanity test FAILED\n");
		return 1;
	}
}
