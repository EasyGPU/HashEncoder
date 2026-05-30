#pragma once

/**
 * @file NeuralHashGrid.h
 * @brief Mini MLP + HashGridEncoding integration for end-to-end training.
 *
 * Demonstrates the canonical Instant-NGP pattern:
 *   position → hash encode → features → MLP → output
 *
 * The MLP is CPU-only for simplicity; the hash encoding can use GPU
 * via HashGridEncoding::EncodeGPU / EncodeGPUBackward.
 */

#include <HashEncoder/HashEncoder.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace HashEncoder {

// ============================================================================
// MiniMLP — tiny CPU multilayer perceptron
// ============================================================================

class MiniMLP {
public:
	struct Config {
		int  InputDim        = 16;
		int  HiddenDim       = 64;
		int  OutputDim       = 3;
		int  NumHiddenLayers = 2;
	};

	MiniMLP() = default;

	explicit MiniMLP(const Config &cfg) {
		Init(cfg);
	}

	void Init(const Config &cfg) {
		cfg_ = cfg;
		std::mt19937 rng(42);

		auto xavier = [&](int fanIn, int fanOut) {
			float limit = std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));
			std::uniform_real_distribution<float> dist(-limit, limit);
			return dist(rng);
		};

		// Build layer list: input → hidden × N → output
		layerSizes_.clear();
		layerSizes_.push_back(cfg.InputDim);
		for (int i = 0; i < cfg.NumHiddenLayers; i++) {
			layerSizes_.push_back(cfg.HiddenDim);
		}
		layerSizes_.push_back(cfg.OutputDim);

		int numLayers = static_cast<int>(layerSizes_.size()) - 1;
		weights_.resize(numLayers);
		biases_.resize(numLayers);
		gradW_.resize(numLayers);
		gradB_.resize(numLayers);

		for (int l = 0; l < numLayers; l++) {
			int fanIn  = layerSizes_[l];
			int fanOut = layerSizes_[l + 1];
			weights_[l].resize(fanIn * fanOut);
			biases_[l].resize(fanOut, 0.0f);
			gradW_[l].resize(fanIn * fanOut, 0.0f);
			gradB_[l].resize(fanOut, 0.0f);
			for (int i = 0; i < fanIn * fanOut; i++) {
				weights_[l][i] = xavier(fanIn, fanOut);
			}
		}
	}

	/// Forward pass: input → output. Returns layer activations needed for backward.
	/// activations[0] = input, activations[L] = output of layer L (after activation)
	void Forward(const float *input, float *output,
				 std::vector<std::vector<float>> &activations) const {
		int numLayers = static_cast<int>(weights_.size());
		activations.resize(numLayers + 1);

		// Layer 0 activation = input
		int inputDim = layerSizes_[0];
		activations[0].assign(input, input + inputDim);

		const float *act = activations[0].data();
		for (int l = 0; l < numLayers; l++) {
			int fanIn  = layerSizes_[l];
			int fanOut = layerSizes_[l + 1];
			bool isLast = (l == numLayers - 1);

			activations[l + 1].resize(fanOut);
			float *out = activations[l + 1].data();
			const float *W = weights_[l].data();
			const float *B = biases_[l].data();

			for (int o = 0; o < fanOut; o++) {
				float sum = B[o];
				for (int i = 0; i < fanIn; i++) {
					sum += act[i] * W[i * fanOut + o];
				}
				// No activation on output layer
				out[o] = isLast ? sum : std::max(0.0f, sum);
			}
			act = out;
		}

		// Copy final layer to output
		int outDim = layerSizes_.back();
		std::copy(activations.back().data(), activations.back().data() + outDim, output);
	}

	/// Backward pass. Given loss gradient w.r.t. output, computes gradients
	/// w.r.t. weights/biases and returns gradient w.r.t. input.
	/// The computed weight/bias gradients are accumulated into gradW_/gradB_.
	void Backward(const float *lossGrad,
				  const std::vector<std::vector<float>> &activations,
				  float *inputGrad) {
		int numLayers = static_cast<int>(weights_.size());
		std::vector<float> delta;

		for (int l = numLayers - 1; l >= 0; l--) {
			int fanIn  = layerSizes_[l];
			int fanOut = layerSizes_[l + 1];
			bool isLast = (l == numLayers - 1);

			// First iteration: delta = lossGrad
			// Subsequent: delta computed from next layer
			if (l == numLayers - 1) {
				delta.assign(lossGrad, lossGrad + fanOut);
			}

			const float *act = activations[l].data();
			const float *W   = weights_[l].data();
			float       *dW  = gradW_[l].data();
			float       *dB  = gradB_[l].data();

			// Gradient w.r.t. input of this layer
			std::vector<float> deltaPrev(fanIn, 0.0f);

			for (int o = 0; o < fanOut; o++) {
				float d = delta[o];

				// ReLU backward: gradient is zero where activation <= 0
				if (!isLast && activations[l + 1][o] <= 0.0f) {
					d = 0.0f;
				}

				dB[o] += d;
				for (int i = 0; i < fanIn; i++) {
					dW[i * fanOut + o] += d * act[i];
					deltaPrev[i] += d * W[i * fanOut + o];
				}
			}

			delta = std::move(deltaPrev);
		}

		int inputDim = layerSizes_[0];
		std::copy(delta.data(), delta.data() + inputDim, inputGrad);
	}

	/// Zero accumulated gradients.
	void ZeroGradients() {
		for (size_t l = 0; l < gradW_.size(); l++) {
			std::fill(gradW_[l].begin(), gradW_[l].end(), 0.0f);
			std::fill(gradB_[l].begin(), gradB_[l].end(), 0.0f);
		}
	}

	/// SGD update.
	void UpdateParameters(float lr) {
		for (size_t l = 0; l < weights_.size(); l++) {
			for (size_t i = 0; i < weights_[l].size(); i++) {
				weights_[l][i] -= lr * gradW_[l][i];
			}
			for (size_t i = 0; i < biases_[l].size(); i++) {
				biases_[l][i] -= lr * gradB_[l][i];
			}
		}
	}

	int InputDim() const { return layerSizes_.empty() ? 0 : layerSizes_[0]; }
	int OutputDim() const { return layerSizes_.empty() ? 0 : layerSizes_.back(); }

private:
	Config                    cfg_;
	std::vector<int>          layerSizes_;
	std::vector<std::vector<float>> weights_;
	std::vector<std::vector<float>> biases_;
	std::vector<std::vector<float>> gradW_;
	std::vector<std::vector<float>> gradB_;
};

// ============================================================================
// NeuralHashGrid — HashGridEncoding + MiniMLP
// ============================================================================

template <int Dim>
class NeuralHashGrid {
public:
	using Encoding = HashGridEncoding<Dim>;

	/// Build from configs.
	NeuralHashGrid(const HashGridConfig &encCfg, const MiniMLP::Config &mlpCfg)
		: encoder_(encCfg), mlp_(mlpCfg) {
		if (mlpCfg.InputDim != encoder_.GetOutputDims()) {
			throw std::invalid_argument(
				"MLP InputDim must match encoder output dims");
		}
	}

	void Initialize(uint64_t seed) {
		encoder_.Initialize(seed);
		// Re-init MLP with the same seed
		MiniMLP::Config mlpCfg;
		mlpCfg.InputDim        = encoder_.GetOutputDims();
		mlpCfg.HiddenDim       = mlp_.OutputDim() > 0 ? 64 : mlpCfg.HiddenDim;  // preserve or default
		// Actually just keep current MLP config — the constructor already set it up
		(void)seed;
	}

	/// Forward pass: positions → features → MLP → output.
	void Forward(const float *positions, float *output, int count) const {
		int featDim = encoder_.GetOutputDims();
		std::vector<float> features(static_cast<size_t>(count) * featDim);
		encoder_.EncodeCPU(positions, features.data(), count);

		std::vector<std::vector<float>> activations;
		for (int n = 0; n < count; n++) {
			float *feat   = features.data() + static_cast<size_t>(n) * featDim;
			float *out    = output + static_cast<size_t>(n) * mlp_.OutputDim();
			mlp_.Forward(feat, out, activations);
		}
	}

	/// Backward pass: loss gradient w.r.t. MLP output → feature gradients
	/// → table gradients. Also accumulates MLP parameter gradients.
	/// @param positions      [count * Dim] input positions
	/// @param lossGrad       [count * mlpOutputDim] dLoss/dOutput
	/// @param tableGrads     [encoder hash table size] accumulated table gradients
	/// @param posGrads       [count * Dim] position gradients (may be null)
	/// @param count          number of positions
	void Backward(const float *positions, const float *lossGrad,
				  float *tableGrads, float *posGrads, int count) {
		int featDim = encoder_.GetOutputDims();
		int outDim  = mlp_.OutputDim();

		std::vector<float> features(static_cast<size_t>(count) * featDim);
		encoder_.EncodeCPU(positions, features.data(), count);

		std::vector<float> featGrads(static_cast<size_t>(count) * featDim, 0.0f);
		std::vector<std::vector<float>> activations;

		for (int n = 0; n < count; n++) {
			const float *feat = features.data() + static_cast<size_t>(n) * featDim;
			const float *grad = lossGrad + static_cast<size_t>(n) * outDim;

			// Forward to get activations for this sample
			std::vector<float> dummyOut(outDim);
			mlp_.Forward(feat, dummyOut.data(), activations);

			// Backward through MLP to get feature gradient
			float *featGrad = featGrads.data() + static_cast<size_t>(n) * featDim;
			mlp_.Backward(grad, activations, featGrad);
		}

		// Backward through encoder: feature grads → table grads + pos grads
		encoder_.EncodeCPUBackward(positions, featGrads.data(), tableGrads,
								   posGrads, count);
	}

	/// MSE loss gradient: d/doutput (0.5 * ||output - target||^2) = output - target
	static void MSELossGrad(const float *output, const float *target, float *grad, int count, int dim) {
		for (int n = 0; n < count; n++) {
			for (int d = 0; d < dim; d++) {
				int idx    = n * dim + d;
				grad[idx]  = output[idx] - target[idx];
			}
		}
	}

	/// One training step: forward → loss → backward → update.
	/// Returns the MSE loss before the update.
	float TrainStep(const float *positions, const float *targets, int count, float lr) {
		int outDim  = mlp_.OutputDim();
		int featDim = encoder_.GetOutputDims();
		size_t tableSize = encoder_.GetHashTableSize();

		std::vector<float> output(static_cast<size_t>(count) * outDim);
		Forward(positions, output.data(), count);

		// MSE loss (averaged over batch)
		float loss = 0.0f;
		for (int i = 0; i < count * outDim; i++) {
			float diff = output[i] - targets[i];
			loss += 0.5f * diff * diff;
		}
		loss /= static_cast<float>(count);

		// Loss gradient: d/doutput of un-scaled MSE = output - target
		std::vector<float> lossGrad(static_cast<size_t>(count) * outDim);
		MSELossGrad(output.data(), targets, lossGrad.data(), count, outDim);

		// Zero gradients
		mlp_.ZeroGradients();
		std::vector<float> tableGrads(tableSize, 0.0f);

		// Backward — accumulates gradients over all samples
		std::vector<float> features(static_cast<size_t>(count) * featDim);
		encoder_.EncodeCPU(positions, features.data(), count);

		std::vector<float> featGrads(static_cast<size_t>(count) * featDim, 0.0f);
		std::vector<std::vector<float>> activations;

		for (int n = 0; n < count; n++) {
			const float *feat = features.data() + static_cast<size_t>(n) * featDim;
			const float *grad = lossGrad.data() + static_cast<size_t>(n) * outDim;
			std::vector<float> dummyOut(outDim);
			mlp_.Forward(feat, dummyOut.data(), activations);
			float *featGrad = featGrads.data() + static_cast<size_t>(n) * featDim;
			mlp_.Backward(grad, activations, featGrad);
		}

		encoder_.EncodeCPUBackward(positions, featGrads.data(), tableGrads.data(),
		                           nullptr, count);

		// Update parameters — gradients are summed over batch, so scale LR by 1/count
		float effectiveLR = lr / static_cast<float>(count);
		mlp_.UpdateParameters(effectiveLR);
		{
			std::vector<float> updatedTable = encoder_.GetHashTable();
			for (size_t i = 0; i < tableSize; i++) {
				updatedTable[i] -= effectiveLR * tableGrads[i];
			}
			encoder_.SetHashTable(updatedTable);
		}

		return loss;
	}

	// Accessors
	Encoding       &GetEncoder() { return encoder_; }
	const Encoding &GetEncoder() const { return encoder_; }
	MiniMLP        &GetMLP() { return mlp_; }
	const MiniMLP  &GetMLP() const { return mlp_; }

private:
	Encoding encoder_;
	MiniMLP  mlp_;
};

} // namespace HashEncoder
