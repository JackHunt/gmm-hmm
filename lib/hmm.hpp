#ifndef CONTINUOUS_HMM_HMM
#define CONTINUOUS_HMM_HMM

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "aliases.hpp"
#include "gmm.hpp"

namespace ContinuousHMM {

using GMM::GaussianMixtureModel;

template <typename T>
T log_sum_exp(const T& a, const T& b) {
  const auto log_a = a.array().log();
  const auto log_b = b.array().log();
  const auto log_sum = (log_a + log_b).exp();
  return log_sum.matrix();
}

/*!
 * This class implements a Hidden Markov Model with continuous
 * observations modelled by an $M$ dimensional Gaussian Mixture Model.
 *
 * HMM parameters are estimated within an Expectation Maximisation
 * framework.
 *
 */
template <size_t K, size_t M, size_t D>
class HiddenMarkovModel final {
 protected:
  // Number of observations (discrete times).
  const size_t T;

  // KX1 initial state probabilities.
  const Vector<K> initial_state_prob;

  // D dimensional observation vectors.
  const VectorList<K> observations;

  // Observation sequence probabilities. Alpha in much of the literature.
  VectorList<K> observation_prob;

  // Scale factors to avoid under/overflow.
  std::vector<double> scale_factors;

  // Emission vectors. Probability of observing state k from observation n.
  VectorList<K> emission_vectors;

  // Sequence probabilities. Beta in much of the literature.
  VectorList<K> sequence_probs;

  // KxK state transition matrix. A in much of the literature.
  Matrix<K, K> state_transition_prob;

  // Gaussian Mixture Model. In place of alphabet distribution in the discrete
  // case.
  std::vector<GaussianMixtureModel<D, M>> mixture_distributions;

 protected:
  void forward();
  void backward();
  void update();

  Vector<K> get_emission_vector(const Vector<D>& observation) const;
  MatrixList<K, K> compute_zetas() const;

  // EM M-Step subroutines.
  MatrixList<K, M> compute_gammas() const;
  std::vector<std::vector<double>> compute_mixture_coeffs(
      const MatrixList<K, M>& gammas) const;
  NestedVectorList<D> compute_gmm_means(const MatrixList<K, M>& gammas) const;
  NestedMatrixList<D, D> compute_gmm_covariances(
      const MatrixList<K, M>& gammas) const;

 public:
  HiddenMarkovModel(const Vector<K>& initial_state_prob,
                    const VectorList<D>& observations);

  void train(unsigned int epochs, double convergence_threshold);

  std::pair<std::vector<size_t>, VectorList<K>> get_state_sequence();
};

}  // namespace ContinuousHMM

#endif