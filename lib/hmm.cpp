#include "hmm.hpp"

using namespace ContinuousHMM;

/*!
 * Continuous observation Hidden Markov Model.
 *
 * \param initial_state_prob Initial system state probability distribution. $K$
 * dimensional vector.
 * \param observations $N$ observations of dimensionality $D$.
 * \param M Number of GMM mixture components.
 */
template <size_t K, size_t M, size_t D>
HiddenMarkovModel<K, M, D>::HiddenMarkovModel(
    const Vector<K>& initial_state_prob, const VectorList<D>& observations)
    : T(observations.size()),
      initial_state_prob(initial_state_prob),
      observations(observations),
      observation_prob(observations.size()),
      scale_factors(observations.size()),
      emission_vectors(observations.size()),
      sequence_probs(observations.size()),
      state_transition_prob(Matrix<K, K>::Identity()),
      mixture_distributions(K, GaussianMixtureModel<D, M>()) {
  if (T == 0) {
    throw std::runtime_error("No observations provided.");
  }

  // Initialise emission vector, observation probability and scale factor for
  // t=0.
  emission_vectors.front() = get_emission_vector(observations.front());
  observation_prob.front() =
      log_sum_exp<Vector<K>>(initial_state_prob, emission_vectors.front());
  scale_factors.front() = 1.0 / observation_prob.front().sum();

  // Scale observation probabilities (alpha) for t=0.
  observation_prob.front() *= scale_factors.front();

  // Initialise state probability for t=T. Uninformative prior.
  sequence_probs.back() = Vector<K>::Ones() / K;
}

/*!
 * Performs the forward pass of the Forward-Backward algorithm.
 *
 */
template <size_t K, size_t M, size_t D>
void HiddenMarkovModel<K, M, D>::forward() {
  for (size_t i = 1; i < observations.size(); i++) {
    // Update emission vector.
    emission_vectors.at(i) = get_emission_vector(observations.at(i));

    // Compute alpha.
    const auto p = log_sum_exp_matmul<K, K, 1>(
        state_transition_prob.transpose(), observation_prob.at(i - 1));
    const auto alpha = log_sum_exp<Vector<K>>(p, emission_vectors.at(i));

    // Compute scale factor.
    auto eps = [](double a) -> bool {
      return std::abs(a) < std::numeric_limits<double>::epsilon();
    };
    const auto alpha_sum = alpha.sum();
    const auto scale_factor = eps(alpha_sum) ? 1.0 : 1.0 / alpha.sum();

    // Update.
    observation_prob.at(i) = alpha * scale_factor;
    scale_factors.at(i) = scale_factor;
  }
}

/*!
 * Performs the backward pass of the Forward-Backward algorithm.
 *
 */
template <size_t K, size_t M, size_t D>
void HiddenMarkovModel<K, M, D>::backward() {
  for (int i = (observations.size() - 2); i >= 0; i--) {
    const auto p = log_sum_exp<Vector<K>>(emission_vectors.at(i + 1),
                                          sequence_probs.at(i + 1));
    const auto beta = state_transition_prob.transpose() * p;
    sequence_probs.at(i) = beta * scale_factors.at(i);
  }
}

/*!
 * Updates model parameters for a single EM iteration.
 *
 */
template <size_t K, size_t M, size_t D>
void HiddenMarkovModel<K, M, D>::update() {
  /*
   * E Step.
   */
  const auto gammas = compute_gammas();

  // Expected value of first state.
  const auto expectation_pi = gammas.front().rowwise().sum();
  observation_prob.front() = expectation_pi;

  // Sum gamma rows, over time.
  VectorList<K> gamma_row_sums;
  gamma_row_sums.reserve(observations.size());
  std::transform(gammas.cbegin(), gammas.cend(),
                 std::back_inserter(gamma_row_sums),
                 [](const auto& gamma) { return gamma.rowwise().sum(); });

  const auto gamma_row_sum_t = std::accumulate(
      gamma_row_sums.cbegin(), gamma_row_sums.cend(), Vector<K>::Zero().eval());

  /*
   * M Step.
   */
  // Expected value of state transition matrix.
  const auto zetas = compute_zetas();
  auto expectation_A = std::accumulate(zetas.cbegin(), zetas.cend(),
                                       Matrix<K, K>::Zero().eval());
  // Normalise. TODO: Vectorise this.
  for (size_t k = 0; k < K; k++) {
    expectation_A.row(k) /= gamma_row_sum_t(k);
    expectation_A.row(k).normalize();
  }
  state_transition_prob = expectation_A;

  // Updated mixture coefficients
  const auto updated_coefficients = compute_mixture_coeffs(gammas);

  // Updated mixture means and covariances.
  const auto updated_means = compute_gmm_means(gammas);
  const auto updated_covariances = compute_gmm_covariances(gammas);

  // Update GMM parameters.
  for (size_t k = 0; k < K; k++) {
    mixture_distributions.at(k).set_mixture_coefficients(
        updated_coefficients.at(k));
    mixture_distributions.at(k).set_means(updated_means.at(k));
    mixture_distributions.at(k).set_covariances(updated_covariances.at(k));
  }
}

/*!
 * Compute the emission vector w.r.t. the GMM for a given
 * observation.
 *
 * \param observation $D$ dimensional observation vector.
 * \return $P(x | \bm{\mu}, \bm{\Sigma})$ - note: P(.) denotes the GMM PDF.
 */
template <size_t K, size_t M, size_t D>
Vector<K> HiddenMarkovModel<K, M, D>::get_emission_vector(
    const Vector<D>& observation) const {
  Vector<K> p;
  for (size_t k = 0; k < K; k++) {
    p(k) = mixture_distributions.at(k)(observation);
  }
  return p;
}

/*!
 * Compute the Expectation of the state transition matrix $\bm{A}$.
 *
 * \return Expectations over times $t$, $t+1$ etc
 */
template <size_t K, size_t M, size_t D>
MatrixList<K, K> HiddenMarkovModel<K, M, D>::compute_zetas() const {
  // Normalizer.
  const auto Z = observation_prob.back().sum();

  // State transitions.
  MatrixList<K, K> zetas;
  zetas.reserve(observations.size());
  for (size_t i = 0; i < observations.size(); i++) {
    const auto tmp =
        log_sum_exp<Vector<K>>(emission_vectors.at(i), sequence_probs.at(i))
            .transpose();
    const auto p = observation_prob.at(i) * tmp;
    zetas.push_back(log_sum_exp<Matrix<K, K>>(state_transition_prob, p) / Z);
  }

  return zetas;
}

/*!
 * Compute emission matrices $\bm{\gamma}_{t} \in \mathbb{R}^{K \times M}$
 * where $K$ is the number of states in the system and $M$ is the number
 * of mixture densities in the GMM.
 *
 * \return $\bm{\gamma}_{t} \forall t$.
 */
template <size_t K, size_t M, size_t D>
MatrixList<K, M> HiddenMarkovModel<K, M, D>::compute_gammas() const {
  MatrixList<K, M> gammas;
  gammas.reserve(observations.size());

  for (size_t t = 0; t < observations.size(); t++) {
    const auto& x = observations.at(t);

    auto w =
        log_sum_exp<Vector<K>>(observation_prob.at(t), sequence_probs.at(t));
    w /= w.sum();

    Matrix<K, M> gamma(K, M);
    for (size_t k = 0; k < K; k++) {
      const auto& p = mixture_distributions.at(k);

      const auto prob = p(x);
      for (size_t m = 0; m < M; m++) {
        gamma(k, m) = p(x, m) / prob;
        gamma.row(k) *= w(k);
      }
    }
    gammas.push_back(gamma);
  }

  return gammas;
}

/*!
 * Compute Expectation over GMM mixture coefficients.
 *
 * \param gammas $\bm{\gamma}_{t} \forall t$ - as computed by
 * compute_gmm_gammas().
 * \return $M$ coefficients for each state $K$.
 */
template <size_t K, size_t M, size_t D>
std::vector<std::vector<double>>
HiddenMarkovModel<K, M, D>::compute_mixture_coeffs(
    const MatrixList<K, M>& gammas) const {
  // Sum row for each state.
  VectorList<K> state_sums;
  state_sums.reserve(gammas.size());
  std::transform(gammas.cbegin(), gammas.cend(), std::back_inserter(state_sums),
                 [](const auto& g) { return g.rowwise().sum(); });

  // Sum the above state_sums over time to get normaliser.
  const auto Z = std::accumulate(state_sums.cbegin(), state_sums.cend(),
                                 Vector<K>::Zero().eval());

  // Sum gammas over all time.
  const auto gamma_sum = std::accumulate(gammas.cbegin(), gammas.cend(),
                                         Matrix<K, M>::Zero().eval());

  // Final mixture coefficients.
  std::vector<std::vector<double>> mixture_coeffs;
  mixture_coeffs.reserve(K);

  for (size_t k = 0; k < K; k++) {
    // Compute normalised coefficients for state k.
    auto state_coeffs = gamma_sum.row(k).eval();
    state_coeffs /= Z(k);

    mixture_coeffs.emplace_back(state_coeffs.data(),
                                state_coeffs.data() + state_coeffs.size());
  }

  return mixture_coeffs;
}

/*!
 * Compute Expectation over GMM mixture means.
 *
 * \param gammas $\bm{\gamma}_{t} \forall t$ - as computed by
 * compute_gmm_gammas().
 * \return $M$ mean vectors, $\bm{\mu}_{m}$ for each state
 * $K$.
 */
template <size_t K, size_t M, size_t D>
NestedVectorList<D> HiddenMarkovModel<K, M, D>::compute_gmm_means(
    const MatrixList<K, M>& gammas) const {
  // Output.
  NestedVectorList<D> updated_means(K, VectorList<D>(M, Vector<D>::Zero()));

  // Compute unnormalised mean vectors.
  for (size_t t = 0; t < observations.size(); t++) {
    const auto& x = observations.at(t);
    const auto& gamma = gammas.at(t);
    for (size_t k = 0; k < K; k++) {
      for (size_t m = 0; m < M; m++) {
        updated_means.at(k).at(m) += gamma(k, m) * x;
      }
    }
  }

  // Sum gammas over all time. Normaliser.
  const auto Z = std::accumulate(gammas.cbegin(), gammas.cend(),
                                 Matrix<K, M>::Zero().eval());

  // Normalise.
  for (size_t k = 0; k < K; k++) {
    for (size_t m = 0; m < M; m++) {
      updated_means.at(k).at(m) /= Z(k, m);
    }
  }

  return updated_means;
}

/*!
 * Compute Expectation over GMM mixture covariance matrices.
 *
 * \param gammas $\bm{\gamma}_{t} \forall t$ - as computed by
 * compute_gmm_gammas().
 * \return $M$ covariance matrices, $\bm{\Sigma}_{m} \in
 * \mathbb{R}^{D \times D}$ for each state $K$.
 */
template <size_t K, size_t M, size_t D>
NestedMatrixList<D, D> HiddenMarkovModel<K, M, D>::compute_gmm_covariances(
    const MatrixList<K, M>& gammas) const {
  // Output.
  NestedMatrixList<D, D> updated_covariances(
      K, MatrixList<D, D>(M, Matrix<D, D>::Zero()));

  // Compute unnormalised covariances.
  for (size_t t = 0; t < observations.size(); t++) {
    const auto& x = observations.at(t);
    const auto& gamma = gammas.at(t);
    for (size_t k = 0; k < K; k++) {
      for (size_t m = 0; m < M; m++) {
        const auto mu = mixture_distributions.at(k).get_means().at(m);
        const auto sigma = mixture_distributions.at(k).get_covariances().at(m);
        updated_covariances.at(k).at(m) +=
            gamma(k, m) * (x - mu) * (x - mu).transpose();
      }
    }
  }

  // Sum gammas over all time. Normaliser.
  const auto Z = std::accumulate(gammas.cbegin(), gammas.cend(),
                                 Matrix<K, M>::Zero().eval());

  // Normalise and make symmetric,  positive semidefinite.
  // I.E. a valid covariance matrix.
  for (size_t k = 0; k < K; k++) {
    for (size_t m = 0; m < M; m++) {
      // Normalise.
      updated_covariances.at(k).at(m) /= Z(k, m);

      // Make symmetric, positive semidefinite.
      const auto diag =
          updated_covariances.at(k).at(m) * Matrix<D, D>::Identity();
      Matrix<D, D> lower_triangular =
          updated_covariances.at(k)
              .at(m)
              .template triangularView<Eigen::Lower>();
      updated_covariances.at(k).at(m) =
          lower_triangular + lower_triangular.transpose();
      updated_covariances.at(k).at(m) -= diag;
      updated_covariances.at(k).at(m) += 0.1 * Matrix<D, D>::Identity();
      // ***Should now be symmetric, positive semidefinite***
    }
  }

  return updated_covariances;
}

/*!
 * Train the HMM on observation data within the Expectation Maximisation
 * framework.
 *
 * \param epochs Maximum number of training epochs.
 * \param convergence_threshold Training termination threshold.
 */
template <size_t K, size_t M, size_t D>
void HiddenMarkovModel<K, M, D>::train(unsigned int epochs,
                                       double convergence_threshold) {
  unsigned int epoch = 0;
  auto log_likelihood = std::numeric_limits<double>::min();
  auto prev_log_likelihood = log_likelihood;

  for (unsigned int epoch = 0; epoch < epochs; epoch++) {
    // Forwards-Backwards.
    forward();
    backward();

    // Compute log likelihood.
    prev_log_likelihood = log_likelihood;
    log_likelihood =
        -1.0 *
        std::accumulate(scale_factors.cbegin(), scale_factors.cend(), 0.0,
                        [](double acc, double c) { return acc + std::log(c); });
    log_likelihood /= observations.size();
    std::cout << "Epoch: " << epoch << " Log-Likelihood: " << log_likelihood
              << std::endl;

    // Test for convergence.
    if (std::abs(log_likelihood - prev_log_likelihood) <
        convergence_threshold) {
      std::cout << "Model has converged. Done!" << std::endl;
      break;
    }

    // Expectation Maximisation.
    update();
  }
}

/*!
 * Computes the most likely state sequence of the underlying stochastic process.
 * Viterbi algorithm.
 *
 * \return Most probable states and the state distributions for all $t$.
 */
template <size_t K, size_t M, size_t D>
std::pair<std::vector<size_t>, VectorList<K>>
HiddenMarkovModel<K, M, D>::get_state_sequence() {
  // State probabilities.
  VectorList<K> delta;
  delta.reserve(observations.size());

  // State id's with max probability.
  std::vector<size_t> psi;
  psi.reserve(observations.size());

  // Initial state.
  delta.push_back(log_sum_exp<Vector<K>>(observation_prob.front(),
                                         emission_vectors.front()));
  psi.push_back(0);

  // Compute the Viterbi Path; the most likely state sequence.
  for (size_t t = 1; t < observations.size(); t++) {
    const auto trans_prob = delta.at(t - 1).transpose() * state_transition_prob;

    typename Vector<K>::Index max_idx;
    delta.push_back(trans_prob.maxCoeff(&max_idx) * emission_vectors.at(t));
    psi.push_back(static_cast<size_t>(max_idx));
  }

  // Predicted states and state distributions.
  return {psi, delta};
}

namespace ContinuousHMM {
template class HiddenMarkovModel<3, 4, 3>;
}