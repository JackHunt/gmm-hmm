#include "gmm.hpp"

using namespace HMM::GMM;

/*!
 * Gaussian Mixture Model.
 *
 * \param M Number of mixture distributions.
 * \param D Dimensionality of mixture distributions.
 */
template <size_t D, size_t M>
GaussianMixtureModel<D, M>::GaussianMixtureModel() {
  // Initialise GMM Mixture Distributions to a Mean of 0, and Covariance of I.
  const auto mu = Vector<D>::Zero();
  const auto sigma = Matrix<D, D>::Identity();
  const auto c = 1.0 / static_cast<double>(M);
  for (size_t m = 0; m < M; m++) {
    mixture_distributions.emplace_back(c, Normal<D>(mu, sigma));
  }
}

/*!
 * Evaluate the GMM PDF at some point $x$.
 *
 * \param x D dimensional vector.
 * \return $P(x | \bm{\mu}, \bm{\Sigma}$ - a scalar.
 */
template <size_t D, size_t M>
double GaussianMixtureModel<D, M>::operator()(const Vector<D>& x) const {
  double prob = 0.0;
  for (const auto& [c, p] : mixture_distributions) {
    prob += c * p(x);
  }

  // Clamp value at extremum.
  if (prob < 0.0 || prob > 1.0) {
    prob = (prob < 0.0) ? 0.0 : 1.0;
  }

  return prob;
}

/*!
 * Evaluate a component distribution at some point $x$.
 *
 * \param x D dimensional vector.
 * \param m ID of the component distribution to evaluate.
 * \param weight If true, multiply by mixture coefficient $c_{m}$
 * \return Scalar PDF value for distribution $m$.
 */
template <size_t D, size_t M>
double GaussianMixtureModel<D, M>::operator()(const Vector<D>& x, size_t m,
                                              bool weight) const {
  if (m >= M) {
    throw std::invalid_argument("Mixture index out of bounds!");
  }

  const auto& dist = mixture_distributions.at(m);

  auto p = dist.second(x);
  if (weight) {
    p *= dist.first;
  }

  return p;
}

/*!
 * Get $\bm{\mu}_{m}$ for each component distribution $m$.
 *
 * \return Mean vector of each component distribution.
 */
template <size_t D, size_t M>
VectorList<D> GaussianMixtureModel<D, M>::get_means() const {
  VectorList<D> means;
  means.reserve(mixture_distributions.size());

  for (const auto& d : mixture_distributions) {
    means.push_back(d.second.get_mu());
  }

  return means;
}

/*!
 * Get $\bm{\Sigma}_{m}$ for each component distribution $m$.
 *
 * \return Covariance matrix of each component distribution.
 */
template <size_t D, size_t M>
MatrixList<D, D> GaussianMixtureModel<D, M>::get_covariances() const {
  MatrixList<D, D> covariances;
  covariances.reserve(mixture_distributions.size());

  for (const auto& d : mixture_distributions) {
    covariances.push_back(d.second.get_sigma());
  }

  return covariances;
}

/*!
 * Get $c_{m}$ for each component distribution $m$.
 *
 * \return Mixture coefficient of each component distribution.
 */
template <size_t D, size_t M>
std::vector<double> GaussianMixtureModel<D, M>::get_mixture_coefficients()
    const {
  std::vector<double> coeffs;
  coeffs.reserve(mixture_distributions.size());

  std::transform(mixture_distributions.cbegin(), mixture_distributions.cend(),
                 std::back_inserter(coeffs),
                 [](const auto& d) { return d.first; });

  return coeffs;
}

/*!
 * Set the component distribution mean vectors.
 *
 * NOTE: Will fail and assert upon a dimension mismatch.
 *
 * \param means $M$ vectors of dimensionality $D$.
 */
template <size_t D, size_t M>
void GaussianMixtureModel<D, M>::set_means(const VectorList<D>& means) {
  for (size_t i = 0; i < means.size(); i++) {
    const auto& mu = mixture_distributions.at(i).second.get_mu();
    mixture_distributions.at(i).second.set_mu(means.at(i));
  }
}

/*!
 * Set the component distribution covariance matrices.
 *
 * NOTE: Will fail and assert upon a dimension mismatch.
 *
 * \param covariances $M$ matrices of dimensionality $D \times D$.
 */
template <size_t D, size_t M>
void GaussianMixtureModel<D, M>::set_covariances(
    const MatrixList<D, D>& covariances) {
  for (size_t i = 0; i < covariances.size(); i++) {
    const auto& sigma = mixture_distributions.at(i).second.get_sigma();
    mixture_distributions.at(i).second.set_sigma(covariances.at(i));
  }
}

/*!
 * Set the component distribution mixture coefficients.
 *
 * \param coeffs $M$ scalars which sum to 1.
 */
template <size_t D, size_t M>
void GaussianMixtureModel<D, M>::set_mixture_coefficients(
    const std::vector<double>& coeffs) {
  auto valid_dist = [](double sum) -> bool {
    return std::abs(1.0 - sum) < std::numeric_limits<double>::epsilon();
  };

  auto coefficients = coeffs;

  auto sum = std::accumulate(coeffs.cbegin(), coeffs.cend(), 0.0);

  // If coefficients do not sum to one, normalise.
  if (!valid_dist(sum)) {
    std::vector<double> normalised_coeffs;
    normalised_coeffs.reserve(coeffs.size());
    std::transform(coeffs.cbegin(), coeffs.cend(),
                   std::back_inserter(normalised_coeffs),
                   [sum](double p) -> double { return p / sum; });
    coefficients = normalised_coeffs;
  }

  for (size_t i = 0; i < mixture_distributions.size(); i++) {
    mixture_distributions.at(i).first = coeffs.at(i);
  }
}

/*!
 * Returns the number of mixture components, $M$.
 *
 * \return # mixture components.
 */
template <size_t D, size_t M>
constexpr size_t GaussianMixtureModel<D, M>::get_num_mixture_components()
    const {
  return M;
}

/*!
 * Returns the mixture component dimensionality, $D$.
 *
 * \return distribution dimensionality.
 */
template <size_t D, size_t M>
constexpr size_t GaussianMixtureModel<D, M>::get_data_dimension() const {
  return D;
}

namespace HMM::GMM {
template class GaussianMixtureModel<3, 4>;
}