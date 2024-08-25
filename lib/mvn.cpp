#include "mvn.hpp"

using namespace HMM::ProbUtils;

/*!
 *
 *
 * \param mu Mean vector $\bm{\mu} \in \mathbb{R}^{D}$
 * \param sigma Covariance matrix $\bm{\Sigma} \in \mathbb{R}^{D \times D}$
 */
template <size_t D>
MultivariateNormal<D>::MultivariateNormal(const Vector<D>& mu,
                                          const Matrix<D, D>& sigma)
    : mu(mu), sigma(sigma) {
  //
}

/*!
 * Evaluate PDF.
 *
 * \param x $D$ dimensional vector.
 * \return $P(\bm{x} | \bm{\mu}, \bm{\Sigma}$
 */
template <size_t D>
double MultivariateNormal<D>::operator()(const Vector<D>& x) const {
  // Compute normaliser.
  const auto z =
      std::pow(2.0 * M_PI, x.size() / 2.0) * std::sqrt(sigma.determinant());

  // Compute exponential part.
  const auto d = x - mu;
  const double quad = -0.5 * d.transpose() * sigma.inverse() * d;

  return (1.0 / z) * std::exp(quad);
}

/*!
 * Returns the mean vector of the density.
 *
 * \return $\bm{\mu} \in \mathbb{R}^{D}$
 */
template <size_t D>
const Vector<D>& MultivariateNormal<D>::get_mu() const {
  return mu;
}

/*!
 * Sets the densities mean vector.
 *
 * NOTE: Will fail an assert upon a dimension mismatch.
 *
 * \param mu $\bm{\mu} \in \mathbb{R}^{D}$
 */
template <size_t D>
void MultivariateNormal<D>::set_mu(const Vector<D>& mu) {
  this->mu = mu;
}

/*!
 * Returns the covariance matrix of the density.
 *
 * \return $\bm{\Sigma} \in \mathbb{R}^{D \times D}
 */
template <size_t D>
const Matrix<D, D>& MultivariateNormal<D>::get_sigma() const {
  return sigma;
}

/*!
 * Sets the densities covariance matrix.
 *
 * NOTE: Will fail an assert upon a dimension mismatch.
 *
 * \param sigma $\bm{\Sigma} \in \mathbb{R}^{D \times D}
 */
template <size_t D>
void MultivariateNormal<D>::set_sigma(const Matrix<D, D>& sigma) {
  this->sigma = sigma;
}

namespace HMM::ProbUtils {
template class MultivariateNormal<3>;
}