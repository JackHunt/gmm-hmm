#ifndef CONTINUOUS_HMM_MVN
#define CONTINUOUS_HMM_MVN

#include "aliases.hpp"

namespace HMM::ProbUtils {

/*!
 * This class implements a simple Multivariate Normal (Gaussian) Distribution.
 *
 */
template <size_t D>
class MultivariateNormal final {
 protected:
  Vector<D> mu;
  Matrix<D, D> sigma;

 public:
  MultivariateNormal(const Vector<D>& mu, const Matrix<D, D>& sigma);

  double operator()(const Vector<D>& x) const;

  const Vector<D>& get_mu() const;
  void set_mu(const Vector<D>& mu);

  const Matrix<D, D>& get_sigma() const;
  void set_sigma(const Matrix<D, D>& sigma);
};

}  // namespace HMM::ProbUtils

#endif