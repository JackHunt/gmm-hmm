#ifndef CONTINUOUS_HMM_GMM
#define CONTINUOUS_HMM_GMM

#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "aliases.hpp"
#include "mvn.hpp"

namespace ContinuousHMM::GMM {

template <size_t D>
using Normal = ProbUtils::MultivariateNormal<D>;

template <size_t D>
using MixtureDistribution = std::pair<double, Normal<D>>;

/*!
 * This class implements a basic Gaussian Mixture Model, comprised
 * of $M$ Multivariate Normal distributions of dimensionality $D$.
 */
template <size_t D, size_t M>
class GaussianMixtureModel final {
 protected:
  std::vector<MixtureDistribution<D>> mixture_distributions;

 public:
  GaussianMixtureModel();

  double operator()(const Vector<D>& x) const;
  double operator()(const Vector<D>& x, size_t m, bool weight = true) const;

  VectorList<D> get_means() const;
  MatrixList<D, D> get_covariances() const;
  std::vector<double> get_mixture_coefficients() const;

  void set_means(const VectorList<D>& means);
  void set_covariances(const MatrixList<D, D>& covariances);
  void set_mixture_coefficients(const std::vector<double>& coeffs);

  constexpr size_t get_num_mixture_components() const;
  constexpr size_t get_data_dimension() const;
};

}  // namespace ContinuousHMM::GMM

#endif