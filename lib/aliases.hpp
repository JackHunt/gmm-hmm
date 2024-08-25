#ifndef CONTINUOUS_HMM_ALIASES
#define CONTINUOUS_HMM_ALIASES

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <vector>

template <typename T>
using EigenAlloc = Eigen::aligned_allocator<T>;

template <size_t N>
using Vector = Eigen::Matrix<double, N, 1>;

template <size_t R, size_t C>
using Matrix = Eigen::Matrix<double, R, C>;

template <size_t N>
using VectorList = std::vector<Vector<N>, EigenAlloc<Vector<N>>>;

template <size_t R, size_t C>
using MatrixList = std::vector<Matrix<R, C>, EigenAlloc<Matrix<R, C>>>;

template <size_t N>
using NestedVectorList = std::vector<VectorList<N>>;

template <size_t R, size_t C>
using NestedMatrixList = std::vector<MatrixList<R, C>>;

#endif