#ifndef HMM_DEMO
#define HMM_DEMO

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../lib/aliases.hpp"
#include "../lib/hmm.hpp"
#include "MotionData.hpp"

namespace HMMDemo {

using Data::MotionData;
using Data::Timestamp;

using HMM::ContinuousInputHMM;

// Log-Likelihood difference threshold for temrination and max epochs.
constexpr static double convergence_threshold = 1e-5;
constexpr static unsigned int num_epochs = 100;

// Specifics for the given problem.
constexpr static size_t K = 3;  // # states.
constexpr static size_t M = 4;  // # GMM mixture components.
constexpr static size_t D = 3;  // # observation dimensions.

/*!
 * Generate observation vectors $\bm{x}_{t} \in \mathbb{R}^{3}$ from a
 * MotionData data source.
 *
 * \param data MotionData data source.
 * \return Observation vectors.
 */
inline std::pair<std::vector<Timestamp>, VectorList<3>> get_observation_vectors(
    MotionData& data) {
  std::vector<Timestamp> timestamps;
  VectorList<3> observations;

  while (data.has_next()) {
    const auto t = data.get_next_timestamp();
    if (t.has_value()) {
      const auto datum = data.get_motion_at_time(t.value());
      if (datum.has_value()) {
        // Get displacement.
        const auto displacement = datum.value().first;
        const auto normalised_displacement = displacement.normalized();

        // Get heading.
        const auto heading = datum.value().second;
        const auto heading_rad = heading * M_PI / 180.0;

        // Create observation vector.
        Vector<3> obs;
        obs << normalised_displacement(0), normalised_displacement(1),
            heading_rad;
        observations.push_back(obs);
        timestamps.push_back(t.value());
      }
    }
  }
  return {timestamps, observations};
}

/*!
 * Writes out a csv file in which column 0 contains sample timestamps
 * and column 1 contains the probability of being in state $m$.
 *
 * \param timestamps Sample timestamps.
 * \param dists State Distributions, output of HMM.
 * \param k State to write probabilities for.
 * \param fname_prefix Optional prefix for output filenames.
 */
template <size_t K>
inline void write_probabilities_for_state(
    const std::vector<Timestamp>& timestamps, const VectorList<K>& dists,
    size_t k, const std::string fname_prefix = "prob_state_") {
  // Check lengths.
  if (timestamps.size() != dists.size()) {
    throw std::runtime_error("Inconsistent timestamp to distribution count.");
  }

  // Create output stream.
  std::stringstream fname;
  fname << fname_prefix << k << ".csv";
  std::ofstream out_stream(fname.str());
  if (!out_stream.good()) {
    throw std::runtime_error("Could not open an output stream");
  }

  // Write out lines as <timestamp, probability state k>
  for (size_t t = 0; t < timestamps.size(); t++) {
    out_stream << timestamps.at(t) << "," << dists.at(t)(k) << std::endl;
  }
}

/*!
 * Takes a set of temporally monotonic samples and estimates
 * the underlying stability state; consistent motion, direction change
 * or erratic motion.
 *
 * \param observations Observation vectors.
 * \return Most probable state and the state distribution for each sample time
 * $t$.
 */
template <size_t K, size_t M, size_t D>
inline std::pair<std::vector<size_t>, VectorList<K>> demo(
    const VectorList<K>& observations) {
  // Create the initial state probability distribution for the HMM.
  // Smooth motion, direction change or erratic motion. K states.
  Vector<K> initial_state_dist = Vector<K>::Ones();
  initial_state_dist /= K;

  // Create and train the HMM.
  ContinuousInputHMM<K, M, D> hmm(initial_state_dist, observations);
  hmm.train(num_epochs, convergence_threshold);

  // Get the optimal state sequence.
  return hmm.get_state_sequence();
}

}  // namespace HMMDemo

#endif