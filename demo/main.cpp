#include "main.hpp"

using namespace HMMDemo;
using namespace HMMDemo::Data;

int main(int argc, char** argv) {
  // Get path of data file.
  if (argc <= 1) {
    std::cout << "Usage: ./demo <input file path>" << std::endl;
    return -1;
  }

  // Load in the data.
  MotionData data(argv[1]);

  // Get timestamps and observation vectors.
  std::vector<Timestamp> timestamps;
  VectorList<K> observations;
  std::tie(timestamps, observations) = get_observation_vectors(data);

  // Run the demo.
  std::vector<size_t> states;
  VectorList<K> state_dists;
  std::tie(states, state_dists) = demo<K, M, D>(observations);

  // Write out state probabilities for each timestamp.
  for (size_t k = 0; k < K; k++) {
    write_probabilities_for_state<K>(timestamps, state_dists, k);
  }
}