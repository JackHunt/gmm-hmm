#ifndef HMM_DEMO_2D_MOTION_DATA
#define HMM_DEMO_2D_MOTION_DATA

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace HMMDemo::Data {

using Timestamp = unsigned long long;
using Displacement = Eigen::Vector2d;
using Heading = double;

using MotionDatum = std::pair<Displacement, Heading>;

/*!
 * This class implements a simple wrapper around a temporal dataset.
 * The intended use of this class is as a queryable data source for
 * the linear traversal of a dataset.
 *
 */
class MotionData final {
 protected:
  // std::map<> inserts elements in a sorted manner,
  // ensuring that samples are temporally monotonic.
  std::map<Timestamp, MotionDatum> motion_data;

  // When stepping through the data linearly with
  // get_next_timestamp(), keep an iterator pointing to the
  // next entry.
  decltype(motion_data)::const_iterator current_entry;

 public:
  MotionData(const std::filesystem::path& in_file);

  bool has_next() const;

  void reset_counter();

  std::optional<Timestamp> get_next_timestamp();

  std::optional<MotionDatum> get_motion_at_time(Timestamp t) const;
};

}  // namespace HMMDemo::Data

namespace HMMDemo::Data::Util {

/*!
 * Utility function to tokenize (split) a given string on a
 * given delimiter.
 *
 * \param input String to tokenize.
 * \param delim Character delimiter to split on.
 * \return String split into tokens on the given delimiter.
 */
inline std::vector<std::string> split_on_delimiter(const std::string& input,
                                                   char delim) {
  std::vector<std::string> tokens;
  std::istringstream in_stream(input);
  std::string token;

  while (std::getline(in_stream, token, delim)) {
    token.erase(std::remove_if(token.begin(), token.end(), isspace),
                token.end());
    tokens.push_back(token);
  }

  return tokens;
}

}  // namespace HMMDemo::Data::Util

#endif