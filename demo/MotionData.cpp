#include "MotionData.hpp"

using namespace HMMDemo::Data;
using HMMDemo::Data::Util::split_on_delimiter;

/*!
 * Load the 2D motion data from a given file.
 *
 * \param in_file Path to data file.
 */
MotionData::MotionData(const std::filesystem::path& in_file) {
  // Ensure that the given file exists.
  if (!std::filesystem::exists(in_file)) {
    const auto err = std::make_error_code(std::errc::no_such_file_or_directory);
    throw std::filesystem::filesystem_error("File does not exist!", in_file,
                                            err);
  }

  // Verify that an input stream can be opened for the file.
  std::ifstream in_stream(in_file);
  if (!in_stream.good()) {
    const auto err = std::make_error_code(std::errc::io_error);
    throw std::filesystem::filesystem_error("Error reading file!", in_file,
                                            err);
  }

  // Read in file as a map of displacements and headings,
  // indexed by their associated timestamps.
  unsigned int line_num = 0;
  std::string datum;
  while (std::getline(in_stream, datum)) {
    const auto tokens = split_on_delimiter(datum, ',');
    try {
      const auto t = std::stoull(tokens[0]);
      const auto x = std::stod(tokens[1]);
      const auto y = std::stod(tokens[2]);
      const auto theta = std::stod(tokens[3]);

      motion_data[t] = std::make_pair(Displacement(x, y), theta);
    } catch (const std::invalid_argument& e) {
      std::cerr << "Error parsing line #" << line_num << std::endl;
      std::cout << e.what();
    }
  }
  reset_counter();
}

/*!
 * Determines if there is another datum that can be provided.
 *
 * \return True if another datum exists.
 */
bool MotionData::has_next() const {
  return current_entry != motion_data.cend();
}

/*!
 * Reset the internal data iterator.
 *
 * Essentially resets linear traversal to the beginning
 * of the sequence.
 *
 */
void MotionData::reset_counter() { current_entry = motion_data.cbegin(); }

/*!
 * Provides the timestamp of the next datum.
 *
 * \return Optional timestamp. nullopt if no future timestamps exist.
 */
std::optional<Timestamp> MotionData::get_next_timestamp() {
  if (current_entry != motion_data.cend()) {
    const auto t = current_entry->first;
    current_entry++;
    return t;
  }
  return std::nullopt;
}

/*!
 * Provide a motion datum at a given time.
 *
 * \param t Timestamp to provide a datum for,
 * \return Optional motiond datum. nullopt if no future timestamps exist.
 */
std::optional<MotionDatum> MotionData::get_motion_at_time(Timestamp t) const {
  const auto entry = motion_data.find(t);
  if (entry != motion_data.cend()) {
    return entry->second;
  }
  return std::nullopt;
}