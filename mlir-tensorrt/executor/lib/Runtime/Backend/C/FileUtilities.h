#ifndef RUNTIME_BACKEND_C_FILEUTILITIES
#define RUNTIME_BACKEND_C_FILEUTILITIES

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static constexpr const char *kArtifactsDirEnvironmentVariable =
    "MTRT_ARTIFACTS_DIR";

static inline bool mtrtIsAbsolutePath(const std::string &path) {
  return !path.empty() && path.front() == '/';
}

static inline std::string mtrtJoinPath(const std::string &dir,
                                       const std::string &rel) {
  if (dir.empty())
    return rel;
  if (rel.empty())
    return dir;
  if (dir.back() == '/')
    return dir + rel;
  return dir + "/" + rel;
}

static inline std::vector<std::string>
mtrtGetSearchCandidates(const std::string &filename) {
  std::vector<std::string> candidates;
  candidates.reserve(3);

  if (filename.empty())
    return candidates;

  if (mtrtIsAbsolutePath(filename)) {
    candidates.push_back(filename);
    return candidates;
  }

  if (const char *env = std::getenv(kArtifactsDirEnvironmentVariable)) {
    std::string base(env);
    if (!base.empty())
      candidates.push_back(mtrtJoinPath(base, filename));
  }

  // Legacy behavior: treat filename as relative to CWD.
  candidates.push_back(filename);
  return candidates;
}

static inline size_t mtrtGetFileSize(const std::string &filename) {
  for (const std::string &candidate : mtrtGetSearchCandidates(filename)) {
    std::ifstream file(candidate, std::ios::binary | std::ios::ate);
    if (file)
      return file.tellg();
  }
  std::cerr << "Error opening file '" << filename << "'.\n";
  std::cerr << "Tried:\n";
  for (const std::string &candidate : mtrtGetSearchCandidates(filename))
    std::cerr << "  - " << candidate << "\n";
  std::cerr << "Hint: set " << kArtifactsDirEnvironmentVariable
            << " to the directory containing manifest.json\n";
  return 0;
}

static inline int mtrtReadInputFile(const std::string &filename,
                                    std::vector<char> &buffer) {
  std::ifstream file;
  std::string openedPath;
  for (const std::string &candidate : mtrtGetSearchCandidates(filename)) {
    file.open(candidate, std::ios::binary | std::ios::ate);
    if (file) {
      openedPath = candidate;
      break;
    }
    file.clear();
  }
  if (!file) {
    std::cerr << "Error opening file '" << filename << "'.\n";
    std::cerr << "Tried:\n";
    for (const std::string &candidate : mtrtGetSearchCandidates(filename))
      std::cerr << "  - " << candidate << "\n";
    std::cerr << "Hint: set " << kArtifactsDirEnvironmentVariable
              << " to the directory containing manifest.json\n";
    return 1;
  }

  // Get the size of the file
  std::streamsize size = file.tellg();

  // Move back to the beginning of the file
  file.seekg(0, std::ios::beg);

  // Create a vector to hold the file contents
  buffer.resize(size);

  // Read the entire file into the vector
  if (file.read(buffer.data(), size))
    return 0;

  std::cerr << "Error reading file!" << std::endl;
  return 1;
}

static inline int mtrtReadInputFile(const std::string &filename, char *buffer,
                                    size_t expectedSize) {
  std::ifstream file;
  std::string openedPath;
  for (const std::string &candidate : mtrtGetSearchCandidates(filename)) {
    file.open(candidate, std::ios::binary | std::ios::ate);
    if (file) {
      openedPath = candidate;
      break;
    }
    file.clear();
  }
  if (!file) {
    std::cerr << "Error opening file '" << filename << "'.\n";
    std::cerr << "Tried:\n";
    for (const std::string &candidate : mtrtGetSearchCandidates(filename))
      std::cerr << "  - " << candidate << "\n";
    std::cerr << "Hint: set " << kArtifactsDirEnvironmentVariable
              << " to the directory containing manifest.json\n";
    return 1;
  }

  // Get the size of the file
  size_t size = file.tellg();

  if (size != expectedSize) {
    std::cerr << "Error: file size mismatch: " << size << " != " << expectedSize
              << std::endl;
    return 1;
  }

  // Move back to the beginning of the file
  file.seekg(0, std::ios::beg);

  // Read the entire file into the vector
  if (file.read(buffer, size))
    return 0;

  std::cerr << "Error reading file!" << std::endl;
  return 1;
}

#endif // RUNTIME_BACKEND_C_FILEUTILITIES
