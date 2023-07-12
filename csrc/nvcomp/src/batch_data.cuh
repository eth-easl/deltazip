#pragma once
#include <thrust/device_vector.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"

using namespace nvcomp;

namespace {

class BatchData {
 public:
  BatchData(const std::vector<std::vector<char>>& host_data)
      : m_ptrs(), m_sizes(), m_data(), m_size(0) {
    m_size = host_data.size();

    // find max chunk size and build prefixsum
    std::vector<size_t> prefixsum(m_size + 1, 0);
    size_t chunk_size = 0;
    for (size_t i = 0; i < m_size; ++i) {
      if (chunk_size < host_data[i].size()) {
        chunk_size = host_data[i].size();
      }
      prefixsum[i + 1] = prefixsum[i] + host_data[i].size();
    }

    m_data = thrust::device_vector<uint8_t>(prefixsum.back());

    std::vector<void*> data_ptrs(size());
    for (size_t i = 0; i < size(); ++i) {
      data_ptrs[i] = static_cast<void*>(data() + prefixsum[i]);
    }

    m_ptrs = thrust::device_vector<void*>(data_ptrs);
    std::vector<size_t> sizes(m_size);
    for (size_t i = 0; i < sizes.size(); ++i) {
      sizes[i] = host_data[i].size();
    }
    m_sizes = thrust::device_vector<size_t>(sizes);
    // copy data to GPU
    for (size_t i = 0; i < host_data.size(); ++i) {
      CUDA_CHECK(cudaMemcpy(data_ptrs[i], host_data[i].data(),
                            host_data[i].size(), cudaMemcpyHostToDevice));
    }
  }

  BatchData(const size_t max_output_size, const size_t batch_size)
      : m_ptrs(), m_sizes(), m_data(), m_size(batch_size) {
    m_data = thrust::device_vector<uint8_t>(max_output_size * size());

    std::vector<size_t> sizes(size(), max_output_size);
    m_sizes = thrust::device_vector<size_t>(sizes);

    std::vector<void*> ptrs(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      ptrs[i] = data() + max_output_size * i;
    }
    m_ptrs = thrust::device_vector<void*>(ptrs);
  }

  BatchData(BatchData&& other) = default;

  // disable copying
  BatchData(const BatchData& other) = delete;
  BatchData& operator=(const BatchData& other) = delete;

  void** ptrs() { return m_ptrs.data().get(); }

  size_t* sizes() { return m_sizes.data().get(); }

  uint8_t* data() { return m_data.data().get(); }

  size_t total_size() const { return m_data.size(); }

  size_t size() const { return m_size; }

 private:
  thrust::device_vector<void*> m_ptrs;
  thrust::device_vector<size_t> m_sizes;
  thrust::device_vector<uint8_t> m_data;
  size_t m_size;
};

std::vector<std::vector<char>> readFileWithPageSizes(
    const std::string& filename) {
  std::vector<std::vector<char>> res;

  std::ifstream fin(filename, std::ifstream::binary);

  while (!fin.eof()) {
    uint64_t chunk_size;
    fin.read(reinterpret_cast<char*>(&chunk_size), sizeof(uint64_t));
    if (fin.eof()) break;
    res.emplace_back(chunk_size);
    fin.read(reinterpret_cast<char*>(res.back().data()), chunk_size);
  }

  return res;
}

std::vector<char> readFile(const std::string& filename) {
  std::ifstream fin(filename, std::ifstream::binary);
  if (!fin) {
    std::cerr << "ERROR: Unable to open \"" << filename << "\" for reading."
              << std::endl;
    throw std::runtime_error("Error opening file for reading.");
  }

  fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  fin.seekg(0, std::ios_base::end);
  auto fileSize = static_cast<std::streamoff>(fin.tellg());
  fin.seekg(0, std::ios_base::beg);

  std::vector<char> host_data(fileSize);
  fin.read(host_data.data(), fileSize);

  if (!fin) {
    std::cerr << "ERROR: Unable to read all of file \"" << filename << "\"."
              << std::endl;
    throw std::runtime_error("Error reading file.");
  }

  return host_data;
}

std::vector<std::vector<char>> multi_file(
    const std::vector<std::string>& filenames, const size_t chunk_size,
    const bool has_page_sizes, const size_t duplicate_count) {
  std::vector<std::vector<char>> split_data;

  for (auto const& filename : filenames) {
    if (!has_page_sizes) {
      std::vector<char> filedata = readFile(filename);
      const size_t num_chunks = (filedata.size() + chunk_size - 1) / chunk_size;
      size_t offset = 0;
      for (size_t c = 0; c < num_chunks; ++c) {
        const size_t size_of_this_chunk =
            std::min(chunk_size, filedata.size() - offset);
        split_data.emplace_back(
            std::vector<char>(filedata.data() + offset,
                              filedata.data() + offset + size_of_this_chunk));
        offset += size_of_this_chunk;
        assert(offset <= filedata.size());
      }
    } else {
      std::vector<std::vector<char>> filedata = readFileWithPageSizes(filename);
      split_data.insert(split_data.end(), filedata.begin(), filedata.end());
    }
  }

  if (duplicate_count > 1) {
    // Make duplicate_count copies of the contents of split_data,
    // but copy into a separate std::vector, to avoid issues with the
    // memory being reallocated while the contents are being copied.
    std::vector<std::vector<char>> duplicated;
    const size_t original_num_chunks = split_data.size();
    duplicated.reserve(original_num_chunks * duplicate_count);
    for (size_t d = 0; d < duplicate_count; ++d) {
      duplicated.insert(duplicated.end(), split_data.begin(), split_data.end());
    }
    // Now that there are duplicate_count copies of split_data in
    // duplicated, swap them, so that they're in split_data.
    duplicated.swap(split_data);
  }

  return split_data;
}

}  // namespace