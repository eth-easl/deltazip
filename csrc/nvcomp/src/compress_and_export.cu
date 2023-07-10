#include "nvcomp/gdeflate.h"

static nvcompBatchedGdeflateOpts_t nvcompBatchedGdeflateOpts = {2};

static bool handleCommandLineArgument(const std::string& arg,
                                      const char* const* additionalArgs,
                                      size_t& additionalArgsUsed) {
  if (arg == "--algorithm" || arg == "-a") {
    int algorithm_type = atoi(*additionalArgs);
    additionalArgsUsed = 1;
    if (algorithm_type < 0 || algorithm_type > 2) {
      std::cerr << "ERROR: Gdeflate algorithm must be 0, 1, or 2, but it is "
                << algorithm_type << std::endl;
      return false;
    }
    nvcompBatchedGdeflateOpts.algo = algorithm_type;
    return true;
  }
  return false;
}

static bool isGdeflateInputValid(const std::vector<std::vector<char>>& data) {
  for (const auto& chunk : data) {
    if (chunk.size() > 65536) {
      std::cerr << "ERROR: Gdeflate doesn't support chunk sizes larger than "
                   "65536 bytes."
                << std::endl;
      return false;
    }
  }

  return true;
}

void run_compression(const std::vector<std::vector<char>>& data,
                     const bool warmup, const size_t count,
                     const bool csv_output, const bool tab_separator,
                     const size_t duplicate_count, const size_t num_files) {
  benchmark_assert(IsInputValid(data), "Invalid input data");
  const std::string separator = use_tabs ? "\t" : ",";
  size_t total_bytes = 0;
  size_t chunk_size = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
    if (part.size() > chunk_size) {
      chunk_size = part.size();
    }
  }
  BatchData input_data(data);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  const size_t batch_size = input_data.size();

  std::vector<size_t> h_input_sizes(batch_size);
  CUDA_CHECK(cudaMemcpy(h_input_sizes.data(), input_data.sizes(),
                        sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost));

  size_t compressed_size = 0;
  double comp_time = 0.0;
  double decomp_time = 0.0;
  for (size_t iter = 0; iter<count; ++iter) {
    //compression
    nvcompStatus_t status;
    size_t comp_temp_bytes;
  }
}