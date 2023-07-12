#include "common.cuh"
#include "nvcomp/gdeflate.h"

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

int main(int argc, char** argv) {
  args_type args = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(args.gpu));
  std::string filename = args.filenames.front();

  decompress_file(nvcompBatchedGdeflateCompressGetTempSize,
                  nvcompBatchedGdeflateCompressGetMaxOutputChunkSize,
                  nvcompBatchedGdeflateCompressAsync,
                  nvcompBatchedGdeflateDecompressGetTempSize,
                  nvcompBatchedGdeflateDecompressAsync, isGdeflateInputValid,
                  nvcompBatchedGdeflateOpts, filename, args.chunk_size,
                  args.batch_size, args.input_size);
}