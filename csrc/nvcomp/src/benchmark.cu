#include "benchmark_template_chunked.cuh"
#include "nvcomp/gdeflate.h"

static nvcompBatchedGdeflateOpts_t nvcompBatchedGdeflateOpts = {0};

static bool handleCommandLineArgument(
    const std::string& arg,
    const char* const* additionalArgs,
    size_t& additionalArgsUsed)
{
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