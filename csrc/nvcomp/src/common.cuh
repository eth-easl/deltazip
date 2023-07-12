#pragma once
#include <thrust/device_vector.h>

#include <iostream>
#include <string>
#include <vector>

#include "batch_data.cuh"
#include "common.h"
#include "nvcomp/gdeflate.h"
constexpr const char* const REQUIRED_PARAMTER = "_REQUIRED_";

static bool handleCommandLineArgument(const std::string& arg,
                                      const char* const* additionalArgs,
                                      size_t& additionalArgsUsed);

struct args_type {
  int gpu;
  std::vector<std::string> filenames;
  std::string output_filename;
  size_t warmup_count;
  size_t iteration_count;
  size_t duplicate_count;
  bool csv_output;
  bool use_tabs;
  bool has_page_sizes;
  size_t chunk_size;
  size_t batch_size;
  size_t input_size;
};

struct parameter_type {
  std::string short_flag;
  std::string long_flag;
  std::string description;
  std::string default_value;
};

void usage(const std::string& name,
           const std::vector<parameter_type>& parameters) {
  std::cout << "Usage: " << name << " [OPTIONS]" << std::endl;
  for (const parameter_type& parameter : parameters) {
    std::cout << "  -" << parameter.short_flag << ",--" << parameter.long_flag;
    std::cout << "  : " << parameter.description << std::endl;
    if (parameter.default_value.empty()) {
      // no default value
    } else if (parameter.default_value == REQUIRED_PARAMTER) {
      std::cout << "    required" << std::endl;
    } else {
      std::cout << "    default=" << parameter.default_value << std::endl;
    }
  }
}

std::string bool_to_string(const bool b) {
  if (b) {
    return "true";
  } else {
    return "false";
  }
}

args_type parse_args(int argc, char** argv) {
  args_type args;
  args.gpu = 0;
  args.warmup_count = 1;
  args.iteration_count = 1;
  args.duplicate_count = 0;
  args.csv_output = false;
  args.use_tabs = false;
  args.has_page_sizes = false;
  args.chunk_size = 1 << 16;
  args.batch_size = 1;
  args.input_size = 0;
  const std::vector<parameter_type> params{
      {"?", "help", "Show options.", ""},
      {"g", "gpu", "GPU device number", std::to_string(args.gpu)},
      {"f", "input_file",
       "The list of inputs files. All files must start "
       "with a character other than '-'",
       "_required_"},
      {"o", "output_file", "The output file", "_required_"},
      {"w", "warmup_count", "The number of warmup iterations to perform.",
       std::to_string(args.warmup_count)},
      {"i", "iteration_count", "The number of runs to average.",
       std::to_string(args.iteration_count)},
      {"x", "duplicate_data", "Clone uncompressed chunks multiple times.",
       std::to_string(args.duplicate_count)},
      {"c", "csv_output", "Output in column/csv format.",
       bool_to_string(args.csv_output)},
      {"e", "tab_separator",
       "Use tabs instead of commas when "
       "'--csv_output' is specificed.",
       bool_to_string(args.use_tabs)},
      {"s", "file_with_page_sizes",
       "File(s) contain pages, each prefixed "
       "with int64 size.",
       bool_to_string(args.has_page_sizes)},
      {"p", "chunk_size", "Chunk size when splitting uncompressed data.",
       std::to_string(args.chunk_size)},
      {"b", "batch_size", "batch size when decompress.",
       std::to_string(args.chunk_size)},
      {"inp", "input_size", "input size when decompress.",
       std::to_string(args.input_size)},
  };

  char** argv_end = argv + argc;
  const std::string name(argv[0]);
  argv += 1;

  while (argv != argv_end) {
    std::string arg(*(argv++));
    bool found = false;
    for (const parameter_type& param : params) {
      if (arg == "-" + param.short_flag || arg == "--" + param.long_flag) {
        found = true;
        // found the parameter
        if (param.long_flag == "help") {
          usage(name, params);
          std::exit(0);
        }
        // everything from here on out requires an extra parameter
        if (argv >= argv_end) {
          std::cerr << "ERROR: Missing argument" << std::endl;
          usage(name, params);
          std::exit(1);
        }
        if (param.long_flag == "gpu") {
          args.gpu = std::stol(*(argv++));
          break;
        } else if (param.long_flag == "input_file") {
          // read all following arguments until a new flag is found
          char** next_argv_ptr = argv;
          while (next_argv_ptr < argv_end && (*next_argv_ptr)[0] != '-') {
            args.filenames.emplace_back(*next_argv_ptr);
            next_argv_ptr = ++argv;
          }
          break;
        } else if (param.long_flag == "warmup_count") {
          args.warmup_count = size_t(std::stoull(*(argv++)));
          break;
        } else if (param.long_flag == "iteration_count") {
          args.iteration_count = size_t(std::stoull(*(argv++)));
          break;
        } else if (param.long_flag == "duplicate_data") {
          args.duplicate_count = size_t(std::stoull(*(argv++)));
          break;
        } else if (param.long_flag == "output_file") {
          args.output_filename = *(argv++);
        } else if (param.long_flag == "batch_size") {
          args.batch_size = size_t(std::stoull(*(argv++)));
        } else if (param.long_flag == "input_size") {
          args.input_size = size_t(std::stoull(*(argv++)));
        } else if (param.long_flag == "chunk_size") {
          args.chunk_size = size_t(std::stoull(*(argv++)));
          break;
        } else {
          std::cerr << "INTERNAL ERROR: Unhandled parameter '" << arg << "'."
                    << std::endl;
          usage(name, params);
          std::exit(1);
        }
      }
    }
    size_t argumentsUsed = 0;
    if (!found && !handleCommandLineArgument(arg, argv, argumentsUsed)) {
      std::cerr << "ERROR: Unknown argument '" << arg << "'." << std::endl;
      usage(name, params);
      std::exit(1);
    }
    argv += argumentsUsed;
  }

  if (args.filenames.empty()) {
    std::cerr << "ERROR: Must specify at least one input file." << std::endl;
    std::exit(1);
  }

  return args;
}

static nvcompBatchedGdeflateOpts_t nvcompBatchedGdeflateOpts = {2};

template <typename CompGetTempT, typename CompGetSizeT, typename CompAsyncT,
          typename DecompGetTempT, typename DecompAsyncT,
          typename IsInputValidT, typename FormatOptsT>
void run_compress_template(CompGetTempT BatchedCompressGetTempSize,
                           CompGetSizeT BatchedCompressGetMaxOutputChunkSize,
                           CompAsyncT BatchedCompressAsync,
                           DecompGetTempT BatchedDecompressGetTempSize,
                           DecompAsyncT BatchedDecompressAsync,
                           IsInputValidT IsInputValid,
                           const FormatOptsT format_opts,
                           const std::vector<std::vector<char>>& data,
                           const bool warmup, const size_t count,
                           const bool csv_output, const size_t duplicate_count,
                           const size_t num_files,
                           const std::string output_filename = "") {
  const std::string separator = ",";

  size_t total_bytes = 0;
  size_t chunk_size = 0;
  for (const std::vector<char>& part : data) {
    total_bytes += part.size();
    if (part.size() > chunk_size) {
      chunk_size = part.size();
    }
  }
  // build up metadata
  BatchData input_data(data);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const size_t batch_size = input_data.size();
  std::cout << "chunk size: " << chunk_size << std::endl;
  std::cout << "batch_size: " << batch_size << std::endl;
  std::vector<size_t> h_input_sizes(batch_size);

  CUDA_CHECK(cudaMemcpy(h_input_sizes.data(), input_data.sizes(),
                        sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost));
  size_t compressed_size = 0;
  double comp_time = 0.0;
  for (size_t iter = 0; iter < count; ++iter) {
    // compression
    nvcompStatus_t status;

    // Compress on the GPU using batched API
    size_t comp_temp_bytes;
    status = BatchedCompressGetTempSize(batch_size, chunk_size, format_opts,
                                        &comp_temp_bytes);
    benchmark_assert(status == nvcompSuccess,
                     "BatchedCompressGetTempSize() failed.");

    void* d_comp_temp;
    CUDA_CHECK(cudaMalloc(&d_comp_temp, comp_temp_bytes));

    size_t max_out_bytes;
    status = BatchedCompressGetMaxOutputChunkSize(chunk_size, format_opts,
                                                  &max_out_bytes);
    benchmark_assert(status == nvcompSuccess,
                     "BatchedGetMaxOutputChunkSize() failed.");

    BatchData compress_data(max_out_bytes, batch_size);

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, stream));

    status = BatchedCompressAsync(input_data.ptrs(), input_data.sizes(),
                                  chunk_size, batch_size, d_comp_temp,
                                  comp_temp_bytes, compress_data.ptrs(),
                                  compress_data.sizes(), format_opts, stream);
    benchmark_assert(status == nvcompSuccess, "BatchedCompressAsync() failed.");

    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // free compression memory
    CUDA_CHECK(cudaFree(d_comp_temp));

    float compress_ms;
    CUDA_CHECK(cudaEventElapsedTime(&compress_ms, start, end));

    // compute compression ratio
    std::vector<size_t> compressed_sizes_host(compress_data.size());
    CUDA_CHECK(cudaMemcpy(compressed_sizes_host.data(), compress_data.sizes(),
                          compress_data.size() * sizeof(*compress_data.sizes()),
                          cudaMemcpyDeviceToHost));
    // for (int ix = 0; ix < compress_data.size(); ++ix) {
    //   printf("Frame %d comp ratio %f\n", ix, double{64*1024} /
    //   (double)(compressed_sizes_host[ix]));
    // }
    size_t comp_bytes = 0;
    for (const size_t s : compressed_sizes_host) {
      comp_bytes += s;
    }

    // Then do file output
    std::vector<uint8_t> comp_data(comp_bytes);
    std::vector<uint8_t*> comp_ptrs(batch_size);
    cudaMemcpy(comp_ptrs.data(), compress_data.ptrs(),
               sizeof(size_t) * batch_size, cudaMemcpyDefault);
    size_t ix_offset = 0;
    for (int ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
      cudaMemcpy(&comp_data[ix_offset], comp_ptrs[ix_chunk],
                 compressed_sizes_host[ix_chunk], cudaMemcpyDefault);
      ix_offset += compressed_sizes_host[ix_chunk];
    }
    std::ofstream outfile{output_filename.c_str(), outfile.binary};
    outfile.write(reinterpret_cast<char*>(comp_data.data()), ix_offset);
    outfile.close();
    compressed_size += comp_bytes;
    comp_time += compress_ms * 1.0e-3;
  }
  const double comp_ratio = (double)total_bytes / compressed_size;
  const double compression_throughput_gbs =
      (double)total_bytes / (1.0e9 * comp_time);
  std::cout << "-- FMZip Compression Perf Report --" << std::endl;
  std::cout << "files: " << num_files << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;
  std::cout << "comp_size: " << compressed_size
            << ", compressed ratio: " << std::fixed << std::setprecision(4)
            << comp_ratio << std::endl;
  std::cout << "compression throughput (GB/s): " << compression_throughput_gbs
            << std::endl;
}

template <typename CompGetTempT, typename CompGetSizeT, typename CompAsyncT,
          typename DecompGetTempT, typename DecompAsyncT,
          typename IsInputValidT, typename FormatOptsT>
void decompress_file(CompGetTempT BatchedCompressGetTempSize,
                     CompGetSizeT BatchedCompressGetMaxOutputChunkSize,
                     CompAsyncT BatchedCompressAsync,
                     DecompGetTempT BatchedDecompressGetTempSize,
                     DecompAsyncT BatchedDecompressAsync,
                     IsInputValidT IsInputValid, const FormatOptsT format_opts,
                     const std::string in_filename, const size_t chunk_size,
                     const size_t batch_size, const size_t input_data_size) {
  // variables used in the decompression
  nvcompStatus_t status;
  // read file
  std::vector<std::string> filenames(1);
  filenames[0] = in_filename;
  auto data = multi_file(filenames, chunk_size, false, 1);
  // build up metadata
  // step 1: build up "sizes" vector
  // for batch_size-1, we just use the chunk_size
  // for i=batch_size, we use input_data_size - (batch_size-1)*chunk_size
  size_t max_out_bytes;
  status = BatchedCompressGetMaxOutputChunkSize(chunk_size, format_opts,
                                                &max_out_bytes);
  benchmark_assert(status == nvcompSuccess,
                   "BatchedCompressGetMaxOutputChunkSize() failed.");
  std::vector<size_t> input_data_sizes(batch_size);
  for (size_t i = 0; i < batch_size - 1; ++i) {
    input_data_sizes[i] = chunk_size;
  }
  input_data_sizes[batch_size - 1] =
      input_data_size - (batch_size - 1) * chunk_size;
  BatchData compress_data(data);
  std::cout<<"max_out_bytes: "<<max_out_bytes<<std::endl;
  std::cout<<"input data sizes: "<<input_data_sizes.size()<<std::endl;
  BatchData decomp_data(max_out_bytes, input_data_sizes.size());

  // Create CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // CUDA events to measure decompression time
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  size_t decomp_temp_bytes;
  status = nvcompBatchedGdeflateDecompressGetTempSize(
      compress_data.size(), chunk_size, &decomp_temp_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error(
        "ERROR: nvcompBatchedGdeflateDecompressGetTempSize() not successful");
  }

  void* d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

  size_t* d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, decomp_data.size() * sizeof(size_t)));

  nvcompStatus_t* d_statuses;
  CUDA_CHECK(
      cudaMalloc(&d_statuses, decomp_data.size() * sizeof(nvcompStatus_t)));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEventRecord(start, stream);
  status = nvcompBatchedGdeflateDecompressAsync(
      compress_data.ptrs(), compress_data.sizes(), decomp_data.sizes(),
      d_decomp_sizes, compress_data.size(), d_decomp_temp, decomp_temp_bytes,
      decomp_data.ptrs(), d_statuses, stream);

  if (status != nvcompSuccess) {
    throw std::runtime_error(
        "ERROR: nvcompBatchedGdeflateDecompressAsync() not successful");
  }
  cudaEventRecord(end, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  //   size_t total_bytes = 0;
  //   for (const std::vector<char>& part : data) {
  //     total_bytes += part.size();
  //   }
  //   std::cout << "----------" << std::endl;
  //   std::cout << "files: " << filenames.size() << std::endl;
  //   std::cout << "compressed (B): " << total_bytes << std::endl;

  //   const double decompression_throughput_gbs =
  //       (double)total_bytes / (1.0e9 * decompress_ms * 1.0e-3);

  //   std::cout << "decompression throughput (GB/s): "
  //             << decompression_throughput_gbs << std::endl;
  //   std::cout << "decompression takes: " << decompress_ms * 1.0e-3 << " s"
  //             << std::endl;
}