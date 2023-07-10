//
// Created by mfuntowicz on 3/28/23.
//
// ! Requires std=c++20 
#include <span>
#include "safetensors.hpp"
#include "nlohmann/json.hpp"

namespace huggingface::safetensors {

    safetensors_t deserialize(std::basic_istream<char> &in) {
        uint64_t header_size = 0;

        // todo: handle exception
        in.read(reinterpret_cast<char *>(&header_size), sizeof header_size);

        std::vector<char> meta_block(header_size);
        in.read(meta_block.data(), static_cast<std::streamsize>(header_size));
        const auto metadatas = json::parse(meta_block);

        // How many bytes remaining to pre-allocate the storage tensor
        in.seekg(0, std::ios::end);
        std::streamsize f_size = in.tellg();
        in.seekg(8 + header_size, std::ios::beg);
        const auto tensors_size = f_size - 8 - header_size;

        auto metas_table = std::unordered_map<std::string, const metadata_t>(metadatas.size());
        auto tensors_storage = std::vector<char>(tensors_size);

        // Read the remaining content
        in.read(tensors_storage.data(), static_cast<std::streamsize>(tensors_size));

        // Populate the meta lookup table
        if (metadatas.is_object()) {
            for (auto &item: metadatas.items()) {
                if (item.key() != "__metadata__") {
                    const auto name = std::string(item.key());
                    const auto& info = item.value();

                    const metadata_t meta = {info["dtype"].get<dtype>(), info["shape"], info["data_offsets"]};
                    metas_table.insert(std::pair(name, meta));
                }
            }
        }

        return {metas_table, tensors_storage};
    }

    safetensors_t::safetensors_t(std::unordered_map<std::string, const metadata_t> &metas, std::vector<char> &storage)
            : metas(metas), storage(storage) {}

    std::span<const char> safetensors_t::operator[](const char *name) const {
        const auto meta = metas.at(name);
        const auto [t_begin, t_end] = meta.data_offsets;
        return {storage.begin() + static_cast<ptrdiff_t>(t_begin), storage.begin() + static_cast<ptrdiff_t>(t_end)};
    }
}

int main(){
    return 0;
}