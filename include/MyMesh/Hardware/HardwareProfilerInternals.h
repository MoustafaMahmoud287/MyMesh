#pragma once

#include <string>
#include <cstdint>

namespace MyMesh {
    namespace Hardware {

#pragma pack(push, 1)
        struct GPUProfile {
            int id;
            size_t total_vram_mb;
            int max_threads_per_block;
            int compute_capability_major;
            int compute_capability_minor;
            bool is_valid;
        };
#pragma pack(pop)

#pragma pack(push, 1)
        struct LimitsProfile {
            size_t MinSizeToRunGPU;
            size_t MaxAlocatedGPUMemory;
            size_t TestBlockSize;
        };
#pragma pack(pop)

#pragma pack(push, 1)
        struct FullGPUSpec {
            GPUProfile m_gpu_profile;
            LimitsProfile m_gpu_limits;
        };
#pragma pack(pop)

        struct LimitsPrams {
            size_t testing_block_default_size = 128 * 1024 * 1024;
            size_t min_dim = 1;
            size_t max_dim = 15000;
            size_t pc_safty_vram = 1024ULL * 1024 * 1024;
            size_t min_size = 1024 * 1024;
            int num_of_dummy_persistent_blocks = 3;
            int num_of_dummy_temp_blocks = 3;
            int num_of_total_dummy_blocks = num_of_dummy_persistent_blocks + num_of_dummy_temp_blocks;
            int number_of_rounds = 7;
            float sparsity = 0.005f;
        };

    }
}


