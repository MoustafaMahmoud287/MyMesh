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
            size_t MaxSizeToRunGPU;
        };
#pragma pack(pop)

#pragma pack(push, 1)
        struct FullGPUSpec {
            GPUProfile m_gpu_profile;
            LimitsProfile m_gpu_limits;
        };
#pragma pack(pop)

    }
}