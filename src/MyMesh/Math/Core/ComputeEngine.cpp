#pragma once
#include <MyMesh/Math/Core/ComputeEngine.h>

namespace MyMesh {
    namespace MathInternal{

        ComputeEngine::ComputeEngine(Strategy str, int BlockSizeinMB_Or_PersistentBlockCount, int TempBlockCount, int deviceID)
            : m_gpu_enabled(false)
        {
            m_hardware_specs = Hardware::HardwareProfiler::loadProfile(deviceID);

            if (!m_hardware_specs.m_gpu_profile.is_valid) {
                std::cout << "[ComputeEngine] No valid GPU found. Running in CPU-Only Mode.\n";
                return;
            }

            size_t final_block_size = 0;
            int persistent_blocks = 0;

            if (str == Strategy::FIXED_BLOCK_SIZE) {
                size_t max_block_Size = m_hardware_specs.m_gpu_profile.total_vram_mb
                final_block_size = min static_cast<size_t>(BlockSizeinMB_Or_PersistentBlockCount) * 1024 * 1024;
                persistent_blocks = MIN_BLOCKS_NUM;

                // Safety check: Don't exceed physical Arena limits
                if (final_block_size > m_hardware_specs.m_gpu_limits.TestBlockSize) {
                    std::cout << "[ComputeEngine] Warning: Requested Block Size exceeds physical safe limits. Capping to Max Safe Size.\n";
                    final_block_size = m_hardware_specs.m_gpu_limits.TestBlockSize;
                }
            }
            else if (str == strategy::FIXED_BLOCK_COUNT) {
                persistent_blocks = BlockSizeinMB_Or_PersistentBlockCount;
                int total_blocks = persistent_blocks + TempBlockCount;

                // Divide safe VRAM by total requested blocks
                final_block_size = m_hardware_specs.m_gpu_limits.MaxSizeToRunGPU / total_blocks;
            }

            std::cout << "[ComputeEngine] Initializing GPU Arena. Block Size: " << (final_block_size / (1024 * 1024)) << " MB\n";
            m_arena = std::make_unique<CudaMemoryArena>(persistent_blocks, TempBlockCount, final_block_size);
            m_gpu_solver = std::make_unique<CudaSolver>(m_arena.get());

            m_gpu_enabled = true;
        }

        };
    }
}
