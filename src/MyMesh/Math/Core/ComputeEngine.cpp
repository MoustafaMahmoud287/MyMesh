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
            BlockCounterType persistent_blocks = 0;
            BlockCounterType temporary_blocks = 0;
            BlockCounterType blocks_count = 0;

            if (str == Strategy::FIXED_BLOCK_SIZE) {

                size_t max_block_Size = m_hardware_specs.m_gpu_limits.MaxAlocatedGPUMemory / (2 * MIN_BLOCKS_NUM);
                final_block_size = std::min (static_cast<size_t>(BlockSizeinMB_Or_PersistentBlockCount) * 1024 * 1024,  max_block_Size);
                blocks_count = m_hardware_specs.m_gpu_limits.MaxAlocatedGPUMemory / final_block_size;

                if (blocks_count <= BLOCKS_DIVIDE_THSHOLD) {
                    persistent_blocks = blocks_count / 2;
                    temporary_blocks = blocks_count / 2;
                }

                else {
                    persistent_blocks = (blocks_count * 2) / 3;
                    temporary_blocks = blocks_count - persistent_blocks;
                }

            }

            else if (str == Strategy::FIXED_BLOCK_COUNT) {

                persistent_blocks = std::max(BlockSizeinMB_Or_PersistentBlockCount, MIN_BLOCKS_NUM);
                temporary_blocks = std::max(TempBlockCount, MIN_BLOCKS_NUM);
                blocks_count = persistent_blocks + temporary_blocks;
                final_block_size = m_hardware_specs.m_gpu_limits.MaxAlocatedGPUMemory / blocks_count;
            }

            std::cout << "[ComputeEngine] Initializing GPU Arena. Block Size: " << (final_block_size / (1024 * 1024)) << " MB\n";
            
            m_arena = std::make_unique<CudaMemoryArena>(persistent_blocks, temporary_blocks, final_block_size);
            m_gpu_solver = std::make_unique<CudaSolver>(m_arena.get());

            m_gpu_enabled = true;
        }



    }
}
