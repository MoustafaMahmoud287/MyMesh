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
            std::cout << "[ComputeEngine] Initializing GPU Arena. Number of Persistent Blocks: " << persistent_blocks << "\n";
            std::cout << "[ComputeEngine] Initializing GPU Arena. Number of Temporary Blocks: " << temporary_blocks << "\n";

            m_arena = std::make_unique<CudaMemoryArena>(persistent_blocks, temporary_blocks, final_block_size);
            m_gpu_solver = std::make_unique<CudaSolver>(m_arena.get());

            m_gpu_enabled = true;
        }


        std::optional<CPUSparseMatrix> ComputeEngine::multiply(const ComputeOperand& A, const ComputeOperand& B,
            ComputeTarget& Target)
        {

            if (!m_gpu_enabled) {
                return m_cpu_solver.multiply(*A.cpu_matrix, *B.cpu_matrix);
            }

            if ((A.cpu_matrix == nullptr && A.gpu_block_index == -1) || (B.cpu_matrix == nullptr && B.gpu_block_index == -1)) return std::nullopt;

            if (A.cpu_matrix != nullptr && B.cpu_matrix != nullptr) {
                size_t needed_bytes_A = (A.cpu_matrix->nonZeros() * sizeof(float)) + (A.cpu_matrix->nonZeros() * sizeof(int)) + ((A.cpu_matrix->rows() + 1) * sizeof(int));
                size_t needed_bytes_B = (B.cpu_matrix->nonZeros() * sizeof(float)) + (B.cpu_matrix->nonZeros() * sizeof(int)) + ((B.cpu_matrix->rows() + 1) * sizeof(int));

                if(needed_bytes_A < m_hardware_specs.m_gpu_limits.MinSizeToRunGPU && needed_bytes_B < m_hardware_specs.m_gpu_limits.MinSizeToRunGPU)
                    return m_cpu_solver.multiply(*A.cpu_matrix, *B.cpu_matrix);
            }

            auto save = Target.is_intermediate ? CudaSaveOptions::SCRATCHPAD_BLOCK : CudaSaveOptions::PERSISTENT_BLOCK;
            auto needed_blocks = (save == CudaSaveOptions::SCRATCHPAD_BLOCK) ? 3 : 2;

            if (needed_blocks > m_arena->emptyTemporaryBlockCount()) {
                if (A.cpu_matrix == nullptr || B.cpu_matrix == nullptr) {

                    return m_cpu_solver.multiply(A.cpu_matrix == nullptr ? m_arena->downloadMatrix(A.gpu_block_index) : *A.cpu_matrix,
                        B.cpu_matrix == nullptr ? m_arena->downloadMatrix(B.gpu_block_index) : *B.cpu_matrix);
                }

                m_arena->resetTemporaryBlocks();

                if (needed_blocks > m_arena->emptyTemporaryBlockCount()) {
                    return m_cpu_solver.multiply(*A.cpu_matrix, *B.cpu_matrix);
                }
            }

            const CudaOperatorDescriptor* opA_ptr = nullptr;

            if (A.cpu_matrix == nullptr) {
                opA_ptr = m_arena->getDescriptor(A.gpu_block_index);
            }
            else {  
                bool cached = m_arena->uploadAndCache(A.mesh_id, A.version, A.type, *A.cpu_matrix);
                if (!cached) {
                    if (B.cpu_matrix == nullptr) {
                        return m_cpu_solver.multiply(*A.cpu_matrix, m_arena->downloadMatrix(B.gpu_block_index));
                    }
                    return m_cpu_solver.multiply(*A.cpu_matrix, *B.cpu_matrix);
                }
                opA_ptr = m_arena->getDescriptor(A.mesh_id, A.type);
            }

            const CudaOperatorDescriptor* opB_ptr = nullptr;

            if (B.cpu_matrix == nullptr) {
                opB_ptr = m_arena->getDescriptor(B.gpu_block_index);
            }
            else {
                bool cached = m_arena->uploadAndCache(B.mesh_id, B.version, B.type, *B.cpu_matrix);
                if (!cached) {
                    if (A.cpu_matrix == nullptr) {
                        return m_cpu_solver.multiply(m_arena->downloadMatrix(A.gpu_block_index), *B.cpu_matrix);
                    }
                    return m_cpu_solver.multiply(*A.cpu_matrix, *B.cpu_matrix);
                }
                opB_ptr = m_arena->getDescriptor(B.mesh_id, B.type);
            }

            auto result = m_gpu_solver->multiply(*opA_ptr, *opB_ptr, save, Target.mesh_id, Target.version, Target.type);

            if (result.status == MathStatus::SUCCESS) {

                Target.block_index = result.resultBlock;
                
                if (Target.is_intermediate) {
                    return std::nullopt;
                }
                else {
                    return m_arena->downloadMatrix(result.resultBlock);
                }
            }

            if (result.status == MathStatus::INVALID_DIMENSIONS) {
                std::cerr << "[ComputeEngine] Math failed (""INVALID_DIMENSIONS""). \n";
                return std::nullopt;
            }

            std::cerr << "[ComputeEngine] GPU Math failed (Code: " << (int)result.status << "). Triggering CPU Rescue Pipeline!\n";

            CPUSparseMatrix fallback_A;
            const CPUSparseMatrix* ptr_A = A.cpu_matrix;
            if (!ptr_A) {
                if (A.gpu_block_index == -1) return std::nullopt;
                fallback_A = m_arena->downloadMatrix(A.gpu_block_index);
                ptr_A = &fallback_A;
            }

            CPUSparseMatrix fallback_B;
            const CPUSparseMatrix* ptr_B = B.cpu_matrix;
            if (!ptr_B) {
                if (B.gpu_block_index == -1) return std::nullopt;
                fallback_B = m_arena->downloadMatrix(B.gpu_block_index);
                ptr_B = &fallback_B;
            }

            return m_cpu_solver.multiply(*ptr_A, *ptr_B);
        }

    }
}
