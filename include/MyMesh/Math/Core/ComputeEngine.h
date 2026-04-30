#pragma once

#include "ComputeEngineInternals.h"

namespace MyMesh {
    namespace MathInternal { 

        class ComputeEngine {
        public:

            ComputeEngine(Strategy str, int BlockSizeinMB_Or_PersistentBlockCount, int TempBlockCount = MIN_BLOCKS_NUM, int deviceID = DEFAULT_ID);
            ~ComputeEngine() = default;

            std::optional<CPUSparseMatrix> multiply(const ComputeOperand& A, const ComputeOperand& B, ComputeTarget& Target);

        private:

            Hardware::FullGPUSpec m_hardware_specs;
            bool m_gpu_enabled;

            std::unique_ptr<CudaMemoryArena> m_arena;
            std::unique_ptr<CudaSolver> m_gpu_solver;
            CPUSolver m_cpu_solver;

        };

    }
}
