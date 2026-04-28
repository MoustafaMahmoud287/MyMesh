#pragma once

#include "ComputeEngineInternals.h"

namespace MyMesh {
    namespace MathInternal { 

        class ComputeEngine {
        public:

            ComputeEngine(Strategy str, int BlockSizeinMB_Or_PersistentBlockCount, int TempBlockCount = MIN_BLOCKS_NUM, int deviceID = DEFAULT_ID);
            ~ComputeEngine() = default;

            CPUSparseMatrix multiply(
                const CPUSparseMatrix& A,
                const CPUSparseMatrix& B,
                uint64_t id_A = TRANSIENT_ID,uint64_t version_A = TRANSIENT_ID, OperatorType type_A = OperatorType::OTHER,
                uint64_t id_B = TRANSIENT_ID,uint64_t version_B = TRANSIENT_ID, OperatorType type_B = OperatorType::OTHER
            );

        private:

            Hardware::FullGPUSpec m_hardware_specs;
            bool m_gpu_enabled;

            std::unique_ptr<CudaMemoryArena> m_arena;
            std::unique_ptr<CudaSolver> m_gpu_solver;
            CPUSolver m_cpu_solver;

        };

    }
}
