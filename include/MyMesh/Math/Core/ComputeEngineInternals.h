#pragma once

#include <MyMesh/Math/CPU/CPUBasicMathSolver.h>
#include <MyMesh/Math/GPU/GPUBasicMathSolver.h>
#include <MyMesh/Hardware/HardwareProfiler.h>
#include <memory>

namespace MyMesh {
    namespace MathInternal {

        enum class Strategy {
            FIXED_BLOCK_SIZE,
            FIXED_BLOCK_COUNT
        };

        struct ComputeOperand {
            const CPUSparseMatrix* cpu_matrix;
            uint64_t mesh_id;
            OperatorType type;
            uint32_t version;
            int gpu_block_index;

            ComputeOperand(const CPUSparseMatrix& mat)
                : cpu_matrix(&mat), mesh_id(TRANSIENT_ID), type(OperatorType::OTHER), version(0), gpu_block_index(-1) {
            }

            ComputeOperand(const CPUSparseMatrix& mat, uint64_t id, OperatorType t, uint32_t v = 0)
                : cpu_matrix(&mat), mesh_id(id), type(t), version(v), gpu_block_index(-1) {
            }

            ComputeOperand(int block_idx)
                : cpu_matrix(nullptr), mesh_id(TRANSIENT_ID), type(OperatorType::OTHER), version(0), gpu_block_index(block_idx) {
            }
        };

        struct ComputeTarget {
            BlockCounterType block_index;
            uint64_t mesh_id;
            OperatorType type;
            uint32_t version;
            bool is_intermediate;

            ComputeTarget()
                : block_index(-1), mesh_id(TRANSIENT_ID), type(OperatorType::OTHER), version(0), is_intermediate(false){
            }

            ComputeTarget(uint64_t id, OperatorType t, uint32_t v, bool intr)
                : block_index(-1), mesh_id(id), type(t), version(v), is_intermediate(intr){
            }
        };

        const BlockCounterType MIN_BLOCKS_NUM = 3;
        const int DEFAULT_ID = 0;
        const int BLOCKS_DIVIDE_THSHOLD = 10;

    }
}