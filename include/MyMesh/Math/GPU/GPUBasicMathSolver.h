#pragma once

#include "GPUMemoryManger.h"

    namespace MyMesh{
        namespace MathInternal {
            class CudaSolver {

            public:

                CudaSolver(CudaMemoryArena* memory_arena = nullptr);
                ~CudaSolver();

                MathStatus multiply(const CudaOperatorDescriptor& opA, const CudaOperatorDescriptor& opB, CudaOperatorDescriptor& opC, uint64_t mesh_id = 0, OperatorType type = OperatorType::OTHER);

            private:

                cusparseHandle_t m_cusparse_handle = nullptr;
                CudaMemoryArena* m_memory_arena = nullptr;
               
            };
        }
} 
