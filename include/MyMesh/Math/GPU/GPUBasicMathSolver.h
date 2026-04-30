#pragma once

#include "GPUMemoryManger.h"

    namespace MyMesh{
        namespace MathInternal {
            class CudaSolver {

            public:

                CudaSolver(CudaMemoryArena* memory_arena = nullptr);
                ~CudaSolver();

                CudaMultiplyResult multiply(const CudaOperatorDescriptor& opA, const CudaOperatorDescriptor& opB,
                    CudaSaveOptions save = CudaSaveOptions::PERSISTENT_BLOCK,
                    uint64_t mesh_id = TRANSIENT_ID,  uint64_t version = TRANSIENT_ID,
                    OperatorType type = OperatorType::OTHER);

            private:

                cusparseHandle_t m_cusparse_handle = nullptr;
                CudaMemoryArena* m_memory_arena = nullptr;
               
            };
        }
} 
