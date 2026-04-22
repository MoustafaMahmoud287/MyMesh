#pragma once

#include "GPUSolverInternals.h"

namespace MyMesh {
    namespace MathInternal {

        class CudaMemoryArena {

        public:
            CudaMemoryArena(BlockCounterType num_persistent_blocks, BlockCounterType num_scratchpad_blocks, size_t block_size_bytes);
            ~CudaMemoryArena();

            bool hasOperator(uint64_t mesh_id, uint64_t version, OperatorType type) const;
            bool uploadAndCache(uint64_t mesh_id, uint64_t version, OperatorType type, const CPUSparseMatrix& cpu_matrix);
            BlockCounterType allocatePersistentBlock(uint64_t mesh_id, OperatorType type);
            void evictMesh(uint64_t mesh_id);

            BlockCounterType getTemporaryBlock();
            bool evictTemporaryBlock(BlockCounterType block_index);
            void resetTemporaryBlocks();

            void* getRawBlockPointer(BlockCounterType block_index) const;
            const CudaOperatorDescriptor* getDescriptor(uint64_t mesh_id, OperatorType type) const;
            size_t getBlockSize() const;

        private:

            CudaMemoryPool  m_memory_pool;
            size_t m_block_size_bytes;
            CudaMatrixCache m_matrix_cache;
            CacheManeger m_lru_block_timeline;
            BlocksStack m_free_persistent_blocks;
            BlocksStack m_free_temp_blocks;

            BlockCounterType m_num_scratchpad_blocks;
            BlockCounterType m_scratchpad_offset;

            void evictOldestBlock();
            void bindDescriptorToBlock(CudaOperatorDescriptor& desc, const CPUSparseMatrix& cpu_mat, int block_index);

        };
    
    }
}
