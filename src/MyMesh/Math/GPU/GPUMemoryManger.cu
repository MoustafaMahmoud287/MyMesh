
#include <MyMesh/Math/GPU/GPUMemoryManger.h>

namespace MyMesh {
    namespace MathInternal {

        CudaMemoryArena::CudaMemoryArena(BlockCounterType num_persistent_blocks, BlockCounterType num_scratchpad_blocks, size_t block_size_bytes) 
            : m_block_size_bytes(block_size_bytes), m_num_scratchpad_blocks(num_scratchpad_blocks), m_scratchpad_offset(num_persistent_blocks)
        {

            auto total_blocks = num_persistent_blocks + num_scratchpad_blocks;
            m_memory_pool.reserve(total_blocks);


            for (auto i = 0; i < total_blocks; ++i) {
                
                CudaMemoryBlock block;
                
                block.is_free = true;
                block.capacity_bytes = block_size_bytes;
                block.is_scratchpad = (i >= num_persistent_blocks);
                block.timeline_iterator = m_lru_block_timeline.end();

                cudaError_t err = cudaMalloc(&block.d_raw_memory, block_size_bytes);

                if (err != cudaSuccess) {
                    std::cerr << "[CudaSolver] FATAL: VRAM Allocation failed on block " << i << ". Error: " << cudaGetErrorString(err) << "\n";
                    for (int j = 0; j < i; ++j) {
                        cudaFree(m_memory_pool[j].d_raw_memory);
                    }
                    throw std::runtime_error("GPU Out of Memory during boot allocation.");
                }

                m_memory_pool.push_back(block);
            }

            for (int i = num_persistent_blocks - 1; i >= 0; --i) {
                m_free_persistent_blocks.push_back(i);
            }

            m_free_temp_blocks.reserve(m_num_scratchpad_blocks);

            for (int i = total_blocks - 1; i >= num_persistent_blocks; i--) {
                m_free_temp_blocks.push_back(i);
            }
        }

        CudaMemoryArena::~CudaMemoryArena() {

            for (auto block : m_memory_pool) {

                if (block.current_descriptor.descriptor != nullptr) {
                    cusparseDestroySpMat(block.current_descriptor.descriptor);
                    block.current_descriptor.descriptor = nullptr;
                }

                if (block.d_raw_memory != nullptr) {
                    cudaFree(block.d_raw_memory);
                    block.d_raw_memory = nullptr;
                }

            }

        }

        bool CudaMemoryArena::hasOperator(uint64_t mesh_id, uint64_t version, OperatorType type) const {

            CacheKey key{ mesh_id, type };
            auto it = m_matrix_cache.find(key);
            if (it != m_matrix_cache.end()) { return m_memory_pool[it->second].current_version == version; }
            return false;

        }

        bool CudaMemoryArena::uploadAndCache(uint64_t mesh_id, uint64_t version, OperatorType type, const CPUSparseMatrix& cpu_matrix) {

            const_cast<CPUSparseMatrix&>(cpu_matrix).makeCompressed();

            size_t needed_bytes = (cpu_matrix.nonZeros() * sizeof(float)) + (cpu_matrix.nonZeros() * sizeof(int)) + ((cpu_matrix.rows() + 1) * sizeof(int));

            if (needed_bytes > m_block_size_bytes) {
                std::cerr << "[CudaSolver] Matrix exceeds hardware block limit. Routing to CPU.\n";
                return false;
            }

            CacheKey key{ mesh_id, type };
            auto it = m_matrix_cache.find(key);

            if (it != m_matrix_cache.end()) {

                auto block_index = it->second;
                auto& block = m_memory_pool[block_index];

                if (block.current_version == version) {
                    m_lru_block_timeline.splice(m_lru_block_timeline.begin(), m_lru_block_timeline, block.timeline_iterator);
                    return true;
                }

                if (block.current_descriptor.descriptor != nullptr) {
                    cusparseDestroySpMat(block.current_descriptor.descriptor);
                    block.current_descriptor.descriptor = nullptr;
                }

                block.current_version = version;
                block.current_descriptor.rows = cpu_matrix.rows();
                block.current_descriptor.cols = cpu_matrix.cols();
                block.current_descriptor.nnz = cpu_matrix.nonZeros();

                bindDescriptorToBlock(block.current_descriptor, cpu_matrix, block_index);

                m_lru_block_timeline.splice(m_lru_block_timeline.begin(), m_lru_block_timeline, block.timeline_iterator);

                return true;
            }

            if (m_free_persistent_blocks.empty()) {
                evictOldestBlock();
            }

            auto block_index = m_free_persistent_blocks.back();
            auto& block = m_memory_pool[block_index];
            m_free_persistent_blocks.pop_back();

            block.is_free = false;
            block.current_version = version;
            block.current_descriptor.rows = cpu_matrix.rows();
            block.current_descriptor.cols = cpu_matrix.cols();
            block.current_descriptor.nnz = cpu_matrix.nonZeros();

            bindDescriptorToBlock(block.current_descriptor, cpu_matrix, block_index);

            m_lru_block_timeline.push_front(key);
            block.timeline_iterator = m_lru_block_timeline.begin();
            m_matrix_cache[key] = block_index;

            return true;
        }

        BlockCounterType CudaMemoryArena::allocatePersistentBlock(uint64_t mesh_id, uint64_t version, OperatorType type)
        {
            // used with multiply so always ovewrite the old result so we dont check version
            CacheKey key{ mesh_id, type };
            auto it = m_matrix_cache.find(key);

            if (it != m_matrix_cache.end()) {
                auto block_index = it->second;
                auto& block = m_memory_pool[block_index];

                if (block.current_descriptor.descriptor != nullptr) {
                    cusparseDestroySpMat(block.current_descriptor.descriptor);
                    block.current_descriptor.descriptor = nullptr;
                    block.current_descriptor.cols = 0;
                    block.current_descriptor.rows = 0;
                    block.current_descriptor.nnz = 0;
                }

                m_lru_block_timeline.splice(m_lru_block_timeline.begin(), m_lru_block_timeline, block.timeline_iterator);
                block.current_version = version;

                return block_index;
            }

            if (m_free_persistent_blocks.empty()) {
                evictOldestBlock();
                if (m_free_persistent_blocks.empty()) {
                    return -1;
                }
            }

            auto block_index = m_free_persistent_blocks.back();
            auto& block = m_memory_pool[block_index];

            m_free_persistent_blocks.pop_back();
            block.is_free = false;
            block.current_version = version;

            m_lru_block_timeline.push_front(key);
            block.timeline_iterator = m_lru_block_timeline.begin();
            m_matrix_cache[key] = block_index;

            return block_index;
        }

        void CudaMemoryArena::evictMesh(uint64_t mesh_id) {

            for (OperatorType type : AllTypes) {

                CacheKey key{ mesh_id, type };
                auto it = m_matrix_cache.find(key);

                if (it != m_matrix_cache.end()) {

                    auto block_index = it->second;
                    auto& block = m_memory_pool[block_index];

                    if (block.current_descriptor.descriptor != nullptr) {
                        cusparseDestroySpMat(block.current_descriptor.descriptor);
                        block.current_descriptor.descriptor = nullptr;
                    }

                    m_lru_block_timeline.erase(block.timeline_iterator);
                    block.timeline_iterator = m_lru_block_timeline.end();
                    block.is_free = true;
                    m_free_persistent_blocks.push_back(block_index);

                    m_matrix_cache.erase(it);
                }
            }
        }

        BlockCounterType CudaMemoryArena::getTemporaryBlock() {

            if (m_free_temp_blocks.empty()) return -1;

            BlockCounterType block_index = m_free_temp_blocks.back();
            m_free_temp_blocks.pop_back();

            auto& block = m_memory_pool[block_index];
            block.is_free = false;

            if (block.current_descriptor.descriptor != nullptr) {
                cusparseDestroySpMat(block.current_descriptor.descriptor);
                block.current_descriptor.descriptor = nullptr;
                block.current_descriptor.rows = 0;
                block.current_descriptor.cols = 0;
                block.current_descriptor.nnz = 0;
            }

            return block_index;
        }

        BlockCounterType CudaMemoryArena::emptyTemporaryBlockCount() {
            return m_free_temp_blocks.size();
        }

        bool CudaMemoryArena::evictTemporaryBlock(BlockCounterType block_index) {
            
            auto& block = m_memory_pool[block_index];

            if (block.is_free) return false;

            block.is_free = true;

            if (block.current_descriptor.descriptor != nullptr) {
                cusparseDestroySpMat(block.current_descriptor.descriptor);
                block.current_descriptor.descriptor = nullptr;
                block.current_descriptor.rows = 0;
                block.current_descriptor.cols = 0;
                block.current_descriptor.nnz = 0;
            }

            m_free_temp_blocks.push_back(block_index); 

            return true;
        }

        void CudaMemoryArena::resetTemporaryBlocks() {
            
            m_free_temp_blocks.clear();

            auto total_blocks = m_scratchpad_offset + m_num_scratchpad_blocks;

            
            for (int i = total_blocks - 1; i >= m_scratchpad_offset; i--) {
                auto& block = m_memory_pool[i];
                block.is_free = true;
                if (block.current_descriptor.descriptor != nullptr) {
                    cusparseDestroySpMat(block.current_descriptor.descriptor);
                    block.current_descriptor.descriptor = nullptr;
                    block.current_descriptor.rows = 0;
                    block.current_descriptor.cols = 0;
                    block.current_descriptor.nnz = 0;
                }

                m_free_temp_blocks.push_back(i);
            }
        }

        void* CudaMemoryArena::getRawBlockPointer(BlockCounterType block_index) const {

            if (block_index < 0 || block_index >= m_memory_pool.size()) {
                return nullptr;
            }
            return m_memory_pool[block_index].d_raw_memory;

        }

        const CudaOperatorDescriptor* CudaMemoryArena::getDescriptor(uint64_t mesh_id, OperatorType type) const {

            CacheKey key{ mesh_id, type };
            auto it = m_matrix_cache.find(key);

            if (it != m_matrix_cache.end()) {
                return &m_memory_pool[it->second].current_descriptor; 
            }

            return nullptr; 
        }

        const CudaOperatorDescriptor* CudaMemoryArena::getDescriptor(BlockCounterType block_index) const {
            
            if (block_index < 0 || block_index >= m_memory_pool.size()) {
                return nullptr;
            }
            return &m_memory_pool[block_index].current_descriptor;

        }

        size_t CudaMemoryArena::getBlockSize()  const {
            return m_block_size_bytes;
        }

        void CudaMemoryArena::evictOldestBlock() {

            if (m_lru_block_timeline.empty()) return;

            auto key = m_lru_block_timeline.back();
            auto it = m_matrix_cache.find(key);

            if (it != m_matrix_cache.end()) {

                auto block_index = it->second;
                auto& block = m_memory_pool[block_index];

                if (block.current_descriptor.descriptor != nullptr) {
                    cusparseDestroySpMat(block.current_descriptor.descriptor);
                    block.current_descriptor.descriptor = nullptr;
                    block.current_descriptor.cols = 0;
                    block.current_descriptor.rows = 0;
                    block.current_descriptor.nnz = 0;
                }

                block.is_free = true;
                block.current_version = 0;
                block.timeline_iterator = m_lru_block_timeline.end();

                m_free_persistent_blocks.push_back(block_index);

                m_matrix_cache.erase(it);
            }

            m_lru_block_timeline.pop_back();
        }

        void CudaMemoryArena::bindDescriptorToBlock(MathInternal::CudaOperatorDescriptor& desc, const CPUSparseMatrix& cpu_mat, int block_index)
        {

            size_t row_bytes = (cpu_mat.rows() + 1) * sizeof(int);
            size_t col_bytes = cpu_mat.nonZeros() * sizeof(int);
            size_t val_bytes = cpu_mat.nonZeros() * sizeof(float);

            char* d_base_ptr = static_cast<char*>(m_memory_pool[block_index].d_raw_memory);

            int* d_row_offsets = reinterpret_cast<int*>(d_base_ptr);
            int* d_col_indices = reinterpret_cast<int*>(d_base_ptr + row_bytes);
            float* d_values = reinterpret_cast<float*>(d_base_ptr + row_bytes + col_bytes);

            cudaMemcpy(d_row_offsets, cpu_mat.outerIndexPtr(), row_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_col_indices, cpu_mat.innerIndexPtr(), col_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_values, cpu_mat.valuePtr(), val_bytes, cudaMemcpyHostToDevice);

            cusparseCreateCsr(&desc.descriptor,
                cpu_mat.rows(),
                cpu_mat.cols(),
                cpu_mat.nonZeros(),
                d_row_offsets,
                d_col_indices,
                d_values,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_32F);
        }

        void CudaMemoryArena::commitMatrixToBlock(BlockCounterType block_index, cusparseSpMatDescr_t new_mat, int rows, int cols, int nnz) {
            
            auto& block = m_memory_pool[block_index];

            /
            if (block.current_descriptor.descriptor != nullptr && block.current_descriptor.descriptor != new_mat) {
                cusparseDestroySpMat(block.current_descriptor.descriptor);
            }

            block.current_descriptor.descriptor = new_mat;
            block.current_descriptor.rows = rows;
            block.current_descriptor.cols = cols;
            block.current_descriptor.nnz = nnz;

        }

        CPUSparseMatrix CudaMemoryArena::downloadMatrix(BlockCounterType block_index) const {

            if (block_index < 0 || block_index >= m_memory_pool.size()) {
                throw std::runtime_error("[CudaMemoryArena] Attempted to download an out-of-bounds GPU matrix.");
            }

            const auto& block = m_memory_pool[block_index];

            if (block.d_raw_memory == nullptr || block.is_free) {
                throw std::runtime_error("[CudaMemoryArena] Attempted to download a free or unallocated GPU matrix.");
            }

            size_t row_bytes = (block.current_descriptor.rows + 1) * sizeof(int);
            size_t col_bytes = block.current_descriptor.nnz * sizeof(int);
            size_t val_bytes = block.current_descriptor.nnz * sizeof(float);

            char* d_base_ptr = static_cast<char*>(m_memory_pool[block_index].d_raw_memory);
            int* d_row_offsets = reinterpret_cast<int*>(d_base_ptr);
            int* d_col_indices = reinterpret_cast<int*>(d_base_ptr + row_bytes);
            float* d_values = reinterpret_cast<float*>(d_base_ptr + row_bytes + col_bytes);

            std::vector<int> h_rows(block.current_descriptor.rows + 1);
            std::vector<int> h_cols(block.current_descriptor.nnz);
            std::vector<float> h_vals(block.current_descriptor.nnz);

            cudaMemcpy(h_rows.data(), d_row_offsets, row_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_cols.data(), d_col_indices, col_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vals.data(), d_values, val_bytes, cudaMemcpyDeviceToHost);

            CPUMappedSparseMatrix mapped_mat(
                block.current_descriptor.rows, block.current_descriptor.cols, block.current_descriptor.nnz, h_rows.data(), h_cols.data(), h_vals.data()
            );

            CPUSparseMatrix final_matrix = mapped_mat;
            final_matrix.makeCompressed();

            return final_matrix;

        }

    }
}