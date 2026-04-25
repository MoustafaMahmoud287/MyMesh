
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

            for (int i = num_persistent_blocks; i < total_blocks; i++) {
                m_free_temp_blocks.push_back(i);
            }
        }

        CudaMemoryArena::~CudaMemoryArena() {

            for (auto block : m_memory_pool) {
                if (block.d_raw_memory != nullptr) {
                    cudaFree(block.d_raw_memory);
                    block.d_raw_memory = nullptr;
                }
            }

        }

        bool CudaMemoryArena::hasOperator(uint64_t mesh_id, uint64_t version, OperatorType type) const {

            CacheKey key{ mesh_id, type };
            auto it = m_matrix_cache.find(key);
            if (it != m_matrix_cache.end()) { return it->second.version == version; }
            return false;

        }

        bool CudaMemoryArena::uploadAndCache(uint64_t mesh_id, uint64_t version, OperatorType type, const CPUSparseMatrix& cpu_matrix)
        {
            const_cast<CPUSparseMatrix&>(cpu_matrix).makeCompressed();

            size_t needed_bytes = (cpu_matrix.nonZeros() * sizeof(float)) + (cpu_matrix.nonZeros() * sizeof(int)) + ((cpu_matrix.rows() + 1) * sizeof(int));

            if (needed_bytes > m_block_size_bytes) {
                std::cerr << "[CudaSolver] Matrix exceeds hardware block limit. Routing to CPU.\n";
                return false;
            }

            CacheKey key{ mesh_id, type };
            auto it = m_matrix_cache.find(key);

            if (it != m_matrix_cache.end()) {
                CudaOperatorDescriptor& desc = it->second;

                if (desc.version == version) {
                    m_lru_block_timeline.splice(m_lru_block_timeline.begin(), m_lru_block_timeline, desc.timeline_iterator);
                    return true;
                }

                if (desc.descriptor != nullptr) {
                    cusparseDestroySpMat(desc.descriptor);
                }

                desc.version = version;
                desc.rows = cpu_matrix.rows();
                desc.cols = cpu_matrix.cols();
                desc.nnz = cpu_matrix.nonZeros();

                bindDescriptorToBlock(desc, cpu_matrix, desc.block_index);

                m_lru_block_timeline.splice(m_lru_block_timeline.begin(), m_lru_block_timeline, desc.timeline_iterator);

                return true;
            }

            if (m_free_persistent_blocks.empty()) {
                evictOldestBlock();
            }
            int block_index = m_free_persistent_blocks.back();
            m_free_persistent_blocks.pop_back();
            m_memory_pool[block_index].is_free = false;

            CudaOperatorDescriptor new_desc;
            new_desc.block_index = block_index;
            new_desc.version = version;
            new_desc.rows = cpu_matrix.rows();
            new_desc.cols = cpu_matrix.cols();
            new_desc.nnz = cpu_matrix.nonZeros();

            bindDescriptorToBlock(new_desc, cpu_matrix, block_index);

            m_lru_block_timeline.push_front(key);
            new_desc.timeline_iterator = m_lru_block_timeline.begin();
            m_matrix_cache[key] = new_desc;

            return true;
        }

        BlockCounterType CudaMemoryArena::allocatePersistentBlock(uint64_t mesh_id, OperatorType type)
        {
            CacheKey key{ mesh_id, type };
            auto it = m_matrix_cache.find(key);

            if (it != m_matrix_cache.end()) {
                CudaOperatorDescriptor& desc = it->second;
                if (desc.descriptor != nullptr) {
                    cusparseDestroySpMat(desc.descriptor);
                    desc.descriptor = nullptr; 
                }

                m_lru_block_timeline.splice(m_lru_block_timeline.begin(), m_lru_block_timeline, desc.timeline_iterator);
                return desc.block_index;
            }

            if (m_free_persistent_blocks.empty()) {
                evictOldestBlock();
            }

            auto block_index = m_free_persistent_blocks.back();
            m_free_persistent_blocks.pop_back();
            m_memory_pool[block_index].is_free = false;

            CudaOperatorDescriptor empty_desc;
            empty_desc.block_index = block_index;

            m_lru_block_timeline.push_front(key);
            empty_desc.timeline_iterator = m_lru_block_timeline.begin();
            m_matrix_cache[key] = empty_desc;

            return block_index;
        }

        void CudaMemoryArena::evictMesh(uint64_t mesh_id) {

            for (OperatorType type : AllTypes) {

                CacheKey key{ mesh_id, type };
                auto it = m_matrix_cache.find(key);

                if (it != m_matrix_cache.end()) {

                    auto& desc = it->second;
                    if (desc.descriptor != nullptr) {
                        cusparseDestroySpMat(desc.descriptor);
                    }

                    m_lru_block_timeline.erase(desc.timeline_iterator);

                    auto block_index = desc.block_index;
                    m_memory_pool[block_index].is_free = true;
                    m_free_persistent_blocks.push_back(block_index);

                    m_matrix_cache.erase(it);
                }
            }
        }

        BlockCounterType CudaMemoryArena::getTemporaryBlock() {
            if (m_free_temp_blocks.empty()) return -1;

            BlockCounterType block = m_free_temp_blocks.back();
            m_free_temp_blocks.pop_back();

            m_memory_pool[block].is_free = false;
            return block;
        }

        bool CudaMemoryArena::evictTemporaryBlock(BlockCounterType block_index) {
            
            if (m_memory_pool[block_index].is_free) return false;

            m_memory_pool[block_index].is_free = true;
            m_free_temp_blocks.push_back(block_index); 

            return true;
        }

        void CudaMemoryArena::resetTemporaryBlocks() {
            
            m_free_temp_blocks.clear();

            auto total_blocks = m_scratchpad_offset + m_num_scratchpad_blocks;

            
            for (int i = m_scratchpad_offset; i < total_blocks; i++) {
                m_memory_pool[i].is_free = true;
                m_free_temp_blocks.push_back(i);
            }
        }

        void* CudaMemoryArena::getRawBlockPointer(BlockCounterType block_index) const {
            return m_memory_pool[block_index].d_raw_memory;
        }

        const CudaOperatorDescriptor* CudaMemoryArena::getDescriptor(uint64_t mesh_id, OperatorType type) const {
            CacheKey key{ mesh_id, type };
            auto it = m_matrix_cache.find(key);

            if (it != m_matrix_cache.end()) {
                return &(it->second); 
            }

            return nullptr; 
        }

        size_t CudaMemoryArena::getBlockSize()  const {
            return m_block_size_bytes;
        }

        void CudaMemoryArena::evictOldestBlock() {

            if (m_lru_block_timeline.empty()) return;

            auto key = m_lru_block_timeline.back();
            auto it = m_matrix_cache.find(key);

            if (it != m_matrix_cache.end()) {

                auto& desc = it->second;

                if (desc.descriptor != nullptr) {
                    cusparseDestroySpMat(desc.descriptor);
                }

                auto block_index = desc.block_index;
                m_memory_pool[block_index].is_free = true;
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

        CPUSparseMatrix CudaMemoryArena::downloadMatrix(const CudaOperatorDescriptor& desc) const {

            if (desc.block_index < 0 || m_memory_pool[desc.block_index].d_raw_memory == nullptr) {
                throw std::runtime_error("[CudaMemoryArena] Attempted to download an invalid GPU matrix.");
            }

            size_t row_bytes = (desc.rows + 1) * sizeof(int);
            size_t col_bytes = desc.nnz * sizeof(int);
            size_t val_bytes = desc.nnz * sizeof(float);

            char* d_base_ptr = static_cast<char*>(m_memory_pool[desc.block_index].d_raw_memory);
            int* d_row_offsets = reinterpret_cast<int*>(d_base_ptr);
            int* d_col_indices = reinterpret_cast<int*>(d_base_ptr + row_bytes);
            float* d_values = reinterpret_cast<float*>(d_base_ptr + row_bytes + col_bytes);

            std::vector<int> h_rows(desc.rows + 1);
            std::vector<int> h_cols(desc.nnz);
            std::vector<float> h_vals(desc.nnz);

            cudaMemcpy(h_rows.data(), d_row_offsets, row_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_cols.data(), d_col_indices, col_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_vals.data(), d_values, val_bytes, cudaMemcpyDeviceToHost);

            CPUMappedSparseMatrix mapped_mat(
                desc.rows, desc.cols, desc.nnz, h_rows.data(), h_cols.data(), h_vals.data()
            );

            CPUSparseMatrix final_matrix = mapped_mat;
            final_matrix.makeCompressed();

            return final_matrix;

        }

    }
}