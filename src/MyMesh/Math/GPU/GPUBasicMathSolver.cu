#include <MyMesh/Math/GPU/GPUBasicMathSolver.h>

namespace MyMesh {
    namespace MathInternal {

        CudaSolver::CudaSolver(CudaMemoryArena* memory_arena) : m_memory_arena(memory_arena) {

            cusparseStatus_t status = cusparseCreate(&m_cusparse_handle);
            if (status != CUSPARSE_STATUS_SUCCESS) {
                std::cerr << "[CudaSolver] CRITICAL ERROR: Failed to initialize cuSPARSE.\n";
                throw std::runtime_error("cuSPARSE initialization failed.");
            }
 
        }

        CudaSolver::~CudaSolver() {

            if (m_cusparse_handle) {
                cusparseDestroy(m_cusparse_handle);
                m_cusparse_handle = nullptr;
            }

        }


        CudaMultiplyResult CudaSolver::multiply(const CudaOperatorDescriptor& opA, const CudaOperatorDescriptor& opB, CudaSaveOptions save, uint64_t mesh_id, uint64_t version, OperatorType type)
        {
            cusparseSpMatDescr_t matA = opA.descriptor;
            cusparseSpMatDescr_t matB = opB.descriptor;

            if (!matA || !matB) {
                std::cerr << "[CudaSolver] Error: Invalid input descriptors for multiplication.\n";
                return CudaMultiplyResult{ MathStatus::HARDWARE_ERROR };
            }

            if (opA.cols != opB.rows) {
                std::cerr << "[CudaSolver] Error: INVALID_DIMENSIONS. "
                    << "Cannot multiply A (" << opA.rows << "x" << opA.cols
                    << ") with B (" << opB.rows << "x" << opB.cols << ").\n";
                return CudaMultiplyResult{ MathStatus::INVALID_DIMENSIONS };
            }

            cusparseSpGEMMDescr_t spgemmDesc;
            if (cusparseSpGEMM_createDescr(&spgemmDesc) != CUSPARSE_STATUS_SUCCESS) {
                return CudaMultiplyResult{ MathStatus::HARDWARE_ERROR };
            }

            cusparseSpMatDescr_t matC;
            cusparseCreateCsr(&matC, opA.rows, opB.cols, 0,
                nullptr, nullptr, nullptr,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

            float alpha = 1.0f;
            float beta = 0.0f;
            cusparseOperation_t opA_type = CUSPARSE_OPERATION_NON_TRANSPOSE;
            cusparseOperation_t opB_type = CUSPARSE_OPERATION_NON_TRANSPOSE;

            size_t block_size = m_memory_arena->getBlockSize();
            size_t bufferSize1 = 0;

            cusparseStatus_t stat = cusparseSpGEMM_workEstimation(
                m_cusparse_handle, opA_type, opB_type, &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, nullptr
            );

            if (stat != CUSPARSE_STATUS_SUCCESS || bufferSize1 > block_size) {
                cusparseSpGEMM_destroyDescr(spgemmDesc);
                cusparseDestroySpMat(matC);
                return CudaMultiplyResult{ MathStatus::OUT_OF_MEMORY_VRAM };
            }

            auto workspace_1 = m_memory_arena->getTemporaryBlock();

            if(workspace_1 == -1) return CudaMultiplyResult{ MathStatus::OUT_OF_MEMORY_VRAM };

            void* d_workspace1 = m_memory_arena->getRawBlockPointer(workspace_1);

            if (d_workspace1 == nullptr) {
                m_memory_arena->evictTemporaryBlock(workspace_1);
                return CudaMultiplyResult{ MathStatus::OUT_OF_MEMORY_VRAM };
            }

            stat = cusparseSpGEMM_workEstimation(
                m_cusparse_handle, opA_type, opB_type, &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, d_workspace1
            );

            cudaDeviceSynchronize();
            if (stat != CUSPARSE_STATUS_SUCCESS || cudaGetLastError() != cudaSuccess) {
                cusparseSpGEMM_destroyDescr(spgemmDesc);
                cusparseDestroySpMat(matC);
                m_memory_arena->evictTemporaryBlock(workspace_1);
                return CudaMultiplyResult{ MathStatus::HARDWARE_ERROR };
            }

            size_t bufferSize2 = 0;

            stat = cusparseSpGEMM_compute(
                m_cusparse_handle, opA_type, opB_type, &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, nullptr
            );

            if (stat != CUSPARSE_STATUS_SUCCESS || bufferSize2 > block_size) {
                cusparseSpGEMM_destroyDescr(spgemmDesc);
                cusparseDestroySpMat(matC);
                m_memory_arena->evictTemporaryBlock(workspace_1);
                return CudaMultiplyResult{ MathStatus::OUT_OF_MEMORY_VRAM };
            }

            auto workspace_2 = m_memory_arena->getTemporaryBlock();
            if (workspace_2 == -1) {
                m_memory_arena->evictTemporaryBlock(workspace_1);
                return CudaMultiplyResult{ MathStatus::OUT_OF_MEMORY_VRAM };
            }

            void* d_workspace2 = m_memory_arena->getRawBlockPointer(workspace_2);

            if (d_workspace2 == nullptr) {
                m_memory_arena->evictTemporaryBlock(workspace_1);
                m_memory_arena->evictTemporaryBlock(workspace_2);
                return CudaMultiplyResult{ MathStatus::OUT_OF_MEMORY_VRAM };
            }

            stat = cusparseSpGEMM_compute(
                m_cusparse_handle, opA_type, opB_type, &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, d_workspace2
            );

            cudaDeviceSynchronize();
            if (stat != CUSPARSE_STATUS_SUCCESS || cudaGetLastError() != cudaSuccess) {
                cusparseSpGEMM_destroyDescr(spgemmDesc);
                cusparseDestroySpMat(matC);
                m_memory_arena->evictTemporaryBlock(workspace_1);
                m_memory_arena->evictTemporaryBlock(workspace_2);
                return CudaMultiplyResult{ MathStatus::HARDWARE_ERROR };
            }

            int64_t C_rows, C_cols, C_nnz;
            cusparseSpMatGetSize(matC, &C_rows, &C_cols, &C_nnz);

            size_t C_bytes = (C_nnz * sizeof(float)) + (C_nnz * sizeof(int)) + ((C_rows + 1) * sizeof(int));

            if (C_bytes > block_size) {
                cusparseSpGEMM_destroyDescr(spgemmDesc);
                cusparseDestroySpMat(matC);
                m_memory_arena->evictTemporaryBlock(workspace_1);
                m_memory_arena->evictTemporaryBlock(workspace_2);
                return CudaMultiplyResult{ MathStatus::OUT_OF_MEMORY_VRAM };
            }

            BlockCounterType target_block_index = -1;

            if (save == CudaSaveOptions::SCRATCHPAD_BLOCK) {
                target_block_index = m_memory_arena->getTemporaryBlock();
            }
            else {
                target_block_index = m_memory_arena->allocatePersistentBlock(mesh_id, version, type);
            }

            if (target_block_index == -1) {
                cusparseSpGEMM_destroyDescr(spgemmDesc);
                cusparseDestroySpMat(matC);
                m_memory_arena->evictTemporaryBlock(workspace_1);
                m_memory_arena->evictTemporaryBlock(workspace_2);
                return CudaMultiplyResult{ MathStatus::OUT_OF_MEMORY_VRAM };
            }

            char* d_base_ptr = static_cast<char*>(m_memory_arena->getRawBlockPointer(target_block_index));

            int* d_C_row_offsets = reinterpret_cast<int*>(d_base_ptr);
            int* d_C_col_indices = reinterpret_cast<int*>(d_C_row_offsets + (C_rows + 1));
            float* d_C_values = reinterpret_cast<float*>(d_C_col_indices + C_nnz);

            cusparseCsrSetPointers(matC, d_C_row_offsets, d_C_col_indices, d_C_values);

            cusparseSpGEMM_copy(
                m_cusparse_handle, opA_type, opB_type, &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc
            );

            cusparseSpGEMM_destroyDescr(spgemmDesc);

            m_memory_arena->commitMatrixToBlock(target_block_index, matC, static_cast<int>(C_rows), static_cast<int>(C_cols), static_cast<int>(C_nnz));

            m_memory_arena->evictTemporaryBlock(workspace_1);
            m_memory_arena->evictTemporaryBlock(workspace_2);

            return CudaMultiplyResult{ MathStatus::SUCCESS, target_block_index };

        }
    } 
} 