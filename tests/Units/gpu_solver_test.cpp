#pragma warning(disable : 4996)

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <MyMesh/Math/GPU/GPUBasicMathSolver.h>


using namespace MyMesh;
using namespace MyMesh::MathInternal;

// A simple helper to upload CPU CSR data into your O(1) Arena
void uploadDummyMatrix(CudaMemoryArena& arena, uint64_t mesh_id, OperatorType type,
    int rows, int cols, int nnz,
    const std::vector<int>& cpu_rows,
    const std::vector<int>& cpu_cols,
    const std::vector<float>& cpu_vals,
    CudaOperatorDescriptor& out_desc)
{
    // 1. Claim a persistent block from your Arena
    int block_id = arena.allocatePersistentBlock(mesh_id, type);
    char* d_base = static_cast<char*>(arena.getRawBlockPointer(block_id));

    // 2. Do the exact pointer math you designed!
    int* d_rows = reinterpret_cast<int*>(d_base);
    int* d_cols = reinterpret_cast<int*>(d_rows + (rows + 1));
    float* d_vals = reinterpret_cast<float*>(d_cols + nnz);

    // 3. Copy the data from the CPU to the GPU VRAM
    cudaMemcpy(d_rows, cpu_rows.data(), cpu_rows.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, cpu_cols.data(), cpu_cols.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, cpu_vals.data(), cpu_vals.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Build the cuSPARSE descriptor
    cusparseCreateCsr(&out_desc.descriptor, rows, cols, nnz,
        d_rows, d_cols, d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    out_desc.block_index = block_id;
    out_desc.rows = rows;
    out_desc.cols = cols;
    out_desc.nnz = nnz;
    out_desc.is_intermediate = false;
}

int main() {
    std::cout << "--- MyMesh Engine Vertical Slice Test ---\n\n";

    try {
        // ==========================================
        // STAGE 1: BOOT THE ENGINE
        // ==========================================
        std::cout << "[1] Booting Engine...\n";
        CudaMemoryArena arena(22, 5, 256 * 1024 * 1024);
        CudaSolver solver(&arena);

        // ==========================================
        // STAGE 2: PREPARE DUMMY DATA
        // ==========================================
        std::cout << "[2] Uploading Dummy Matrices to VRAM...\n";

        // Matrix A: 3x3 Identity Matrix [1, 1, 1]
        std::vector<int> A_rows = { 0, 1, 2, 3 };
        std::vector<int> A_cols = { 0, 1, 2 };
        std::vector<float> A_vals = { 1.0f, 1.0f, 1.0f };
        CudaOperatorDescriptor opA;
        uploadDummyMatrix(arena, 999, OperatorType::MASS_MATRIX, 3, 3, 3, A_rows, A_cols, A_vals, opA);

        // Matrix B: 3x3 Matrix with 2s on diagonal [2, 2, 2]
        std::vector<int> B_rows = { 0, 1, 2, 3 };
        std::vector<int> B_cols = { 0, 1, 2 };
        std::vector<float> B_vals = { 2.0f, 2.0f, 2.0f };
        CudaOperatorDescriptor opB;
        uploadDummyMatrix(arena, 999, OperatorType::LAPLACIAN, 3, 3, 3, B_rows, B_cols, B_vals, opB);

        // ==========================================
        // STAGE 3: EXECUTE GPU MATH
        // ==========================================
        std::cout << "[3] Executing cuSPARSE SpGEMM (A * B = C)...\n";
        CudaOperatorDescriptor opC;
        opC.is_intermediate = false; // We want to save this to persistent memory!

        // This will call your masterpiece function!
        MathStatus status = solver.multiply(opA, opB, opC, 999, OperatorType::OTHER);

        if (status != MathStatus::SUCCESS) {
            std::cerr << "Math Failed with code: " << static_cast<int>(status) << "\n";
            return -1;
        }

        // ==========================================
        // STAGE 4: VERIFY RESULTS
        // ==========================================
        std::cout << "[4] Downloading Results to CPU...\n";

        // Grab the raw pointer to wherever your solver decided to put Matrix C
        char* d_C_base = static_cast<char*>(arena.getRawBlockPointer(opC.block_index));
        int* d_C_rows = reinterpret_cast<int*>(d_C_base);
        int* d_C_cols = reinterpret_cast<int*>(d_C_rows + (opC.rows + 1));
        float* d_C_vals = reinterpret_cast<float*>(d_C_cols + opC.nnz);

        // Download the values
        std::vector<float> result_vals(opC.nnz);
        cudaMemcpy(result_vals.data(), d_C_vals, opC.nnz * sizeof(float), cudaMemcpyDeviceToHost);

        // Print them out!
        std::cout << "\nRESULT MATRIX C (Values): \n";
        for (float val : result_vals) {
            std::cout << val << " ";
        }
        std::cout << "\n\n";

        if (result_vals.size() == 3 && result_vals[0] == 2.0f) {
            std::cout << "SUCCESS! Architecture is flawless.\n";
        }
        else {
            std::cout << "FAILED! Math output is incorrect.\n";
        }

        // Cleanup the test descriptors manually (Arena cleans up its own memory)
        cusparseDestroySpMat(opA.descriptor);
        cusparseDestroySpMat(opB.descriptor);
        cusparseDestroySpMat(opC.descriptor);

    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
    }

    return 0;
}