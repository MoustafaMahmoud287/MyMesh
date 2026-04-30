#pragma warning(disable : 4996)

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <MyMesh/Math/GPU/GPUBasicMathSolver.h>


using namespace MyMesh;
using namespace MyMesh::MathInternal;

int main() {
    std::cout << "--- MyMesh Engine Vertical Slice Test ---\n\n";

    try {

        std::cout << "[1] Booting Engine...\n";

        CudaMemoryArena arena(5, 5, 256 * 1024 * 1024);
        CudaSolver solver(&arena);


        std::cout << "[2] Generating CPU Matrices and Uploading to VRAM...\n";

        CPUSparseMatrix A(3, 3);
        A.insert(0, 0) = 1.0f; A.insert(1, 1) = 1.0f; A.insert(2, 2) = 1.0f;
        A.makeCompressed();

        CPUSparseMatrix B(3, 3);
        B.insert(0, 0) = 2.0f; B.insert(1, 1) = 2.0f; B.insert(2, 2) = 2.0f;
        B.makeCompressed();

        arena.uploadAndCache(999, 1, OperatorType::MASS_MATRIX, A);
        arena.uploadAndCache(999, 1, OperatorType::LAPLACIAN, B);

        const auto* opA = arena.getDescriptor(999, OperatorType::MASS_MATRIX);
        const auto* opB = arena.getDescriptor(999, OperatorType::LAPLACIAN);

        if (!opA || !opB) {
            std::cerr << "Failed to cache or retrieve descriptors!\n";
            return -1;
        }

     
        std::cout << "[3] Executing cuSPARSE SpGEMM (A * B = C)...\n";

        CudaMultiplyResult result = solver.multiply(
            *opA, *opB,
            CudaSaveOptions::PERSISTENT_BLOCK,
            999, 1, OperatorType::OTHER 
        );

        if (result.status != MathStatus::SUCCESS) {
            std::cerr << "GPU Math Failed with code: " << static_cast<int>(result.status) << "\n";
            return -1;
        }

        
        std::cout << "[4] Downloading Results to CPU...\n";

        CPUSparseMatrix C = arena.downloadMatrix(result.resultBlock);

        std::cout << "\nRESULT MATRIX C (Non-Zero Values): \n";
        bool passed = true;

        for (int k = 0; k < C.outerSize(); ++k) {
            for (CPUSparseMatrix::InnerIterator it(C, k); it; ++it) {
                std::cout << "C(" << it.row() << ", " << it.col() << ") = " << it.value() << "\n";
                if (it.value() != 2.0f) passed = false;
            }
        }
        std::cout << "\n";

        if (passed && C.nonZeros() == 3) {
            std::cout << "SUCCESS! Architecture is flawless.\n";
        }
        else {
            std::cout << "FAILED! Math output is incorrect.\n";
        }


    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
    }

    return 0;
}