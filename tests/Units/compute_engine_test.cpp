#pragma warning(disable : 4996)

#include <iostream>
#include <vector>
#include <MyMesh/Math/Core/ComputeEngine.h>

using namespace MyMesh;
using namespace MyMesh::MathInternal;

int main() {
    std::cout << "=== ComputeEngine Final Integration Test ===\n\n";

    try {
        // ==========================================
        // STAGE 1: BOOT THE ENGINE
        // ==========================================
        std::cout << "[1] Booting ComputeEngine...\n";
        // Strategy: FIXED_BLOCK_COUNT, 22 Persistent, 5 Temp, Device 0
        ComputeEngine engine(Strategy::FIXED_BLOCK_SIZE, 128);

        // ==========================================
        // STAGE 2: TEST SMALL MATRIX (CPU BYPASS)
        // ==========================================
        std::cout << "\n[2] Executing Small Matrix Math (Should Trigger CPU Bypass)...\n";

        CPUSparseMatrix small_A(3, 3);
        small_A.insert(0, 0) = 1.0f; small_A.insert(1, 1) = 1.0f; small_A.insert(2, 2) = 1.0f;
        small_A.makeCompressed();

        CPUSparseMatrix small_B(3, 3);
        small_B.insert(0, 0) = 2.0f; small_B.insert(1, 1) = 2.0f; small_B.insert(2, 2) = 2.0f;
        small_B.makeCompressed();

        // Setup Operands
        ComputeOperand op_small_A{ small_A, 999, OperatorType::MASS_MATRIX, 1};
        ComputeOperand op_small_B{ small_B, 999, OperatorType::LAPLACIAN, 1};
        ComputeTarget target_small{ 999, OperatorType::OTHER,1 , false};

        // Execute
        auto result_small = engine.multiply(op_small_A, op_small_B, target_small);

        if (result_small.has_value()) {
            std::cout << "  -> SUCCESS! Small Matrix calculated.\n";
            std::cout << "  -> Verification: C(0,0) = " << result_small.value().coeff(0, 0) << " (Expected 2.0)\n";
        }
        else {
            std::cerr << "  -> FAILED! engine.multiply returned std::nullopt for small matrix.\n";
        }

        // ==========================================
        // STAGE 3: TEST LARGE MATRIX (GPU PATH)
        // ==========================================
        std::cout << "\n[3] Executing Large Matrix Math (Should Trigger GPU cuSPARSE)...\n";

        // Make this matrix big enough to exceed the MinSizeToRunGPU hardware limit
        int large_dim = 2000;
        CPUSparseMatrix large_A(large_dim, large_dim);
        for (int i = 0; i < large_dim; i++) large_A.insert(i, i) = 2.0f;
        large_A.makeCompressed();

        CPUSparseMatrix large_B(large_dim, large_dim);
        for (int i = 0; i < large_dim; i++) large_B.insert(i, i) = 3.0f;
        large_B.makeCompressed();

        ComputeOperand op_large_A{ large_A, 888, OperatorType::MASS_MATRIX, 1 };
        ComputeOperand op_large_B{ large_B, 888, OperatorType::LAPLACIAN, 1 };

        // We mark this as intermediate=false so the GPU downloads the result for us to check
        ComputeTarget target_large{ 888, OperatorType::OTHER, 1, false};

        // Execute
        auto result_large = engine.multiply(op_large_A, op_large_B, target_large);

        if (result_large.has_value()) {
            std::cout << "  -> SUCCESS! Large Matrix calculated and downloaded.\n";
            std::cout << "  -> Verification: C(0,0) = " << result_large.value().coeff(0, 0) << " (Expected 6.0)\n";
            std::cout << "  -> Verification: C(" << large_dim - 1 << "," << large_dim - 1 << ") = "
                << result_large.value().coeff(large_dim - 1, large_dim - 1) << " (Expected 6.0)\n";
        }
        else {
            std::cerr << "  -> FAILED! engine.multiply returned std::nullopt for large matrix.\n";
        }

        std::cout << "\n=== Test Complete ===\n";

    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
    }

    return 0;
}