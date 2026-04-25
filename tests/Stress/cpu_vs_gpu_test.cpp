#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

#include <MyMesh/Math/CPU/CPUSolverInternals.h>
#include <MyMesh/Math/CPU/CPUBasicMathSolver.h>
#include <MyMesh/Math/GPU/GPUMemoryManger.h>
#include <MyMesh/Math/GPU/GPUBasicMathSolver.h>

using namespace MyMesh;
using namespace MyMesh::MathInternal;

// Helper: Generate a random matrix that fits safely in the Arena
CPUSparseMatrix generateRandomSparseMatrix(int rows, int cols, float sparsity, int seed) {
    CPUSparseMatrix mat(rows, cols);
    std::vector<CPUTriplet> triplets;

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> val_dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> chance_dist(0.0f, 1.0f);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (chance_dist(gen) < sparsity) {
                triplets.push_back({ i, j, val_dist(gen) });
            }
        }
    }

    mat.setFromTriplets(triplets.begin(), triplets.end());
    mat.makeCompressed(); // CRITICAL for raw pointer extraction
    return mat;
}

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "  TITAN ENGINE - INFINITE PARITY FUZZ TESTER     " << std::endl;
    std::cout << "=================================================" << std::endl;

    try {
        CudaMemoryArena gpu_arena(22, 5, 256 * 1024 * 1024);
        CudaSolver gpu_solver(&gpu_arena);
        CPUSolver cpu_solver;

        // The maximum allowed floating point drift between CPU and GPU
        const float EPSILON = 1e-3f;

        int iteration = 0;

        // Random generator for the matrix dimensions
        std::mt19937 dim_gen(999);
        std::uniform_int_distribution<int> size_dist(100, 15000);
        std::uniform_real_distribution<float> sparsity_dist(0.01f, 0.10f); // 1% to 10% filled

        // INFINITE LOOP
        while (true) {
            iteration++;

            // 1. Generate Random Dimensions (M x K) * (K x N)
            int M = size_dist(dim_gen);
            int K = size_dist(dim_gen);
            int N = size_dist(dim_gen);
            float sparsity = sparsity_dist(dim_gen);

            // 2. Generate Matrices (Using iteration as the seed for absolute reproducibility!)
            CPUSparseMatrix cpu_A = generateRandomSparseMatrix(M, K, sparsity, iteration * 2);
            CPUSparseMatrix cpu_B = generateRandomSparseMatrix(K, N, sparsity, (iteration * 2) + 1);

            // 3. CPU Math (Ground Truth)
            CPUSparseMatrix cpu_result = cpu_solver.multiply(cpu_A, cpu_B);

            // 4. GPU Upload
            uint64_t dummy_id = iteration; // Use iteration as mesh_id to avoid cache collisions

            bool uploaded_A = gpu_arena.uploadAndCache(dummy_id, 1, OperatorType::D0, cpu_A);
            bool uploaded_B = gpu_arena.uploadAndCache(dummy_id, 2, OperatorType::D1, cpu_B);

            if (!uploaded_A || !uploaded_B) {
                std::cout << "[ITER " << iteration << "] Skipped: Matrices exceeded VRAM limits.\n";
                continue;
            }

            const CudaOperatorDescriptor* opA = gpu_arena.getDescriptor(dummy_id, OperatorType::D0);
            const CudaOperatorDescriptor* opB = gpu_arena.getDescriptor(dummy_id, OperatorType::D1);

            CudaOperatorDescriptor opC;
            opC.is_intermediate = false;

            // 5. GPU Math
            MathStatus status = gpu_solver.multiply(*opA, *opB, opC, dummy_id, OperatorType::OTHER);

            // THE USER REQUEST: Only verify if the GPU actually returned SUCCESS!
            if (status != MathStatus::SUCCESS) {
                std::cout << "[ITER " << iteration << "] Skipped: GPU Solver returned non-success flag.\n";
                // Cleanup the cache to prevent memory leaks from skipped iterations
                gpu_arena.evictMesh(dummy_id);
                continue;
            }

            // 6. Download and Parity Check
            CPUSparseMatrix gpu_result = gpu_arena.downloadMatrix(opC);

            // If non-zeros mismatch, it's an instant catastrophic failure
            if (cpu_result.nonZeros() != gpu_result.nonZeros()) {
                std::cout << "\n[CRITICAL FAILURE] Iteration " << iteration << "\n";
                std::cout << "-> CPU Non-Zeros: " << cpu_result.nonZeros() << "\n";
                std::cout << "-> GPU Non-Zeros: " << gpu_result.nonZeros() << "\n";
                break; // Stop the loop!
            }

            // Float Margin Check
            CPUSparseMatrix diff = cpu_result - gpu_result;
            float max_error = diff.coeffs().cwiseAbs().maxCoeff();

            if (max_error > EPSILON) {
                std::cout << "\n[CRITICAL FAILURE] Iteration " << iteration << "\n";
                std::cout << "-> Mathematical drift exceeded EPSILON margin!\n";
                std::cout << "-> Max Error: " << max_error << "\n";
                break; // Stop the loop!
            }

            // If we made it here, the test passed!
            std::cout << "[ITER " << iteration << "] PASS -> " << M << "x" << K << " * " << K << "x" << N << " (Max Float Error: " << max_error << ")\n";

            // Critical: Evict the mesh at the end of the loop so our GPU memory doesn't fill up permanently!
            gpu_arena.evictMesh(dummy_id);
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
    }

    return 0;
}