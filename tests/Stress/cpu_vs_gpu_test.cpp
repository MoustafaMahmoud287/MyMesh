#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <chrono> // <-- ADDED FOR TIMING

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
        CudaMemoryArena gpu_arena(11, 5, 256 * 1024 * 1024);
        CudaSolver gpu_solver(&gpu_arena);
        CPUSolver cpu_solver;

        // The maximum allowed floating point drift between CPU and GPU
        const float EPSILON = 1e-3f;

        int iteration = 0;

        // Random generator for the matrix dimensions
        std::mt19937 dim_gen(999);
        std::uniform_int_distribution<int> size_dist(100, 5000);
        std::uniform_real_distribution<float> sparsity_dist(0.01f, 0.10f); // 1% to 10% filled

        // INFINITE LOOP
        // INFINITE LOOP
        while (true) {
            iteration++;

            // --- 1. START TOTAL PIPELINE TIMER ---
            auto start_pipeline = std::chrono::high_resolution_clock::now();

            int M = size_dist(dim_gen);
            int K = size_dist(dim_gen);
            int N = size_dist(dim_gen);
            float sparsity = sparsity_dist(dim_gen);

            CPUSparseMatrix cpu_A = generateRandomSparseMatrix(M, K, sparsity, iteration * 2);
            CPUSparseMatrix cpu_B = generateRandomSparseMatrix(K, N, sparsity, (iteration * 2) + 1);

            // --- 2. THE CPU MATH TIMER ---
            auto start_cpu = std::chrono::high_resolution_clock::now();

            CPUSparseMatrix cpu_result = cpu_solver.multiply(cpu_A, cpu_B).value_or(CPUSparseMatrix());

            auto end_cpu = std::chrono::high_resolution_clock::now();
            double pure_cpu_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
            // --------------------------------------------

            uint64_t dummy_id = iteration;

            bool uploaded_A = gpu_arena.uploadAndCache(dummy_id, 1, OperatorType::D0, cpu_A);
            bool uploaded_B = gpu_arena.uploadAndCache(dummy_id, 2, OperatorType::D1, cpu_B);

            if (!uploaded_A || !uploaded_B) {
                std::cout << "[ITER " << iteration << "] Skipped: Matrices exceeded VRAM limits.\n";
                continue;
            }

            const CudaOperatorDescriptor* opA = gpu_arena.getDescriptor(dummy_id, OperatorType::D0);
            const CudaOperatorDescriptor* opB = gpu_arena.getDescriptor(dummy_id, OperatorType::D1);

            

            // --- 3. THE GPU MATH TIMER ---
            auto start_gpu = std::chrono::high_resolution_clock::now();

            auto result = gpu_solver.multiply(*opA, *opB, MyMesh::MathInternal::CudaSaveOptions::PERSISTENT_BLOCK, dummy_id,0, OperatorType::OTHER);

            auto end_gpu = std::chrono::high_resolution_clock::now();
            double gpu_math_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
            // ----------------------------------------------------------

            if (result.status != MathStatus::SUCCESS) {
                std::cout << "[ITER " << iteration << "] Skipped: cuSPARSE aborted (Memory/Hardware limit).\n";
                gpu_arena.evictMesh(dummy_id);
                continue;
            }

            CPUSparseMatrix gpu_result = gpu_arena.downloadMatrix(result.resultBlock);

            if (cpu_result.nonZeros() != gpu_result.nonZeros()) {
                std::cout << "\n[CRITICAL FAILURE] Iteration " << iteration << "\n";
                std::cout << "-> CPU Non-Zeros: " << cpu_result.nonZeros() << "\n";
                std::cout << "-> GPU Non-Zeros: " << gpu_result.nonZeros() << "\n";
                break;
            }

            CPUSparseMatrix diff = cpu_result - gpu_result;
            float max_error = diff.coeffs().cwiseAbs().maxCoeff();

            if (max_error > EPSILON) {
                std::cout << "\n[CRITICAL FAILURE] Iteration " << iteration << "\n";
                std::cout << "-> Max Error: " << max_error << "\n";
                break;
            }

            // --- 4. STOP TOTAL PIPELINE TIMER ---
            auto end_pipeline = std::chrono::high_resolution_clock::now();
            double pipeline_ms = std::chrono::duration<double, std::milli>(end_pipeline - start_pipeline).count();

            // --- 5. PRINT THE BATTLE ---
            std::cout << "[ITER " << iteration << "] PASS -> " << M << "x" << K << " * " << K << "x" << N
                << "\n    -> CPU Time : " << pure_cpu_ms << " ms"
                << "\n    -> GPU Time : " << gpu_math_ms << " ms"
                << "\n    -> Pipeline : " << pipeline_ms << " ms\n";

            gpu_arena.evictMesh(dummy_id);
        }

    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
    }

    return 0;
}