#pragma once

#include <MyMesh/Hardware/HardwareProfiler.h>
#include <iostream>
#include <fstream>
#include <filesystem> 
#include <cuda_runtime_api.h>
#include <MyMesh/Math/CPU/CPUBasicMathSolver.h>
#include <MyMesh/Math/GPU/GPUBasicMathSolver.h>
#include <MyMesh/Math/Utils/MathUtils.h>

namespace MyMesh {
    namespace Hardware {

        FullGPUSpec HardwareProfiler::loadProfile(int device_id) {

            auto relative_path(CONFIG_PATH + std::to_string(device_id) + ".dat");
            std::filesystem::path config_file(relative_path);
            std::filesystem::path config_dir = config_file.parent_path();

            if (std::filesystem::exists(config_file)) {
                if (std::filesystem::file_size(config_file) == sizeof(FullGPUSpec)) {
                    
                    FullGPUSpec ans;

                    std::ifstream infile(config_file, std::ios::binary | std::ios::in);
                    infile.read(reinterpret_cast<char*>(&ans), sizeof(ans));

                    return ans;
                }
            }

            auto gpu_profile = detectAndSaveProfile(device_id);
            auto limits_profile = estimateLimits(device_id);
            FullGPUSpec ans{ gpu_profile , limits_profile };

            if (!config_dir.empty() && !std::filesystem::exists(config_dir)) {
                std::filesystem::create_directories(config_dir);
            }

            std::ofstream outfile(config_file, std::ios::binary | std::ios::out | std::ios::trunc);
            outfile.write(reinterpret_cast<const char*>(&ans), sizeof(ans));

            return ans;
        }

        GPUProfile HardwareProfiler::detectAndSaveProfile(int device_id) {

            GPUProfile profile;
            profile.is_valid = false;

            cudaDeviceProp props;
            cudaError_t err = cudaGetDeviceProperties(&props, device_id);

            if (err != cudaSuccess) {
                std::cerr << "[HardwareProfiler] CRITICAL: Failed to query GPU. Is NVIDIA driver installed?\n";
                return profile;
            }

            profile.id = device_id;
            profile.total_vram_mb = props.totalGlobalMem / (1024 * 1024);
            profile.max_threads_per_block = props.maxThreadsPerBlock;
            profile.compute_capability_major = props.major;
            profile.compute_capability_minor = props.minor;
            profile.is_valid = true;
         
            return profile;
        }


        LimitsProfile HardwareProfiler::estimateLimits(int device_id) {
         
            LimitsProfile limits;

            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device_id);

            // --- 1. THE STATIC MAX LIMIT (Safe Math, No Crashing) ---
            size_t one_gb = 1024ULL * 1024 * 1024;
            limits.MaxSizeToRunGPU = (props.totalGlobalMem > one_gb) ? props.totalGlobalMem - one_gb : props.totalGlobalMem / 2;

            // --- 2. THE EMPIRICAL MIN LIMIT (Binary Search Auto-Tuner) ---
            std::cout << "\n[HardwareProfiler] Running initial hardware calibration. This may take a few seconds...\n";

            // Setup Dummy Environment (Small 128MB Arena just for testing)
            MyMesh::MathInternal::CPUSolver cpu_solver;
            MyMesh::MathInternal::CudaMemoryArena temp_arena(2, 2, 32 * 1024 * 1024);
            MyMesh::MathInternal::CudaSolver gpu_solver(&temp_arena);

            // Binary Search Parameters
            int low_dim = 100;    // Definitely CPU territory
            int high_dim = 3000;  // Definitely GPU territory
            int best_crossover_dim = 3000;
            size_t best_crossover_bytes = 1024 * 1024; // Fallback: 1MB

            float sparsity = 0.05f; // 5% dense
            const int NUM_RUNS = 3; // Average over 3 runs to balance accuracy vs boot time

            while (low_dim <= high_dim) {
                int mid_dim = low_dim + (high_dim - low_dim) / 2;

                // Generate Test Data
                MyMesh::MathInternal::CPUSparseMatrix A = MyMesh::MathInternal::generateRandomSparseMatrix(mid_dim, mid_dim, sparsity, 1);
                MyMesh::MathInternal::CPUSparseMatrix B = MyMesh::MathInternal::generateRandomSparseMatrix(mid_dim, mid_dim, sparsity, 2);

                // --- Benchmark CPU ---
                double total_cpu_ms = 0;
                for (int i = 0; i < NUM_RUNS; ++i) {
                    auto start = std::chrono::high_resolution_clock::now();
                    cpu_solver.multiply(A, B);
                    auto end = std::chrono::high_resolution_clock::now();
                    total_cpu_ms += std::chrono::duration<double, std::milli>(end - start).count();
                }
                double avg_cpu_ms = total_cpu_ms / NUM_RUNS;

                // --- Benchmark GPU (The FULL Pipeline) ---
                double total_gpu_ms = 0;
                for (int i = 0; i < NUM_RUNS; ++i) {
                    auto start = std::chrono::high_resolution_clock::now();

                    // Upload, Math, Download
                    temp_arena.uploadAndCache(999, 1, MyMesh::MathInternal::OperatorType::D0, A);
                    temp_arena.uploadAndCache(999, 2, MyMesh::MathInternal::OperatorType::D1, B);

                    const auto* opA = temp_arena.getDescriptor(999, MyMesh::MathInternal::OperatorType::D0);
                    const auto* opB = temp_arena.getDescriptor(999, MyMesh::MathInternal::OperatorType::D1);
                    MyMesh::MathInternal::CudaOperatorDescriptor opC; opC.is_intermediate = false;

                    gpu_solver.multiply(*opA, *opB, opC, 999, MyMesh::MathInternal::OperatorType::OTHER);
                    temp_arena.downloadMatrix(opC);

                    auto end = std::chrono::high_resolution_clock::now();
                    total_gpu_ms += std::chrono::duration<double, std::milli>(end - start).count();

                    temp_arena.evictMesh(999); // Clean up for next loop
                }
                double avg_gpu_ms = total_gpu_ms / NUM_RUNS;

                std::cout << "  -> Testing " << mid_dim << "x" << mid_dim
                    << " | CPU: " << avg_cpu_ms << " ms | GPU: " << avg_gpu_ms << " ms\n";

                // --- The Decision ---
                if (avg_gpu_ms < avg_cpu_ms) {
                    // GPU won! But can we go even smaller and still win?
                    best_crossover_dim = mid_dim;

                    // Convert the winning matrix size into literal BYTES
                    best_crossover_bytes = (A.nonZeros() * sizeof(float)) +
                        (A.nonZeros() * sizeof(int)) +
                        ((A.rows() + 1) * sizeof(int));

                    high_dim = mid_dim - 1; // Search lower half
                }
                else {
                    // CPU won. The GPU is choking on the PCIe tax. Go bigger.
                    low_dim = mid_dim + 1; // Search upper half
                }
            }

            limits.MinSizeToRunGPU = best_crossover_bytes;
            std::cout << "[HardwareProfiler] Calibration complete! PCIe Crossover at: "
                << best_crossover_bytes << " Bytes (" << best_crossover_dim << "x" << best_crossover_dim << ")\n\n";

            return limits;
        }
    }
}