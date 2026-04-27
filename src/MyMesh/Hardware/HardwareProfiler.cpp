#pragma once

#include <MyMesh/Hardware/HardwareProfiler.h>
#include <iostream>
#include <fstream>
#include <filesystem> 
#include <algorithm>
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
            LimitsPrams prams;

            cudaGetDeviceProperties(&props, device_id);
            auto avalibale_memory = (props.totalGlobalMem > prams.pc_safty_vram) ? props.totalGlobalMem - prams.pc_safty_vram : props.totalGlobalMem / 2;
            limits.MinSizeToRunGPU = prams.min_size;
            limits.TestBlockSize = std::min(prams.testing_block_default_size,
                avalibale_memory / prams.num_of_total_dummy_blocks);
            limits.MaxSizeToRunGPU = limits.TestBlockSize;

            try {

                std::cout << "\n[HardwareProfiler] Running initial hardware calibration. This may take a few seconds...\n";

                MyMesh::MathInternal::CPUSolver cpu_solver;
                MyMesh::MathInternal::CudaMemoryArena temp_arena(prams.num_of_dummy_persistent_blocks,
                    prams.num_of_dummy_temp_blocks, 
                    limits.TestBlockSize
                );

                MyMesh::MathInternal::CudaSolver gpu_solver(&temp_arena);

                auto high = prams.max_dim;
                auto low = prams.min_dim;

                while (low <= high) {

                    int mid = low + (high - low) / 2;

                    size_t matrix_size = 0;
                    int gpu_failed_count = 0; 
                    int nnz = 0;
                    double total_cpu_ms = 0;
                    double total_gpu_ms = 0;
                    double avg_cpu_ms;
                    double avg_gpu_ms;
                    
                    for (int i = 0; i < prams.number_of_rounds; ++i) {

                        MyMesh::MathInternal::CPUSparseMatrix A = MyMesh::MathInternal::generateRandomSparseMatrix(mid, mid, prams.sparsity, 1);
                        MyMesh::MathInternal::CPUSparseMatrix B = MyMesh::MathInternal::generateRandomSparseMatrix(mid, mid, prams.sparsity, 2);

                        if (i == 0) {
                            matrix_size = (A.nonZeros() * sizeof(float)) + (A.nonZeros() * sizeof(int)) + ((A.rows() + 1) * sizeof(int));
                            nnz = A.nonZeros();
                        }

                        auto start_cpu = std::chrono::high_resolution_clock::now();
                        cpu_solver.multiply(A, B);
                        auto end_cpu = std::chrono::high_resolution_clock::now();
                        total_cpu_ms += std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();


                        auto start_gpu = std::chrono::high_resolution_clock::now();

                        temp_arena.uploadAndCache(999, 1, MyMesh::MathInternal::OperatorType::D0, A);
                        temp_arena.uploadAndCache(999, 2, MyMesh::MathInternal::OperatorType::D1, B);

                        const auto* opA = temp_arena.getDescriptor(999, MyMesh::MathInternal::OperatorType::D0);
                        const auto* opB = temp_arena.getDescriptor(999, MyMesh::MathInternal::OperatorType::D1);
                        MyMesh::MathInternal::CudaOperatorDescriptor opC; opC.is_intermediate = false;

                        MyMesh::MathInternal::MathStatus status = gpu_solver.multiply(*opA, *opB, opC, 999, MyMesh::MathInternal::OperatorType::OTHER);
                        if (status != MyMesh::MathInternal::MathStatus::SUCCESS) {
                            gpu_failed_count++;
                        }
                        else {
                            auto res = temp_arena.downloadMatrix(opC);

                            auto end_gpu = std::chrono::high_resolution_clock::now();
                            total_gpu_ms += std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();
                        }

                        temp_arena.evictMesh(999);
                    }
                    
                    avg_cpu_ms = total_cpu_ms / prams.number_of_rounds;
                    avg_gpu_ms = total_gpu_ms / (std::max(prams.number_of_rounds - gpu_failed_count, 1));
                    
                    if (gpu_failed_count < prams.number_of_rounds/2) {
                        std::cout << "  -> Testing " << mid << "x" << mid << " | CPU: " << avg_cpu_ms << " ms | GPU: " << avg_gpu_ms << " ms\n";
                    }
                    else {
                        std::cout << "  -> Testing " << mid << "x" << mid << " | GPU mostly failed";
                    }

                    if (gpu_failed_count > prams.number_of_rounds/2) {
                        high = mid - 1;
                    }

                    else if (avg_gpu_ms < avg_cpu_ms) {
                        limits.MinSizeToRunGPU = matrix_size;
                        high = mid - 1;
                    }

                    else {
                        low = mid + 1;
                    }

                    std::cout << "\n[HardwareProfiler] MinSize to run GPU is : " << limits.MinSizeToRunGPU << " Bytes\n\n" << std::endl; 
                }

            }
            catch (const std::exception& e) {
                std::cerr << "\n[FATAL CALIBRATION ERROR] " << e.what() << "\nUsing safe fallback defaults.\n";
            }
            catch (...) {
                std::cerr << "\n[FATAL CALIBRATION ERROR] Unknown C++ Exception thrown!\nUsing safe fallback defaults.\n";
            }

            return limits;
        }
    }
}