#pragma once

#include <cuda_runtime.h>
#pragma warning(push)
#pragma warning(disable : 4996) 
#include <cusparse.h>
#pragma warning(pop)            
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <MyMesh/Math/CPU/CPUSolverInternals.h>

namespace MyMesh {
    namespace MathInternal {
        
        enum class OperatorType {
            D0, D1, STAR0, STAR1, STAR2, OTHER, MASS_MATRIX, LAPLACIAN
        };

        constexpr OperatorType AllTypes[] = {
            OperatorType::D0,
            OperatorType::D1,
            OperatorType::STAR0,
            OperatorType::STAR1,
            OperatorType::STAR2,
            OperatorType::OTHER,
            OperatorType::MASS_MATRIX,
            OperatorType::LAPLACIAN
        };

        enum class MathStatus {
            SUCCESS,
            OUT_OF_MEMORY_VRAM, 
            INVALID_DIMENSIONS, 
            HARDWARE_ERROR
        };

        struct CacheKey {
            uint64_t mesh_id;
            OperatorType type;

            bool operator==(const CacheKey& other) const {
                return (mesh_id == other.mesh_id) && (type == other.type);
            }
        };

        struct CacheKeyHash {
            std::size_t operator()(const CacheKey& k) const {
                return std::hash<uint64_t>()(k.mesh_id) ^ (std::hash<int>()(static_cast<int>(k.type)) << 1);
            }
        };

        struct CudaMemoryBlock {
            bool is_free = true;
            bool is_scratchpad = false;
            size_t capacity_bytes = 0;
            void* d_raw_memory = nullptr;
            CudaOperatorDescriptor current_descriptor;
            uint64_t current_version = 0;
            std::list<CacheKey>::iterator timeline_iterator;
        };

        struct CudaOperatorDescriptor {
            cusparseSpMatDescr_t descriptor = nullptr;
            int rows = 0;
            int cols = 0;
            int nnz = 0;
        };

        using CudaMemoryPool = std::vector<CudaMemoryBlock>;
        using CudaMatrixCache = std::unordered_map<CacheKey, int, CacheKeyHash>;
        using CacheManeger = std::list<CacheKey>;
        using BlocksStack = std::vector<int>;
        using BlockCounterType = int;
    }
}