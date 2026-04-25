#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
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
        };

        struct CudaOperatorDescriptor {

            int block_index = -1; 
            uint64_t version = 0;

            cusparseSpMatDescr_t descriptor = nullptr;

            int rows = 0;
            int cols = 0;
            int nnz = 0;

            std::list<CacheKey>::iterator timeline_iterator;

            bool is_intermediate = false;

        };

        using CudaMemoryPool = std::vector<CudaMemoryBlock>;
        using CudaMatrixCache = std::unordered_map<CacheKey, MathInternal::CudaOperatorDescriptor, CacheKeyHash>;
        using CacheManeger = std::list<CacheKey>;
        using BlocksStack = std::vector<int>;
        using BlockCounterType = int;
    }
}