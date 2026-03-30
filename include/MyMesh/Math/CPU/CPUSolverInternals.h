#pragma once

#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>

#include <MyMesh/Core/mesh.h> 
#include <MyMesh/Math/GeometryMath.h>

namespace MyMesh {
    namespace MathInternal {

        using CPUFloat = float;
        using CPUSparseMatrix = Eigen::SparseMatrix<CPUFloat>;
        using CPUTriplet = Eigen::Triplet<CPUFloat>;

        struct CPUSparseMatrixCache {

            uint64_t d0_version = 0;
            uint64_t d1_version = 0;

            uint64_t star0_version = 0;
            uint64_t star1_version = 0;
            uint64_t star2_version = 0;

            CPUSparseMatrix d0;
            CPUSparseMatrix d1;
            CPUSparseMatrix star0;
            CPUSparseMatrix star1;
            CPUSparseMatrix star2;
        };

        using CPUCache = std::unordered_map<uint64_t, CPUSparseMatrixCache>;
    }
}