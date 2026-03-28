#pragma once

#include <Eigen/Sparse>
#include <vector>

#include "Mesh.h" 
#include "GeometryMath.h"

template<typename T>
using SparseMatrix = Eigen::SparseMatrix<T>;

namespace MyMesh {
    namespace MathInternal {
        class CPUSolver {
        public:
            CPUSolver() = default;
            ~CPUSolver() = default;

            // 1. TOPOLOGICAL OPERATORS 
            SparseMatrix<float> buildD0(const Geometry::Mesh& mesh);
            SparseMatrix<float> buildD1(const Geometry::Mesh& mesh);

            // 2. GEOMETRIC OPERATORS
            SparseMatrix<float> buildHodgeStar0(const Geometry::Mesh& mesh);
            SparseMatrix<float> buildHodgeStar1(const Geometry::Mesh& mesh);
            SparseMatrix<float> buildHodgeStar2(const Geometry::Mesh& mesh);

            // 3. METRIC OPERATORS
            SparseMatrix<float> buildMassMatrix(const Geometry::Mesh& mesh);
            SparseMatrix<float> buildCotangentLaplacian(const Geometry::Mesh& mesh);

            // 4. CORE MATH (Only takes and returns pure Eigen / std::vector types!)
            SparseMatrix<float> multiply(const Eigen::SparseMatrix<float>& A,
                const Eigen::SparseMatrix<float>& B);

            std::vector<float> multiply(const Eigen::SparseMatrix<float>& A,
                const std::vector<float>& x);

            std::vector<float> solveLinearSystem(const Eigen::SparseMatrix<float>& A,
                const std::vector<float>& b);
        };
    }
}