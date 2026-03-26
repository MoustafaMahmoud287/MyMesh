#pragma once

#include <Eigen/Sparse>
#include <vector>
#include "Mesh.h" 

template<typename T>
using SparseMatrix = Eigen::SparseMatrix<T>;

namespace MyMesh {
    namespace MathInternal {
        class CPUSolver {
        public:
            CPUSolver() = default;
            ~CPUSolver() = default;

            // 1. TOPOLOGICAL OPERATORS (Returns raw Eigen Matrices)
            SparseMatrix<float> buildD0(const Geometry::Mesh& mesh);
            SparseMatrix<float> buildD1(const Geometry::Mesh& mesh);

            // 2. METRIC OPERATORS
            SparseMatrix<float> buildMassMatrix(const Geometry::Mesh& mesh);
            SparseMatrix<float> buildCotangentLaplacian(const Geometry::Mesh& mesh);

            // 3. CORE MATH (Only takes and returns pure Eigen / std::vector types!)
            SparseMatrix<float> multiply(const Eigen::SparseMatrix<float>& A,
                const Eigen::SparseMatrix<float>& B);

            std::vector<float> multiply(const Eigen::SparseMatrix<float>& A,
                const std::vector<float>& x);

            std::vector<float> solveLinearSystem(const Eigen::SparseMatrix<float>& A,
                const std::vector<float>& b);

        private:
            std::vector<float> computeEdgeLengths(const Geometry::Mesh& mesh);
            std::vector<float> computeFaceAreas(const Geometry::Mesh& mesh);
        };
    }
}