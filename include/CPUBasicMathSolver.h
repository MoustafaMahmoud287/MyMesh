#pragma once
#include "CPUSolverInternals.h"

namespace MyMesh {
    namespace MathInternal {
        class CPUSolver {
        public:
            CPUSolver() = default;
            ~CPUSolver() = default;

            // 1. TOPOLOGICAL OPERATORS 
            const CPUSparseMatrix& buildD0(const Geometry::Mesh& mesh);
            const CPUSparseMatrix& buildD1(const Geometry::Mesh& mesh);

            // 2. GEOMETRIC OPERATORS
            const CPUSparseMatrix& buildHodgeStar0(const Geometry::Mesh& mesh);
            const CPUSparseMatrix& buildHodgeStar1(const Geometry::Mesh& mesh);
            const CPUSparseMatrix& buildHodgeStar2(const Geometry::Mesh& mesh);

            // 3. CORE MATH (Only takes and returns pure Eigen / std::vector types!) (still not implemented yet)
            CPUSparseMatrix multiply(const CPUSparseMatrix& A, const CPUSparseMatrix& B);
            std::vector<CPUFloat> multiply(const CPUSparseMatrix& A, const std::vector<CPUFloat>& x);
            std::vector<CPUFloat> solveLinearSystem(const CPUSparseMatrix& A, const std::vector<CPUFloat>& b);

        private:

            CPUCache m_cache;
            void registerMesh(const Geometry::Mesh& mesh);
            void unregisterMesh(const Geometry::Mesh& mesh);

        };
    }
}
