#pragma once
#include "CPUSolverInternals.h"
#include <MyMesh/Core/mesh.h>

namespace MyMesh {
    namespace MathInternal {

        class CPUTopologyCache {
        public:
            CPUTopologyCache() = default;
            ~CPUTopologyCache() = default;

            void registerMesh(const Geometry::Mesh& mesh);
            void unregisterMesh(const Geometry::Mesh& mesh);

            const CPUSparseMatrix& getD0(const Geometry::Mesh& mesh);
            const CPUSparseMatrix& getD1(const Geometry::Mesh& mesh);
            const CPUSparseMatrix& getHodgeStar0(const Geometry::Mesh& mesh);
            const CPUSparseMatrix& getHodgeStar1(const Geometry::Mesh& mesh);
            const CPUSparseMatrix& getHodgeStar2(const Geometry::Mesh& mesh);

            void setD0(const Geometry::Mesh& mesh, CPUSparseMatrix&& mat);
            void setD1(const Geometry::Mesh& mesh, CPUSparseMatrix&& mat);
            void setHodgeStar0(const Geometry::Mesh& mesh, CPUSparseMatrix&& mat);
            void setHodgeStar1(const Geometry::Mesh& mesh, CPUSparseMatrix&& mat);
            void setHodgeStar2(const Geometry::Mesh& mesh, CPUSparseMatrix&& mat);

        private:
            CPUCache m_cache;
        };

    }
}