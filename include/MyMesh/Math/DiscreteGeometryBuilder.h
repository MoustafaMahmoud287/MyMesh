#pragma once
#include <MyMesh/Math/CPU/CPUSolverInternals.h>
#include <MyMesh/Core/mesh.h>
#include <MyMesh/Math/GeometryMath.h>

namespace MyMesh {
    namespace MathInternal {

        class DiscreteGeometryBuilder {
        public:
            static CPUSparseMatrix buildD0(const Geometry::Mesh& mesh);
            static CPUSparseMatrix buildD1(const Geometry::Mesh& mesh);
            static CPUSparseMatrix buildHodgeStar0(const Geometry::Mesh& mesh);
            static CPUSparseMatrix buildHodgeStar1(const Geometry::Mesh& mesh);
            static CPUSparseMatrix buildHodgeStar2(const Geometry::Mesh& mesh);
        };

    }
}