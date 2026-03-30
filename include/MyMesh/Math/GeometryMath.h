#pragma once
#include <MyMesh/Core/mesh.h>

namespace MyMesh {
    namespace Math {

        float faceArea(const Geometry::Mesh& mesh, Geometry::FaceHandle face);
        float barycentricDualArea(const Geometry::Mesh& mesh, Geometry::VertexHandle vertex);
        float cotan(const Geometry::Mesh& mesh, Geometry::HalfEdgeHandle halfedge);

    }
}