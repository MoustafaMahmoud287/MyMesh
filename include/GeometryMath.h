#pragma once
#include "Mesh.h"

namespace myMesh {
    namespace Math {

        float faceArea(const Geometry::Mesh& mesh, Geometry::FaceHandle face);
        float barycentricDualArea(const Geometry::Mesh& mesh, Geometry::VertexHandle vertex);

    }
}