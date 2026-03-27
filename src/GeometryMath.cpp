#include "GeometryMath.h"

namespace myMesh {
    namespace Math {

        float faceArea(const Geometry::Mesh& mesh, Geometry::FaceHandle face) {
            
            auto it = mesh.surroundingVertices(face).begin();

            Point vertex1position = mesh.getVertexCopy(*it); ++it;
            Point vertex2position = mesh.getVertexCopy(*it); ++it;
            Point vertex3position = mesh.getVertexCopy(*it);

            Point edge1 = vertex2position - vertex1position;
            Point edge2 = vertex3position - vertex1position;

            float area = 0.5f * cross(edge1, edge2).length();

            return area;
        }

        float barycentricDualArea(const Geometry::Mesh& mesh, Geometry::VertexHandle vertex) {
            return 0.0f;
        }

    }
}