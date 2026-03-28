#include "GeometryMath.h"

namespace MyMesh {
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
            
            float dual_area = 0.0f;

            for (auto face : mesh.surroundingFaces(vertex)) {
                dual_area += faceArea(mesh, face);
            }

            dual_area *= (1.0f / 3.0f);

            return dual_area;
        }

        float cotan(const Geometry::Mesh& mesh, Geometry::HalfEdgeHandle halfedge) {

            float sigma = 1e-7;
            float result = 0.0f;

            if (mesh.isInteriorHalfEdge(halfedge)) {
                auto v1 = mesh.getVertexCopy(mesh.toVertex(halfedge)); halfedge = mesh.next(halfedge);
                auto v2 = mesh.getVertexCopy(mesh.toVertex(halfedge)); halfedge = mesh.next(halfedge);
                auto v3 = mesh.getVertexCopy(mesh.toVertex(halfedge));

                auto vector1 = v1 - v2;
                auto vector2 = v3 - v2;

                float cross_result = cross(vector1, vector2).length();
                float dot_result = dot(vector1, vector2);

                if (cross_result >= sigma) result = dot_result / cross_result;
            }

            return result;
        }
    }
}