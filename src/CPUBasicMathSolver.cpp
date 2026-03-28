#include "CPUBasicMathSolver.h"

namespace MyMesh {
    namespace MathInternal {

        SparseMatrix<float> CPUSolver::buildD0(const Geometry::Mesh& mesh) {
            
            auto nEdges = mesh.edgesCount();
            auto nVertices = mesh.verticesCount();

            SparseMatrix<float> d0(nEdges, nVertices);
            std::vector<Eigen::Triplet<float>> triplet_list;
            triplet_list.reserve(nEdges * 2);
            
            for (Geometry::HandleIndexType edge_index = 0; edge_index < nEdges; edge_index++) {
                
                auto half_edge_1 = mesh.getHalfEdgeHandle(edge_index * 2);
                auto half_edge_2 = mesh.twin(half_edge_1);

                triplet_list.push_back(Eigen::Triplet<float>(edge_index, mesh.toVertex(half_edge_1).idx(), 1.0f));
                triplet_list.push_back(Eigen::Triplet<float>(edge_index, mesh.toVertex(half_edge_2).idx(), -1.0f));
            }

            d0.setFromTriplets(triplet_list.begin(), triplet_list.end());
            return d0;
        }


        SparseMatrix<float> CPUSolver::buildD1(const Geometry::Mesh& mesh) {
            
            auto nEdges = mesh.edgesCount();
            auto nFaces = mesh.facesCount();

            SparseMatrix<float> d1(nFaces, nEdges);
            std::vector<Eigen::Triplet<float>> triplet_list;
            triplet_list.reserve(nFaces * 3);

            for (auto face : mesh.faces()) {

                auto face_index = face.idx();

                for (auto half_edge : mesh.surroundingHalfEdges(face)) {

                    auto edge_index = half_edge.idx() / 2;
                    float orientation = half_edge.idx() % 2 == 0 ? 1.0f : -1.0f;

                    triplet_list.push_back(Eigen::Triplet<float>(face_index, edge_index, orientation));
                }
            }

            d1.setFromTriplets(triplet_list.begin(), triplet_list.end());
            return d1;
        }

        SparseMatrix<float> CPUSolver::buildHodgeStar0(const Geometry::Mesh& mesh) {

            auto nVertices = mesh.verticesCount();

            SparseMatrix<float> hodge0(nVertices, nVertices);

            std::vector<Eigen::Triplet<float>> triplet_list;
            triplet_list.reserve(nVertices);

            for (auto vertex : mesh.vertices()) {
                triplet_list.push_back(Eigen::Triplet<float>(vertex.idx(), vertex.idx(), Math::barycentricDualArea(mesh, vertex)));
            }


            hodge0.setFromTriplets(triplet_list.begin(), triplet_list.end());

            return hodge0;
        }

        SparseMatrix<float> CPUSolver::buildHodgeStar1(const Geometry::Mesh& mesh) {

            auto nEdges = mesh.edgesCount();

            SparseMatrix<float> hodge1(nEdges, nEdges);

            std::vector<Eigen::Triplet<float>> triplet_list;
            triplet_list.reserve(nEdges);

            for (auto edge_index = 0; edge_index < nEdges; edge_index++) {
                
                auto halfedge1 = mesh.getHalfEdgeHandle(edge_index * 2);
                auto halfedge2 = mesh.getHalfEdgeHandle(edge_index * 2 + 1);

                float cotan_weight = 0.5f * (Math::cotan(mesh, halfedge1) + Math::cotan(mesh, halfedge2));
                
                triplet_list.push_back(Eigen::Triplet<float>(edge_index, edge_index, cotan_weight));
            }

            hodge1.setFromTriplets(triplet_list.begin(), triplet_list.end());

            return hodge1;
        }
    }
}