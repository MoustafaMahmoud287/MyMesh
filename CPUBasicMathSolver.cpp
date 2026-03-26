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

                triplet_list.push_back(Eigen::Triplet<float>(edge_index, mesh.toVertex(half_edge_1).idx(), 1.0));
                triplet_list.push_back(Eigen::Triplet<float>(edge_index, mesh.toVertex(half_edge_2).idx(), -1.0));
            }

            d0.setFromTriplets(triplet_list.begin(), triplet_list.end());
            return d0;
        }


        SparseMatrix<float> CPUSolver::buildD1(const Geometry::Mesh& mesh) {
            
            auto nEdges = mesh.edgesCount();
            auto nFaces = mesh.facesCount();

            SparseMatrix<float> d1(nFaces, nEdges);
            std::vector<Eigen::Triplet<double>> triplet_list;
            triplet_list.reserve(nFaces * 3);

            for (auto face : mesh.faces()) {
                for(auto half_edge:mesh.)
            }

            return d1;

            /*
                SparseMatrix<double> answer(mesh.nFaces(), mesh.nEdges());

    std::vector<Eigen::Triplet<double>> triplet_list;
    triplet_list.reserve(mesh.nFaces() * 3);

    for(auto f : mesh.faces()){
        for(auto e : f.adjacentEdges()){
            auto he = e.halfedge();
            if(he.face() == f) {
                triplet_list.push_back(Eigen::Triplet<double>(f.getIndex(), e.getIndex(), 1.0));
            }
            else {
                triplet_list.push_back(Eigen::Triplet<double>(f.getIndex(), e.getIndex(), -1.0));
            } 
        }
    }
     
    answer.setFromTriplets(triplet_list.begin(), triplet_list.end());

    return answer;
            */
        }
    }
}