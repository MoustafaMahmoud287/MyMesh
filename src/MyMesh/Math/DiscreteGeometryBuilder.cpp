#include <MyMesh/Math/DiscreteGeometryBuilder.h>

namespace MyMesh {
    namespace MathInternal {

        CPUSparseMatrix DiscreteGeometryBuilder::buildD0(const Geometry::Mesh& mesh) {

            auto nEdges = mesh.edgesCount();
            auto nVertices = mesh.verticesCount();

            CPUSparseMatrix d0(nEdges, nVertices);
            std::vector<CPUTriplet> triplet_list;
            triplet_list.reserve(nEdges * 2);

            for (Geometry::HandleIndexType edge_index = 0; edge_index < nEdges; edge_index++) {

                auto half_edge_1 = mesh.getHalfEdgeHandle(edge_index * 2);
                auto half_edge_2 = mesh.twin(half_edge_1);

                triplet_list.push_back(CPUTriplet(edge_index, mesh.toVertex(half_edge_1).idx(), 1.0f));
                triplet_list.push_back(CPUTriplet(edge_index, mesh.toVertex(half_edge_2).idx(), -1.0f));
            }

            d0.setFromTriplets(triplet_list.begin(), triplet_list.end());
            return d0;

        }


        CPUSparseMatrix DiscreteGeometryBuilder::buildD1(const Geometry::Mesh& mesh) {
           
            auto nEdges = mesh.edgesCount();
            auto nFaces = mesh.facesCount();

            CPUSparseMatrix d1(nFaces, nEdges);
            std::vector<CPUTriplet> triplet_list;
            triplet_list.reserve(nFaces * 3);

            for (auto face : mesh.faces()) {

                auto face_index = face.idx();

                for (auto half_edge : mesh.surroundingHalfEdges(face)) {

                    auto edge_index = half_edge.idx() / 2;
                    CPUFloat orientation = half_edge.idx() % 2 == 0 ? 1.0f : -1.0f;

                    triplet_list.push_back(CPUTriplet(face_index, edge_index, orientation));
                }
            }

            d1.setFromTriplets(triplet_list.begin(), triplet_list.end());
            return d1;

        }

        CPUSparseMatrix DiscreteGeometryBuilder::buildHodgeStar0(const Geometry::Mesh& mesh) {

            auto nVertices = mesh.verticesCount();

            CPUSparseMatrix star0(nVertices, nVertices);
            std::vector<CPUTriplet> triplet_list;
            triplet_list.reserve(nVertices);

            for (auto vertex : mesh.vertices()) {
                triplet_list.push_back(CPUTriplet(vertex.idx(), vertex.idx(), Math::barycentricDualArea(mesh, vertex)));
            }

            star0.setFromTriplets(triplet_list.begin(), triplet_list.end());
            return star0;
        }

        CPUSparseMatrix DiscreteGeometryBuilder::buildHodgeStar1(const Geometry::Mesh& mesh) {

            auto nEdges = mesh.edgesCount();

            CPUSparseMatrix star1(nEdges, nEdges);
            std::vector<CPUTriplet> triplet_list;
            triplet_list.reserve(nEdges);

            for (auto edge_index = 0; edge_index < nEdges; edge_index++) {

                auto halfedge1 = mesh.getHalfEdgeHandle(edge_index * 2);
                auto halfedge2 = mesh.getHalfEdgeHandle(edge_index * 2 + 1);

                CPUFloat cotan_weight = 0.5f * (Math::cotan(mesh, halfedge1) + Math::cotan(mesh, halfedge2));

                triplet_list.push_back(CPUTriplet(edge_index, edge_index, cotan_weight));
            }

            star1.setFromTriplets(triplet_list.begin(), triplet_list.end());
            return star1;
        }

        CPUSparseMatrix DiscreteGeometryBuilder::buildHodgeStar2(const Geometry::Mesh& mesh) {

            auto nFaces = mesh.facesCount();

            CPUSparseMatrix star2(nFaces, nFaces);

            std::vector<CPUTriplet> triplet_list;
            triplet_list.reserve(nFaces);

            for (auto face : mesh.faces()) {

                CPUFloat area = Math::faceArea(mesh, face);
                CPUFloat safe_area = std::max(area, 1e-8f);

                triplet_list.push_back(CPUTriplet(face.idx(), face.idx(), 1.0f / safe_area));
            }

            star2.setFromTriplets(triplet_list.begin(), triplet_list.end());
            return star2;
        }
    }
}