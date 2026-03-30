#include "CPUBasicMathSolver.h"

namespace MyMesh {
    namespace MathInternal {

        const CPUSparseMatrix& CPUSolver::buildD0(const Geometry::Mesh& mesh) {
            
            if (!mesh.isCPURigestered()) registerMesh(mesh);

            auto& cache_entry = m_cache[mesh.getID()];

            if (cache_entry.d0_version != mesh.getTopologyVersion()) {

                auto nEdges = mesh.edgesCount();
                auto nVertices = mesh.verticesCount();

                cache_entry.d0.resize(nEdges, nVertices);

                std::vector<CPUTriplet> triplet_list;
                triplet_list.reserve(nEdges * 2);

                for (Geometry::HandleIndexType edge_index = 0; edge_index < nEdges; edge_index++) {

                    auto half_edge_1 = mesh.getHalfEdgeHandle(edge_index * 2);
                    auto half_edge_2 = mesh.twin(half_edge_1);

                    triplet_list.push_back(CPUTriplet(edge_index, mesh.toVertex(half_edge_1).idx(), 1.0f));
                    triplet_list.push_back(CPUTriplet(edge_index, mesh.toVertex(half_edge_2).idx(), -1.0f));
                }

                cache_entry.d0.setFromTriplets(triplet_list.begin(), triplet_list.end());
                cache_entry.d0_version = mesh.getTopologyVersion();
            }

            return cache_entry.d0;
        }


        const CPUSparseMatrix& CPUSolver::buildD1(const Geometry::Mesh& mesh) {

            if (!mesh.isCPURigestered()) registerMesh(mesh);
            
            auto& cache_entry = m_cache[mesh.getID()];

            if (cache_entry.d1_version != mesh.getTopologyVersion()) {

                auto nEdges = mesh.edgesCount();
                auto nFaces = mesh.facesCount();

                cache_entry.d1.resize(nFaces, nEdges);

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

                cache_entry.d1.setFromTriplets(triplet_list.begin(), triplet_list.end());
                cache_entry.d1_version = mesh.getTopologyVersion();
            }

            return cache_entry.d1;
        }

        const CPUSparseMatrix& CPUSolver::buildHodgeStar0(const Geometry::Mesh& mesh) {

            if (!mesh.isCPURigestered()) registerMesh(mesh);

            auto& cache_entry = m_cache[mesh.getID()];

            if (cache_entry.star0_version != mesh.getGeometryVersion()) {

                auto nVertices = mesh.verticesCount();

                cache_entry.star0.resize(nVertices, nVertices);

                std::vector<CPUTriplet> triplet_list;
                triplet_list.reserve(nVertices);

                for (auto vertex : mesh.vertices()) {
                    triplet_list.push_back(CPUTriplet(vertex.idx(), vertex.idx(), Math::barycentricDualArea(mesh, vertex)));
                }

                cache_entry.star0.setFromTriplets(triplet_list.begin(), triplet_list.end());
                cache_entry.star0_version = mesh.getGeometryVersion();
            }

            return cache_entry.star0;
        }

        const CPUSparseMatrix& CPUSolver::buildHodgeStar1(const Geometry::Mesh& mesh) {

            if (!mesh.isCPURigestered()) registerMesh(mesh);

            auto& cache_entry = m_cache[mesh.getID()];

            if (cache_entry.star1_version != mesh.getGeometryVersion()) {

                auto nEdges = mesh.edgesCount();

                cache_entry.star1.resize(nEdges, nEdges);

                std::vector<CPUTriplet> triplet_list;
                triplet_list.reserve(nEdges);

                for (auto edge_index = 0; edge_index < nEdges; edge_index++) {

                    auto halfedge1 = mesh.getHalfEdgeHandle(edge_index * 2);
                    auto halfedge2 = mesh.getHalfEdgeHandle(edge_index * 2 + 1);

                    CPUFloat cotan_weight = 0.5f * (Math::cotan(mesh, halfedge1) + Math::cotan(mesh, halfedge2));

                    triplet_list.push_back(CPUTriplet(edge_index, edge_index, cotan_weight));
                }

                cache_entry.star1.setFromTriplets(triplet_list.begin(), triplet_list.end());
                cache_entry.star1_version = mesh.getGeometryVersion();
            }

            return cache_entry.star1;
        }

        const CPUSparseMatrix& CPUSolver::buildHodgeStar2(const Geometry::Mesh& mesh) {

            if (!mesh.isCPURigestered()) registerMesh(mesh);

            auto& cache_entry = m_cache[mesh.getID()];

            if (cache_entry.star2_version != mesh.getGeometryVersion()) {

                auto nFaces = mesh.facesCount();

                cache_entry.star2.resize(nFaces, nFaces);

                std::vector<CPUTriplet> triplet_list;
                triplet_list.reserve(nFaces);

                for (auto face : mesh.faces()) {

                    CPUFloat area = Math::faceArea(mesh, face);
                    CPUFloat safe_area = std::max(area, 1e-8f);

                    triplet_list.push_back(CPUTriplet(face.idx(), face.idx(), 1.0f / safe_area));
                }

                cache_entry.star2.setFromTriplets(triplet_list.begin(), triplet_list.end());
                cache_entry.star2_version = mesh.getGeometryVersion();
            }

            return cache_entry.star2;
        }

        void CPUSolver::registerMesh(const Geometry::Mesh& mesh) {
            const_cast<Geometry::Mesh&>(mesh).rigesterToCPU([this](uint64_t id) { this->m_cache.erase(id); });
        }

        void CPUSolver::unregisterMesh(const Geometry::Mesh& mesh) {
            const_cast<Geometry::Mesh&>(mesh).unRigesterfromCpu();
        }
    }
}