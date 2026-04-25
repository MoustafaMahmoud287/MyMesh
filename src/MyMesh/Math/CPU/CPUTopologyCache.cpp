#include <MyMesh/Math/CPU/CPUTopologyCache.h>
#include <MyMesh/Math/DiscreteGeometryBuilder.h>

namespace MyMesh {
    namespace MathInternal {

        void CPUTopologyCache::registerMesh(const Geometry::Mesh& mesh) {
            if (!mesh.isCPURigestered()) {
                const_cast<Geometry::Mesh&>(mesh).rigesterToCPU([this](uint64_t id) { this->m_cache.erase(id); });
            }
        }

        void CPUTopologyCache::unregisterMesh(const Geometry::Mesh& mesh) {
            const_cast<Geometry::Mesh&>(mesh).unRigesterfromCpu();
        }

        const CPUSparseMatrix& CPUTopologyCache::getD0(const Geometry::Mesh& mesh) {
           
            registerMesh(mesh); 
            auto d0_version = m_cache[mesh.getID()].d0_version;

            if (d0_version != mesh.getTopologyVersion()) {
                setD0(mesh, DiscreteGeometryBuilder::buildD0(mesh));
            }

            return m_cache[mesh.getID()].d0;
   
        }

        const CPUSparseMatrix& CPUTopologyCache::getD1(const Geometry::Mesh& mesh) {

            registerMesh(mesh);
            auto d1_version = m_cache[mesh.getID()].d1_version;

            if (d1_version != mesh.getTopologyVersion()) {
                setD1(mesh, DiscreteGeometryBuilder::buildD1(mesh));
            }

            return m_cache[mesh.getID()].d1;

        }

        const CPUSparseMatrix& CPUTopologyCache::getHodgeStar0(const Geometry::Mesh& mesh) {

            registerMesh(mesh);
            auto star0_version = m_cache[mesh.getID()].star0_version;

            if (star0_version != mesh.getGeometryVersion()) {
                setHodgeStar0(mesh, DiscreteGeometryBuilder::buildHodgeStar0(mesh));
            }

            return m_cache[mesh.getID()].star0;

        }

        const CPUSparseMatrix& CPUTopologyCache::getHodgeStar1(const Geometry::Mesh& mesh) {

            registerMesh(mesh);
            auto star1_version = m_cache[mesh.getID()].star1_version;

            if (star1_version != mesh.getGeometryVersion()) {
                setHodgeStar1(mesh, DiscreteGeometryBuilder::buildHodgeStar1(mesh));
            }

            return m_cache[mesh.getID()].star1;

        }

        const CPUSparseMatrix& CPUTopologyCache::getHodgeStar2(const Geometry::Mesh& mesh) {

            registerMesh(mesh);
            auto star2_version = m_cache[mesh.getID()].star2_version;

            if (star2_version != mesh.getGeometryVersion()) {
                setHodgeStar2(mesh, DiscreteGeometryBuilder::buildHodgeStar2(mesh));
            }

            return m_cache[mesh.getID()].star2;

        }

        void CPUTopologyCache::setD0(const Geometry::Mesh& mesh, CPUSparseMatrix&& mat) {

            auto& entry = m_cache[mesh.getID()];
            entry.d0 = std::move(mat); 
            entry.d0_version = mesh.getTopologyVersion(); 

        }

        void CPUTopologyCache::setD1(const Geometry::Mesh& mesh, CPUSparseMatrix&& mat) {

            auto& entry = m_cache[mesh.getID()];
            entry.d1 = std::move(mat); 
            entry.d1_version = mesh.getTopologyVersion(); 

        }

        void CPUTopologyCache::setHodgeStar0(const Geometry::Mesh& mesh, CPUSparseMatrix&& mat) {

            auto& entry = m_cache[mesh.getID()];
            entry.star0 = std::move(mat);
            entry.star0_version = mesh.getGeometryVersion();

        }

        void CPUTopologyCache::setHodgeStar1(const Geometry::Mesh& mesh, CPUSparseMatrix&& mat) {

            auto& entry = m_cache[mesh.getID()];
            entry.star1 = std::move(mat);
            entry.star1_version = mesh.getGeometryVersion();

        }

        void CPUTopologyCache::setHodgeStar2(const Geometry::Mesh& mesh, CPUSparseMatrix&& mat) {

            auto& entry = m_cache[mesh.getID()];
            entry.star2 = std::move(mat);
            entry.star2_version = mesh.getGeometryVersion();

        }

    }
}