#include <MyMesh/Core/mesh.h>

namespace Geometry {
    // Constructor
    VertexToFaceCirculator::VertexToFaceCirculator(const Mesh* mesh, HalfEdgeHandle start_he, bool is_end)
        : m_mesh(mesh), m_start_he(start_he), m_curr_he(start_he), m_active(!is_end) {
        if (!m_start_he.isValid()) {
            m_active = false;
        }
    }

    // Core Iterator Functions 
    FaceHandle VertexToFaceCirculator::operator*() const {
        return m_mesh->face(m_curr_he);
    }

    VertexToFaceCirculator& VertexToFaceCirculator::operator++() {
        assert(m_active && "Cannot increment an iterator that has reached the end!");
        m_curr_he = m_mesh->next(m_mesh->twin(m_curr_he));
        if (m_curr_he == m_start_he) { m_active = false; }
        return *this;
    }

    VertexToFaceCirculator VertexToFaceCirculator::operator++(int) {
        VertexToFaceCirculator temp = *this;
        ++(*this);
        return temp;
    }

    // Equality checks
    bool VertexToFaceCirculator::operator!=(const VertexToFaceCirculator& other) const {
        assert(m_mesh == other.m_mesh && "Comparing iterators from different meshes!");
        if (m_active != other.m_active) return true;
        return m_active ? (m_curr_he != other.m_curr_he) : false;
    }

    bool VertexToFaceCirculator::operator==(const VertexToFaceCirculator& other) const {
        return !(*this != other);
    }
}