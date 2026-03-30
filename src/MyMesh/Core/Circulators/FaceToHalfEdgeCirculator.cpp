#include <MyMesh/Core/mesh.h>

namespace Geometry {
    // Constructor
    FaceToHalfEdgeCirculator::FaceToHalfEdgeCirculator(const Mesh* mesh, HalfEdgeHandle start_he, bool is_end)
        : m_mesh(mesh), m_start_he(start_he), m_curr_he(start_he), m_active(!is_end) {
        if (!m_start_he.isValid()) {
            m_active = false;
        }
    }

    // Core Iterator Functions 
    HalfEdgeHandle FaceToHalfEdgeCirculator::operator*() const {
        return m_curr_he;
    }

    FaceToHalfEdgeCirculator& FaceToHalfEdgeCirculator::operator++() {
        assert(m_active && "Cannot increment an iterator that has reached the end!");
        m_curr_he = m_mesh->next(m_curr_he);
        if (m_curr_he == m_start_he) { m_active = false; }
        return *this;
    }

    FaceToHalfEdgeCirculator FaceToHalfEdgeCirculator::operator++(int) {
        FaceToHalfEdgeCirculator temp = *this;
        ++(*this);
        return temp;
    }

    // Equality checks
    bool FaceToHalfEdgeCirculator::operator!=(const FaceToHalfEdgeCirculator& other) const {
        assert(m_mesh == other.m_mesh && "Comparing iterators from different meshes!");
        if (m_active != other.m_active) return true;
        return m_active ? (m_curr_he != other.m_curr_he) : false;
    }

    bool FaceToHalfEdgeCirculator::operator==(const FaceToHalfEdgeCirculator& other) const {
        return !(*this != other);
    }
}