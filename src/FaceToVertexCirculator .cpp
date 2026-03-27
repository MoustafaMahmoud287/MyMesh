#include "mesh.h"

namespace Geometry {
    // Constructor
    FaceToVertexCirculator::FaceToVertexCirculator(const Mesh* mesh, HalfEdgeHandle start_he, bool is_end)
        : m_mesh(mesh), m_start_he(start_he), m_curr_he(start_he), m_active(!is_end) {
        if (!m_start_he.isValid()) {
            m_active = false;
        }
    }

    // Core Iterator Functions 
    VertexHandle FaceToVertexCirculator::operator*() const {
        return m_mesh->toVertex(m_curr_he);
    }

    FaceToVertexCirculator& FaceToVertexCirculator::operator++() {
        assert(m_active && "Cannot increment an iterator that has reached the end!");
        m_curr_he = m_mesh->next(m_curr_he);
        if (m_curr_he == m_start_he) { m_active = false; }
        return *this;
    }

    FaceToVertexCirculator FaceToVertexCirculator::operator++(int) {
        FaceToVertexCirculator temp = *this;
        ++(*this);
        return temp;
    }

    // Equality checks
    bool FaceToVertexCirculator::operator!=(const FaceToVertexCirculator& other) const {
        assert(m_mesh == other.m_mesh && "Comparing iterators from different meshes!");
        if (m_active != other.m_active) return true;
        return m_active ? (m_curr_he != other.m_curr_he) : false;
    }

    bool FaceToVertexCirculator::operator==(const FaceToVertexCirculator& other) const {
        return !(*this != other);
    }
}