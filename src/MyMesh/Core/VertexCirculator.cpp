#include <MyMesh/Core/mesh.h>

namespace Geometry {
    // Constructor
    VertexCirculator::VertexCirculator(const Mesh* mesh, HalfEdgeHandle start_he, bool is_end) 
        : m_mesh(mesh), m_start_he(start_he), m_curr_he(start_he), m_active(!is_end) {
        if (!m_start_he.isValid()) {
            m_active = false;
        }
    }

    // Core Iterator Functions 
    VertexHandle VertexCirculator::operator*() const {
        return m_mesh->toVertex(m_curr_he);
    }

    VertexCirculator& VertexCirculator::operator++() {
        assert(m_active && "Cannot increment an iterator that has reached the end!");
        HalfEdgeHandle twin_he = m_mesh->twin(m_curr_he);
        m_curr_he = m_mesh->next(twin_he);
        if (m_curr_he == m_start_he) { m_active = false; }
        return *this;
    }

    VertexCirculator VertexCirculator::operator++(int) {
        VertexCirculator temp = *this; 
        ++(*this);                    
        return temp;                   
    }

    // Equality checks
    bool VertexCirculator::operator!=(const VertexCirculator& other) const {
        assert(m_mesh == other.m_mesh && "Comparing iterators from different meshes!");
        if (m_active != other.m_active) return true;
        return m_active ? (m_curr_he != other.m_curr_he) : false;
    }

    bool VertexCirculator::operator==(const VertexCirculator& other) const {
        return !(*this != other);
    }
}