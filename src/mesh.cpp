#include "Mesh.h"
#include <cassert> 

namespace Geometry {

    Mesh::Mesh(const Mesh& other) {
        // copy geometry & topology
        m_vertex_data = other.m_vertex_data;
        m_half_edge_data = other.m_half_edge_data;
        m_face_data = other.m_face_data;
        m_single_half_edge_map = other.m_single_half_edge_map;

        // deep copy properties
        for (const auto& [name, prop_ptr] : other.m_per_vertex_property) {
            m_per_vertex_property[name] = prop_ptr->clone();
        }
        for (const auto& [name, prop_ptr] : other.m_per_half_edge_property) {
            m_per_half_edge_property[name] = prop_ptr->clone();
        }
        for (const auto& [name, prop_ptr] : other.m_per_face_property) {
            m_per_face_property[name] = prop_ptr->clone();
        }
    }

    Mesh& Mesh::operator=(const Mesh& other) {
        if (this == &other) return *this;

        // copy geometry & topology
        m_vertex_data = other.m_vertex_data;
        m_half_edge_data = other.m_half_edge_data;
        m_face_data = other.m_face_data;
        m_single_half_edge_map = other.m_single_half_edge_map;

        // clear old properties
        m_per_vertex_property.clear();
        m_per_half_edge_property.clear();
        m_per_face_property.clear();

        // deep copy new properties
        for (const auto& [name, prop_ptr] : other.m_per_vertex_property) {
            m_per_vertex_property[name] = prop_ptr->clone();
        }
        for (const auto& [name, prop_ptr] : other.m_per_half_edge_property) {
            m_per_half_edge_property[name] = prop_ptr->clone();
        }
        for (const auto& [name, prop_ptr] : other.m_per_face_property) {
            m_per_face_property[name] = prop_ptr->clone();
        }

        return *this;
    }

    void Mesh::reserve(size_t numVertices, size_t numFaces) {
        size_t estimated_half_edges = 3 * numFaces;
        estimated_half_edges += estimated_half_edges / 10;

        // reserving vertices data
        m_vertex_data.position_x.reserve(numVertices);
        m_vertex_data.position_y.reserve(numVertices);
        m_vertex_data.position_z.reserve(numVertices);
        m_vertex_data.point_to_vertex_half_edge.reserve(numVertices);

        // reserving half edges data
        m_half_edge_data.next.reserve(estimated_half_edges);
        m_half_edge_data.twin.reserve(estimated_half_edges);
        m_half_edge_data.vertex.reserve(estimated_half_edges);
        m_half_edge_data.face.reserve(estimated_half_edges);

        // reserving faces data
        m_face_data.start_half_edge.reserve(numFaces);

        // reserve properties too
        for (auto& [name, prop] : m_per_vertex_property) prop->reserve(numVertices);
        for (auto& [name, prop] : m_per_half_edge_property) prop->reserve(estimated_half_edges);
        for (auto& [name, prop] : m_per_face_property) prop->reserve(numFaces);
    }

    VertexHandle Mesh::addVertex(float x, float y, float z) {
        VertexHandle added_vertex_handle{ static_cast<HandleIndexType>(m_vertex_data.position_x.size()) };

        m_vertex_data.position_x.push_back(x);
        m_vertex_data.position_y.push_back(y);
        m_vertex_data.position_z.push_back(z);
        m_vertex_data.point_to_vertex_half_edge.push_back(-1);

        for (auto& [name, prop_ptr] : m_per_vertex_property) {
            prop_ptr->push_back();
        }

        return added_vertex_handle;
    }

    VertexHandle Mesh::addVertex(const Point& vertex) {
        return addVertex(vertex.x, vertex.y, vertex.z);
    }

    FaceHandle Mesh::addTriangle(VertexHandle v1, VertexHandle v2, VertexHandle v3) {
        if (!v1.isValid() || !v2.isValid() || !v3.isValid()) return FaceHandle();

        HandleIndexType face_index = static_cast<HandleIndexType>(m_face_data.start_half_edge.size());
        m_face_data.start_half_edge.push_back(-1);

        for (auto& [name, prop_ptr] : m_per_face_property) {
            prop_ptr->push_back();
        }

        VertexHandle face_vertices[3] = { v1, v2, v3 };
        HandleIndexType face_half_edge_indices[3];

        for (int i = 0; i < 3; ++i) {
            HandleIndexType from_vertex_index = face_vertices[i].idx();
            HandleIndexType to_vertex_index = face_vertices[(i + 1) % 3].idx();

            std::pair<HandleIndexType, HandleIndexType> twin_key = { to_vertex_index, from_vertex_index };
            auto it = m_single_half_edge_map.find(twin_key);

            if (it != m_single_half_edge_map.end()) {
                face_half_edge_indices[i] = it->second;
                m_half_edge_data.face[face_half_edge_indices[i]] = face_index;
                m_single_half_edge_map.erase(it);
            }
            else {
                HandleIndexType new_he_index = static_cast<HandleIndexType>(m_half_edge_data.face.size());
                face_half_edge_indices[i] = new_he_index; 

                m_half_edge_data.face.push_back(face_index); 
                m_half_edge_data.face.push_back(-1);         

                m_half_edge_data.vertex.push_back(to_vertex_index);   
                m_half_edge_data.vertex.push_back(from_vertex_index); 

                m_half_edge_data.twin.push_back(new_he_index + 1);
                m_half_edge_data.twin.push_back(new_he_index);

                m_half_edge_data.next.push_back(-1); 
                m_half_edge_data.next.push_back(-1); 

                for (auto& [name, prop_ptr] : m_per_half_edge_property) {
                    prop_ptr->push_back(); 
                    prop_ptr->push_back(); 
                }

                std::pair<HandleIndexType, HandleIndexType> new_twin_key = { from_vertex_index, to_vertex_index };
                m_single_half_edge_map[new_twin_key] = new_he_index + 1;
            }
        }

        for (int i = 0; i < 3; ++i) {
            HandleIndexType from_vertex_index = face_vertices[i].idx();
            HandleIndexType current_half_edge = face_half_edge_indices[i];
            HandleIndexType next_half_edge = face_half_edge_indices[(i + 1) % 3];

            m_half_edge_data.next[current_half_edge] = next_half_edge;

            m_vertex_data.point_to_vertex_half_edge[from_vertex_index] = current_half_edge;
        }

        m_face_data.start_half_edge[face_index] = face_half_edge_indices[0];
        return FaceHandle(face_index);
    }

    void Mesh::finalizeBoundaries() {
        std::vector<HandleIndexType> outgoing_boundary(m_vertex_data.position_x.size(), -1);

        for (size_t i = 0; i < m_half_edge_data.face.size(); ++i) {
            if (m_half_edge_data.face[i] == -1) {
                HandleIndexType twin_idx = m_half_edge_data.twin[i];
                HandleIndexType start_vertex = m_half_edge_data.vertex[twin_idx];
                outgoing_boundary[start_vertex] = static_cast<HandleIndexType>(i);
            }
        }

        for (size_t i = 0; i < m_half_edge_data.face.size(); ++i) {
            if (m_half_edge_data.face[i] == -1) {
                HandleIndexType target_vertex = m_half_edge_data.vertex[i];
                m_half_edge_data.next[i] = outgoing_boundary[target_vertex];
            }
        }
    }

    size_t Mesh::verticesCount() const{
        return m_vertex_data.position_x.size();
    }

    size_t Mesh::halfEdgeCount() const{
        return m_half_edge_data.twin.size();
    }

    size_t Mesh::edgesCount() const{
        return halfEdgeCount() / 2;
    }

    size_t Mesh::facesCount() const{
        return m_face_data.start_half_edge.size();
    }

    VertexHandle Mesh::getVertexHandle(HandleIndexType index) const {
        if (index >= 0 && index < m_vertex_data.position_x.size()) {
            return VertexHandle(index);
        }
        return VertexHandle();
    }

    HalfEdgeHandle Mesh::getHalfEdgeHandle(HandleIndexType index) const {
        if (index >= 0 && index < m_half_edge_data.next.size()) {
            return HalfEdgeHandle(index);
        }
        return HalfEdgeHandle();
    }

    FaceHandle Mesh::getFaceHandle(HandleIndexType index) const {
        if (index >= 0 && index < m_face_data.start_half_edge.size()) {
            return FaceHandle(index);
        }
        return FaceHandle();
    }

    HalfEdgeHandle Mesh::getHalfEdge(VertexHandle vertex) const {
        if (!vertex.isValid() || vertex.idx() >= m_vertex_data.point_to_vertex_half_edge.size()) {
            return HalfEdgeHandle(); 
        }
        return HalfEdgeHandle{ m_vertex_data.point_to_vertex_half_edge[vertex.idx()] };
    }

    HalfEdgeHandle Mesh::getHalfEdge(FaceHandle face) const {
        if (!face.isValid() || face.idx() >= m_face_data.start_half_edge.size()) {
            return HalfEdgeHandle(); 
        }
        return HalfEdgeHandle{ m_face_data.start_half_edge[face.idx()] };
    }

    HalfEdgeHandle Mesh::next(HalfEdgeHandle halfedge) const {
        assert(halfedge.isValid() && "Invalid HalfEdge Handle!");
        if (!halfedge.isValid()) return HalfEdgeHandle();
        return HalfEdgeHandle{ m_half_edge_data.next[halfedge.idx()] };
    }

    HalfEdgeHandle Mesh::twin(HalfEdgeHandle halfedge) const {
        assert(halfedge.isValid() && "Invalid HalfEdge Handle!");
        if (!halfedge.isValid()) return HalfEdgeHandle();
        return HalfEdgeHandle{ m_half_edge_data.twin[halfedge.idx()] };
    }

    VertexHandle Mesh::toVertex(HalfEdgeHandle halfedge) const {
        assert(halfedge.isValid() && "Invalid HalfEdge Handle!");
        if (!halfedge.isValid()) return VertexHandle();
        return VertexHandle{ m_half_edge_data.vertex[halfedge.idx()] };
    }

    FaceHandle Mesh::face(HalfEdgeHandle halfedge) const {
        assert(halfedge.isValid() && "Invalid HalfEdge Handle!");
        if (!halfedge.isValid()) return FaceHandle();
        return FaceHandle{ m_half_edge_data.face[halfedge.idx()] };
    }

    bool Mesh::isInteriorHalfEdge(HalfEdgeHandle halfedge) const {
        return (halfedge.isValid()) && (face(halfedge).isValid());
    }

    bool Mesh::isBoundaryEdge(HandleIndexType edge_index) const {
        auto he1 = getHalfEdgeHandle(edge_index * 2);
        auto he2 = twin(he1);
        return (!isInteriorHalfEdge(he1)) || (!isInteriorHalfEdge(he2));
    }

    Point Mesh::getVertexCopy(VertexHandle vertex) const {
        assert(vertex.isValid() && "Invalid Vertex Handle!");
        if (!vertex.isValid()) return Point{ -1,-1,-1 };
        return Point{ m_vertex_data.position_x[vertex.idx()], m_vertex_data.position_y[vertex.idx()], m_vertex_data.position_z[vertex.idx()] };
    }

    bool Mesh::setVertex(VertexHandle vertex, const Point& newVertexValue) {
        assert(vertex.isValid() && "Invalid Vertex Handle!");
        if (!vertex.isValid()) return false;
        m_vertex_data.position_x[vertex.idx()] = newVertexValue.x;
        m_vertex_data.position_y[vertex.idx()] = newVertexValue.y;
        m_vertex_data.position_z[vertex.idx()] = newVertexValue.z;
        return true;
    }

    VertexGlobalRange Mesh::vertices() const {
        return VertexGlobalRange(this, static_cast<HandleIndexType>(m_vertex_data.position_x.size()));
    }

    FaceGlobalRange Mesh::faces() const {
        return FaceGlobalRange(this, static_cast<HandleIndexType>(m_face_data.start_half_edge.size()));
    }

    HalfEdgeGlobalRange Mesh::halfEdges() const {
        return HalfEdgeGlobalRange(this, static_cast<HandleIndexType>(m_half_edge_data.next.size()));
    }

    VertexCirculatorRange Mesh::surroundingVertices(VertexHandle vertex) const {
        assert(vertex.isValid() && "invalid handle");
        return VertexCirculatorRange(this, getHalfEdge(vertex));
    }

    VertexToFaceCirculatorRange Mesh::surroundingFaces(VertexHandle vertex) const {
        assert(vertex.isValid() && "invalid handle");
        return VertexToFaceCirculatorRange(this, getHalfEdge(vertex));
    }

    HalfEdgeCirculatorRange Mesh::outgoingHalfEdges(VertexHandle vertex) const {
        assert(vertex.isValid() && "invalid handle");
        return HalfEdgeCirculatorRange(this, getHalfEdge(vertex));
    }

    FaceCirculatorRange Mesh::surroundingFaces(FaceHandle face) const {
        assert(face.isValid() && "invalid handle");
        return FaceCirculatorRange(this, getHalfEdge(face));
    }

    FaceToHalfEdgeCirculatorRange Mesh::surroundingHalfEdges(FaceHandle face) const {
        assert(face.isValid() && "invalid handle");
        return FaceToHalfEdgeCirculatorRange(this, getHalfEdge(face));
    }

    FaceToVertexCirculatorRange Mesh::surroundingVertices(FaceHandle face) const {
        assert(face.isValid() && "invalid handle");
        return FaceToVertexCirculatorRange(this, getHalfEdge(face));
    }


    const std::vector<float>& Mesh::getVertexData(VertexGeometryAttribute attr) const {
        switch (attr) {
        case VertexGeometryAttribute::PositionX: return m_vertex_data.position_x;
        case VertexGeometryAttribute::PositionY: return m_vertex_data.position_y;
        case VertexGeometryAttribute::PositionZ: return m_vertex_data.position_z;
        }
        static const std::vector<float> empty_float_vec;
        return empty_float_vec;
    }

    const std::vector<HandleIndexType>& Mesh::getVertexTopology(VertexTopologyAttribute attr) const {
        switch (attr) {
        case VertexTopologyAttribute::FirstHalfEdge: return m_vertex_data.point_to_vertex_half_edge;
        }
        static const std::vector<HandleIndexType> empty_int_vec;
        return empty_int_vec;
    }

    const std::vector<HandleIndexType>& Mesh::getHalfEdgeData(HalfEdgeAttribute attr) const {
        switch (attr) {
        case HalfEdgeAttribute::Face: return m_half_edge_data.face;
        case HalfEdgeAttribute::Twin: return m_half_edge_data.twin;
        case HalfEdgeAttribute::Next: return m_half_edge_data.next;
        case HalfEdgeAttribute::TargetVertex: return m_half_edge_data.vertex;
        }
        static const std::vector<HandleIndexType> empty_int_vec;
        return empty_int_vec;
    }

    const std::vector<HandleIndexType>& Mesh::getFaceData(FaceAttribute attr) const {
        switch (attr) {
        case FaceAttribute::StartHalfEdge: return m_face_data.start_half_edge;
        }
        static const std::vector<HandleIndexType> empty_int_vec;
        return empty_int_vec;
    }

} 