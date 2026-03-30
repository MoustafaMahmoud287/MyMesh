#pragma once
#include "MeshInternals.h" 

namespace Geometry {

    class Mesh
    {
    public:
        // constructors and destructors 
        Mesh();
        Mesh(const Mesh& other);
        Mesh& operator=(const Mesh& other);
        Mesh(Mesh&& other) noexcept;
        Mesh& operator=(Mesh&& other) noexcept;
        ~Mesh();

        // mesh construction
        void reserve(size_t numVertices, size_t numFaces);
        VertexHandle addVertex(float x, float y, float z);
        VertexHandle addVertex(const Point& vertex);
        FaceHandle addTriangle(VertexHandle v1, VertexHandle v2, VertexHandle v3); // sorted counter clockwise
        void finalizeBoundaries();

        // size access
        size_t verticesCount() const;
        size_t halfEdgeCount() const;
        size_t edgesCount() const;
        size_t facesCount() const;

        // topology navigation 
        VertexHandle getVertexHandle(HandleIndexType index) const;
        HalfEdgeHandle getHalfEdgeHandle(HandleIndexType index) const;
        FaceHandle getFaceHandle(HandleIndexType index) const;
        HalfEdgeHandle getHalfEdge(VertexHandle vertex) const;
        HalfEdgeHandle getHalfEdge(FaceHandle face) const;
        HalfEdgeHandle next(HalfEdgeHandle halfedge) const;
        HalfEdgeHandle twin(HalfEdgeHandle halfedge) const;
        VertexHandle   toVertex(HalfEdgeHandle halfedge) const;
        FaceHandle     face(HalfEdgeHandle halfedge) const;
        bool isInteriorHalfEdge(HalfEdgeHandle halfedge) const;
        bool isBoundaryEdge(HandleIndexType edge_index) const;

        // geometry access
        Point getVertexCopy(VertexHandle vertex) const;
        bool setVertex(VertexHandle vertex, const Point& newVertexValue);

        // iterator access
        VertexGlobalRange vertices() const;
        HalfEdgeGlobalRange halfEdges() const;
        FaceGlobalRange faces() const;
        VertexCirculatorRange surroundingVertices(VertexHandle vertex) const;
        VertexToFaceCirculatorRange surroundingFaces(VertexHandle vertex) const;
        HalfEdgeCirculatorRange outgoingHalfEdges(VertexHandle vertex) const;
        FaceCirculatorRange surroundingFaces(FaceHandle face) const;
        FaceToHalfEdgeCirculatorRange surroundingHalfEdges(FaceHandle face) const;
        FaceToVertexCirculatorRange surroundingVertices(FaceHandle face) const;

        // mesh id && versioning access
        uint64_t getID() const;
        uint64_t getTopologyVersion() const;
        uint64_t getGeometryVersion() const;

        // property access 
        template <typename T>
        PropertyCreationState addMeshProperty(MeshComponent property_mesh_component, const std::string& property_name);
        template <typename T>
        std::optional<std::reference_wrapper<Property<T>>> getMeshProperty(MeshComponent property_mesh_component, const std::string& property_name);

        // advanced access for CUDA(GPU) / Raw Data Access
        const std::vector<float>& getVertexData(VertexGeometryAttribute attr) const;
        const std::vector<HandleIndexType>& getVertexTopology(VertexTopologyAttribute attr = VertexTopologyAttribute::FirstHalfEdge) const;
        const std::vector<HandleIndexType>& getHalfEdgeData(HalfEdgeAttribute attr) const;
        const std::vector<HandleIndexType>& getFaceData(FaceAttribute attr = FaceAttribute::StartHalfEdge) const;
        template <typename T>
        const std::vector<T>& getPropertyData(MeshComponent property_mesh_component, const std::string& property_name);

    private:
        // helper functions
        template <typename T>
        PropertyCreationState addMeshProperty(MeshProperty& property_container, const std::string& property_name);

        template <typename T>
        std::optional<std::reference_wrapper<Property<T>>> getMeshProperty(MeshProperty& property_container, const std::string& property_name);


        // for versioning
        void markTopologyDirty();
        void markGeometryDirty();

        //for observer
        friend class ::MyMesh::MathInternal::CPUSolver;
        SolverCallBack onMeshDestroyed = nullptr;
        void rigesterToCPU(SolverCallBack call_back);
        void unRigesterfromCpu();
        bool isCPURigestered() const;

        // internal data 
        inline static std::atomic<uint64_t> s_next_id{ 1 };
        uint64_t m_id;

        uint64_t m_topology_version = 1;
        uint64_t m_geometry_version = 1;

        VertexData m_vertex_data;
        HalfEdgeData m_half_edge_data;
        FaceData m_face_data;
        EdgeMap m_single_half_edge_map;

        MeshProperty m_per_vertex_property;
        MeshProperty m_per_half_edge_property;
        MeshProperty m_per_face_property;
    };

}

// for template implemntation 
#include "mesh.tpp"