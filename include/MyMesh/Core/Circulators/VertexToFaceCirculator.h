#pragma once
#include <iterator>
#include <cassert>

#include <MyMesh/Core/Base/Handles.h>

namespace Geometry {

    class Mesh;

    class VertexToFaceCirculator
    {

    private:
        const Mesh* m_mesh;
        HalfEdgeHandle m_start_he;
        HalfEdgeHandle m_curr_he;
        bool m_active; // true until we complete the 360 loop

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = FaceHandle;
        using difference_type = std::ptrdiff_t;
        using pointer = FaceHandle*;
        using reference = FaceHandle&;

        // Constructor
        VertexToFaceCirculator(const Mesh* mesh, HalfEdgeHandle start_he, bool is_end = false);

        // Core Iterator Functions 
        FaceHandle operator*() const;
        VertexToFaceCirculator& operator++();
        VertexToFaceCirculator operator++(int);

        // Equality checks
        bool operator!=(const VertexToFaceCirculator& other) const;
        bool operator==(const VertexToFaceCirculator& other) const;
    };

    class VertexToFaceCirculatorRange {
    private:
        const Mesh* m_mesh;
        HalfEdgeHandle m_start_he;
        friend class Mesh;

        VertexToFaceCirculatorRange(const Mesh* mesh, HalfEdgeHandle start_he) : m_mesh(mesh), m_start_he(start_he) {}

    public:
        VertexToFaceCirculator begin() const { return VertexToFaceCirculator(m_mesh, m_start_he, false); }
        VertexToFaceCirculator end() const { return VertexToFaceCirculator(m_mesh, m_start_he, true); }
    };
}
