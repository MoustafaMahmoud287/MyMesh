#pragma once
#include <iterator>
#include <cassert>

#include <MyMesh/Core/Base/Handles.h>

namespace Geometry {
    class Mesh;

    class FaceToVertexCirculator
    {
    private:
        const Mesh* m_mesh;
        HalfEdgeHandle m_start_he;
        HalfEdgeHandle m_curr_he;
        bool m_active; // true until we complete the 360 loop

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = VertexHandle;
        using difference_type = std::ptrdiff_t;
        using pointer = VertexHandle*;
        using reference = VertexHandle&;

        // Constructor
        FaceToVertexCirculator(const Mesh* mesh, HalfEdgeHandle start_he, bool is_end = false);

        // Core Iterator Functions 
        VertexHandle operator*() const;
        FaceToVertexCirculator& operator++();
        FaceToVertexCirculator operator++(int);

        // Equality checks
        bool operator!=(const FaceToVertexCirculator& other) const;
        bool operator==(const FaceToVertexCirculator& other) const;
    };

    class FaceToVertexCirculatorRange {
    private:
        const Mesh* m_mesh;
        HalfEdgeHandle m_start_he;
        friend class Mesh;

        FaceToVertexCirculatorRange(const Mesh* mesh, HalfEdgeHandle start_he) : m_mesh(mesh), m_start_he(start_he) {}

    public:
        FaceToVertexCirculator begin() const { return FaceToVertexCirculator(m_mesh, m_start_he, false); }
        FaceToVertexCirculator end() const { return FaceToVertexCirculator(m_mesh, m_start_he, true); }
    };
}
