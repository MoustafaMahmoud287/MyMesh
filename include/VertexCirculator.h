#pragma once
#include <iterator>
#include <cassert>
#include "Handles.h"

namespace Geometry {

    class Mesh;

    class VertexCirculator
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
        VertexCirculator(const Mesh* mesh, HalfEdgeHandle start_he, bool is_end = false);

        // Core Iterator Functions 
        VertexHandle operator*() const;
        VertexCirculator& operator++();
        VertexCirculator operator++(int);

        // Equality checks
        bool operator!=(const VertexCirculator& other) const;
        bool operator==(const VertexCirculator& other) const;
    };

    class VertexCirculatorRange {
    private:
        const Mesh* m_mesh;
        HalfEdgeHandle m_start_he;
        friend class Mesh;

        VertexCirculatorRange(const Mesh* mesh, HalfEdgeHandle start_he) : m_mesh(mesh), m_start_he(start_he) {}

    public:
        VertexCirculator begin() const { return VertexCirculator(m_mesh, m_start_he, false); }
        VertexCirculator end() const { return VertexCirculator(m_mesh, m_start_he, true); }
    };
}
