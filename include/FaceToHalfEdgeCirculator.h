#pragma once
#include <iterator>
#include <cassert>
#include "Handles.h"

namespace Geometry {
    class Mesh;

    class FaceToHalfEdgeCirculator
    {
    private:
        const Mesh* m_mesh;
        HalfEdgeHandle m_start_he;
        HalfEdgeHandle m_curr_he;
        bool m_active; // true until we complete the 360 loop

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = HalfEdgeHandle;
        using difference_type = std::ptrdiff_t;
        using pointer = HalfEdgeHandle*;
        using reference = HalfEdgeHandle&;

        // Constructor
        FaceToHalfEdgeCirculator(const Mesh* mesh, HalfEdgeHandle start_he, bool is_end = false);

        // Core Iterator Functions 
        HalfEdgeHandle operator*() const;
        FaceToHalfEdgeCirculator& operator++();
        FaceToHalfEdgeCirculator operator++(int);

        // Equality checks
        bool operator!=(const FaceToHalfEdgeCirculator& other) const;
        bool operator==(const FaceToHalfEdgeCirculator& other) const;
    };

    class FaceToHalfEdgeCirculatorRange {
    private:
        const Mesh* m_mesh;
        HalfEdgeHandle m_start_he;
        friend class Mesh;

        FaceToHalfEdgeCirculatorRange(const Mesh* mesh, HalfEdgeHandle start_he) : m_mesh(mesh), m_start_he(start_he) {}

    public:
        FaceToHalfEdgeCirculator begin() const { return FaceToHalfEdgeCirculator(m_mesh, m_start_he, false); }
        FaceToHalfEdgeCirculator end() const { return FaceToHalfEdgeCirculator(m_mesh, m_start_he, true); }
    };
}
