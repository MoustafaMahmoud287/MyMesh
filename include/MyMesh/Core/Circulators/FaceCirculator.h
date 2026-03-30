#pragma once
#include <iterator>
#include <cassert>

#include <MyMesh/Core/Base/Handles.h>

namespace Geometry {
    class Mesh;

    class FaceCirculator
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
        FaceCirculator(const Mesh* mesh, HalfEdgeHandle start_he, bool is_end = false);

        // Core Iterator Functions 
        FaceHandle operator*() const;
        FaceCirculator& operator++();
        FaceCirculator operator++(int);

        // Equality checks
        bool operator!=(const FaceCirculator& other) const;
        bool operator==(const FaceCirculator& other) const;
    };

    class FaceCirculatorRange {
    private:
        const Mesh* m_mesh;
        HalfEdgeHandle m_start_he;
        friend class Mesh;

        FaceCirculatorRange(const Mesh* mesh, HalfEdgeHandle start_he) : m_mesh(mesh), m_start_he(start_he) {}

    public:
        FaceCirculator begin() const { return FaceCirculator(m_mesh, m_start_he, false); }
        FaceCirculator end() const { return FaceCirculator(m_mesh, m_start_he, true); }
    };
}
