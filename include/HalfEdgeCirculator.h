#pragma once
#include <iterator>
#include <cassert>
#include "Handles.h"

namespace Geometry {

	class HalfEdgeCirculator
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
        using reference = HalfEdgeHandle;

        // Constructor
        HalfEdgeCirculator(const Mesh* mesh, HalfEdgeHandle start_he, bool is_end = false);

        // Core Iterator Functions 
        HalfEdgeHandle operator*() const;
        HalfEdgeCirculator& operator++();
        HalfEdgeCirculator operator++(int);

        // Equality checks
        bool operator!=(const HalfEdgeCirculator& other) const;
        bool operator==(const HalfEdgeCirculator& other) const;
	};

    class HalfEdgeCirculatorRange {
    private:
        const Mesh* m_mesh;
        HalfEdgeHandle m_start_he;
        friend class Mesh;

        HalfEdgeCirculatorRange(const Mesh* mesh, HalfEdgeHandle start_he) : m_mesh(mesh), m_start_he(start_he) {}

    public:
        HalfEdgeCirculator begin() const { return HalfEdgeCirculator(m_mesh, m_start_he, false); }
        HalfEdgeCirculator end() const { return HalfEdgeCirculator(m_mesh, m_start_he, true); }
    };
}

