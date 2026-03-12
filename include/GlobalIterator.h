#pragma once

#include <iterator>  
#include <cassert>   
#include <cstddef>   
#include "Handles.h"

namespace Geometry {
    class Mesh;

    // 1. THE UNIVERSAL GLOBAL ITERATOR

    template <typename HandleType>
    class GlobalIterator {
    private:
        const Mesh* m_mesh;
        HandleIndexType m_idx;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = HandleType;
        using difference_type = std::ptrdiff_t;
        using pointer = HandleType*;
        using reference = HandleType&;

        GlobalIterator(const Mesh* mesh, HandleIndexType idx)
            : m_mesh(mesh), m_idx(idx) {
        }

        HandleType operator*() const {
            return HandleType(m_idx);
        }

        GlobalIterator& operator++() {
            ++m_idx;
            return *this;
        }

        GlobalIterator operator++(int) {
            GlobalIterator temp = *this; 
            ++(*this);                   
            return temp;                 
        }

        bool operator!=(const GlobalIterator& other) const {
            assert(m_mesh == other.m_mesh && "ERROR: Comparing iterators from different meshes!");
            return m_idx != other.m_idx;
        }

        bool operator==(const GlobalIterator& other) const {
            return !(*this != other);
        }
    };

    // 2. THE UNIVERSAL RANGE WRAPPER
    
    template <typename HandleType>
    class GlobalRange {
    private:
        const Mesh* m_mesh;
        HandleIndexType m_num_elements;

        GlobalRange(const Mesh* mesh, HandleIndexType num_elements)
            : m_mesh(mesh), m_num_elements(num_elements) {
        }

        friend class Mesh;

    public:

        GlobalIterator<HandleType> begin() const {
            return GlobalIterator<HandleType>(m_mesh, 0);
        }

        GlobalIterator<HandleType> end() const {
            return GlobalIterator<HandleType>(m_mesh, m_num_elements);
        }
    };

    // Iterator and ranges tags
    using VertexGlobalIterator = GlobalIterator<VertexHandle>;
    using HalfEdgeGlobalIterator = GlobalIterator<HalfEdgeHandle>;
    using FaceGlobalIterator = GlobalIterator<FaceHandle>;

    using VertexGlobalRange = GlobalRange<VertexHandle>;
    using HalfEdgeGlobalRange = GlobalRange<HalfEdgeHandle>;
    using FaceGlobalRange = GlobalRange<FaceHandle>;
}
