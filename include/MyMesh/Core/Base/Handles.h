#pragma once
#include <iostream>

namespace Geometry { 

    class Mesh; // forward declaration

    template <typename HandleType>
    class GlobalIterator;

    using HandleIndexType = int;

    template <typename Tag>
    class BaseHandle {
        friend class Mesh;
        friend class GlobalIterator<BaseHandle<Tag>>;

    private:
        HandleIndexType m_idx; // index to mesh component on the mesh

        // safety: only Mesh can create
        explicit BaseHandle(HandleIndexType index) : m_idx(index) {}

    public:
        BaseHandle() : m_idx(-1) {}

        HandleIndexType idx() const { return m_idx; }
        bool isValid() const { return m_idx >= 0; }
        void invalidate() { m_idx = -1; }

        bool operator==(const BaseHandle& rhs) const { return m_idx == rhs.m_idx; }
        bool operator!=(const BaseHandle& rhs) const { return m_idx != rhs.m_idx; }
        bool operator<(const BaseHandle& rhs) const { return m_idx < rhs.m_idx; }

        friend std::ostream& operator<<(std::ostream& os, const BaseHandle& h) {
            return os << h.m_idx;
        }
    };

    // Handles tags
    struct VertexTag {};
    struct HalfEdgeTag {};
    struct FaceTag {};

    using VertexHandle = BaseHandle<VertexTag>;
    using HalfEdgeHandle = BaseHandle<HalfEdgeTag>;
    using FaceHandle = BaseHandle<FaceTag>;
} 