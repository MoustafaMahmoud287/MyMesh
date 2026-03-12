#pragma once
#include <functional> 
#include <utility>    
#include "Handles.h"  

namespace Geometry {

    // 1. hash for std::pair (used in internal EdgeMap)
    struct PairHash {
        inline size_t operator()(const std::pair<HandleIndexType, HandleIndexType>& v) const {
            std::hash<HandleIndexType> hasher;
            size_t seed = 0;

            // standard bit-mixing to combine two integers
            seed ^= hasher(v.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= hasher(v.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

            return seed;
        }
    };

} 


// 2. hash for BaseHandle (VertexHandle, FaceHandle, etc.)
// to use std::unordered_set<VertexHandle> 

namespace std {
    template <typename Tag>
    struct hash<Geometry::BaseHandle<Tag>> {
        size_t operator()(const Geometry::BaseHandle<Tag>& h) const {
            return hash<Geometry::HandleIndexType>()(h.idx());
        }
    };
}