#pragma once
#include <vector>
#include <unordered_map>
#include <utility>
#include <memory>
#include <functional>
#include <optional>
#include <string>

#include "Point.h"
#include "Handles.h"
#include "Property.h"
#include "Utils.h" 
#include "GlobalIterator.h"
#include "VertexCirculator.h"
#include "FaceCirculator.h"
#include "HalfEdgeCirculator.h"

namespace Geometry {

    // 1. core data structures
    using EdgeMap = std::unordered_map<std::pair<HandleIndexType, HandleIndexType>, HandleIndexType, PairHash>;

    struct VertexData {
        std::vector<float> position_x;
        std::vector<float> position_y;
        std::vector<float> position_z;
        std::vector<HandleIndexType> point_to_vertex_half_edge;
    };

    struct HalfEdgeData {
        std::vector<HandleIndexType> next;
        std::vector<HandleIndexType> twin;
        std::vector<HandleIndexType> vertex; 
        std::vector<HandleIndexType> face;
    };

    struct FaceData {
        std::vector<HandleIndexType> start_half_edge;
    };

    // 2. enums and properties 

    enum class VertexGeometryAttribute {
        PositionX, PositionY, PositionZ
    };

    enum class VertexTopologyAttribute {
        FirstHalfEdge
    };

    enum class HalfEdgeAttribute {
        Next, Twin, TargetVertex, Face
    };

    enum class FaceAttribute {
        StartHalfEdge
    };

    enum class MeshComponent {
        Face, Vertex, HalfEdge
    };

    using MeshProperty = std::unordered_map<std::string, std::unique_ptr<BaseProperty>>;

    enum class PropertyCreationState {
        Created,
        AlreadyExists,
        TypeMismatch
    };

} 