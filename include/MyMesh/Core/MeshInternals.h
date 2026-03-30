#pragma once
#include <vector>
#include <unordered_map>
#include <utility>
#include <memory>
#include <functional>
#include <optional>
#include <string>
#include <atomic>
#include <cstdint>

#include <MyMesh/Core/Base/Point.h>
#include <MyMesh/Core/Base/Handles.h>
#include <MyMesh/Core/Properties/Property.h>
#include <MyMesh/Core/Base/Utils.h> 
#include <MyMesh/Core/Iterators/GlobalIterator.h>
#include <MyMesh/Core/Circulators/VertexCirculator.h>
#include <MyMesh/Core/Circulators/VertexToFaceCirculator.h>
#include <MyMesh/Core/Circulators/FaceCirculator.h>
#include <MyMesh/Core/Circulators/FaceToHalfEdgeCirculator.h>
#include <MyMesh/Core/Circulators/FaceToVertexCirculator.h>
#include <MyMesh/Core/Circulators/HalfEdgeCirculator.h>

namespace MyMesh {
    namespace MathInternal {
        class CPUSolver;
    }
}

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
    
    using SolverCallBack = std::function<void(uint64_t)>;
} 
