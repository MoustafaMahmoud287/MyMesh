#pragma once

#include <MyMesh/Math/CPU/CPUBasicMathSolver.h>
#include <MyMesh/Math/GPU/GPUBasicMathSolver.h>
#include <MyMesh/Hardware/HardwareProfiler.h>
#include <memory>

namespace MyMesh {
    namespace MathInternal {

        enum class Strategy {
            FIXED_BLOCK_SIZE,
            FIXED_BLOCK_COUNT
        };

        const BlockCounterType MIN_BLOCKS_NUM = 3;
        const int DEFAULT_ID = 0;
        const int BLOCKS_DIVIDE_THSHOLD = 10;
        constexpr uint64_t TRANSIENT_ID = std::numeric_limits<uint64_t>::max();
    }
}