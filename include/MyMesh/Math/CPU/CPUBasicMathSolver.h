#pragma once
#include "CPUSolverInternals.h"

namespace MyMesh {
    namespace MathInternal{

        class CPUSolver {
        public:
            CPUSolver() = default;
            ~CPUSolver() = default;

            std::optional<CPUSparseMatrix> multiply(const CPUSparseMatrix& A, const CPUSparseMatrix& B);

        };
    }
}
