#include <MyMesh/Math/CPU/CPUBasicMathSolver.h>

namespace MyMesh {
    namespace MathInternal {

        std::optional<CPUSparseMatrix> CPUSolver::multiply(const CPUSparseMatrix& A, const CPUSparseMatrix& B) {

            if (A.cols() != B.rows()) {
                std::cerr << "matrix A and matrix B do not have the same dims" << std::endl;
                return std::nullopt;
            }

            return A * B;

        }

    }
}