#include <MyMesh/Math/CPU/CPUBasicMathSolver.h>

namespace MyMesh {
    namespace MathInternal {

        CPUSparseMatrix CPUSolver::multiply(const CPUSparseMatrix& A, const CPUSparseMatrix& B) {

            return A * B;

        }

    }
}