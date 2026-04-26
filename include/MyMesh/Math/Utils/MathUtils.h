#include <MyMesh/Math/CPU/CPUSolverInternals.h>
#include <vector>
#include <random>

namespace MyMesh {
    namespace MathInternal {

        inline CPUSparseMatrix generateRandomSparseMatrix(int rows, int cols, float sparsity, int seed) {
            CPUSparseMatrix mat(rows, cols);

            size_t target_nnz = static_cast<size_t>(rows * cols * sparsity);

            std::vector<Eigen::Triplet<float>> tripletList;
            tripletList.reserve(target_nnz);

            std::mt19937 gen(seed);
            std::uniform_int_distribution<int> row_dist(0, rows - 1);
            std::uniform_int_distribution<int> col_dist(0, cols - 1);

            std::uniform_real_distribution<float> val_dist(0.1f, 1.0f);

            for (size_t i = 0; i < target_nnz; ++i) {
                int r = row_dist(gen);
                int c = col_dist(gen);
                float v = val_dist(gen);

                tripletList.emplace_back(r, c, v);
            }

            mat.setFromTriplets(tripletList.begin(), tripletList.end());
            mat.makeCompressed();

            return mat;
        }
    }
}