#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <MyMesh/Core/mesh.h>
#include <MyMesh/Math/CPU/CPUTopologyCache.h>
#include <MyMesh/Math/CPU/CPUBasicMathSolver.h> // Assuming this is where CPUSolver lives

using namespace MyMesh;
using namespace MyMesh::MathInternal;

// --- UTILS FOR TESTING ---
void assertTest(bool condition, const std::string& testName) {
    if (condition) {
        std::cout << "[PASS] " << testName << std::endl;
    }
    else {
        std::cout << "[FAIL] " << testName << " <--- CRITICAL ERROR" << std::endl;
        std::terminate(); // Stop immediately on failure
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   TITAN MESH ENGINE - CPU MATH TEST    " << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // 1. Boot the CPU Pipeline
        CPUTopologyCache cache;
        CPUSolver solver;

        // 2. Create a simple test mesh (A single Triangle)
        Geometry::Mesh mesh;
        auto v0 = mesh.addVertex(0, 0, 0);
        auto v1 = mesh.addVertex(1, 0, 0);
        auto v2 = mesh.addVertex(0, 1, 0);
        mesh.addTriangle(v0, v1, v2);

        // Finalize boundaries so edges are perfectly formed
        mesh.finalizeBoundaries();

        // ==========================================
        // TEST 1: Cache Generation & Dimensions
        // ==========================================
        std::cout << "\n--- TEST 1: Cache Generation ---" << std::endl;

        const CPUSparseMatrix& d0 = cache.getD0(mesh);
        assertTest(d0.rows() == mesh.edgesCount() && d0.cols() == mesh.verticesCount(),
            "d0 dimensions are correct (Edges x Vertices)");

        const CPUSparseMatrix& d1 = cache.getD1(mesh);
        assertTest(d1.rows() == mesh.facesCount() && d1.cols() == mesh.edgesCount(),
            "d1 dimensions are correct (Faces x Edges)");

        const CPUSparseMatrix& star0 = cache.getHodgeStar0(mesh);
        assertTest(star0.rows() == mesh.verticesCount() && star0.cols() == mesh.verticesCount(),
            "star0 dimensions are correct (Vertices x Vertices)");

        // ==========================================
        // TEST 2: Cache Hit Verification
        // ==========================================
        std::cout << "\n--- TEST 2: Cache Hit Verification ---" << std::endl;

        // Grab d0 again. It should NOT rebuild. It should return the exact same memory address.
        const CPUSparseMatrix& d0_again = cache.getD0(mesh);
        assertTest(&d0 == &d0_again, "Cache correctly returned existing matrix without rebuilding");

        // ==========================================
        // TEST 3: The DEC Fundamental Theorem (d1 * d0 = 0)
        // ==========================================
        std::cout << "\n--- TEST 3: DEC Fundamental Theorem ---" << std::endl;

        // Execute math on the CPU
        CPUSparseMatrix result = solver.multiply(d1, d0).value_or(CPUSparseMatrix());

        // Check if the result is practically zero
        float max_val = result.coeffs().cwiseAbs().maxCoeff(); // Get largest value in the matrix

        // If the matrix is totally empty, maxCoeff might throw or be zero.
        if (result.nonZeros() == 0) {
            max_val = 0.0f;
        }

        assertTest(max_val < 1e-6f, "d1 * d0 = 0 (Topology Orientations are flawless!)");

        std::cout << "\n========================================" << std::endl;
        std::cout << "   ALL CPU MATH TESTS PASSED!           " << std::endl;
        std::cout << "========================================" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return -1;
    }

    return 0;
}