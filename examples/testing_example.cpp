#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <Eigen/Sparse>

#include "mesh.h"

// Bring the library into scope
using namespace Geometry;

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

// --- TEST 1: CORE GEOMETRY & TOPOLOGY ---
void testCoreGeometry() {
    std::cout << "\n--- TEST 1: Core Geometry & Topology ---" << std::endl;

    Mesh mesh;

    // 1. Add Vertices
    VertexHandle v0 = mesh.addVertex(0, 0, 0);
    VertexHandle v1 = mesh.addVertex(1, 0, 0);
    VertexHandle v2 = mesh.addVertex(0, 1, 0);

    assertTest(v0.isValid() && v1.isValid() && v2.isValid(), "Vertices Created Validly");
    assertTest(v0.idx() == 0 && v1.idx() == 1, "Vertex Indices are Sequential");

    // 2. Add Face
    FaceHandle f0 = mesh.addTriangle(v0, v1, v2);
    assertTest(f0.isValid(), "Face Created Validly");

    // 3. Check Topology (Half-Edge Connectivity)
    HalfEdgeHandle he = mesh.getHalfEdge(v0);
    assertTest(he.isValid(), "Vertex points to a HalfEdge");

    VertexHandle target = mesh.toVertex(he);
    assertTest(target.isValid(), "HalfEdge has a target");

    // Check loop (Next -> Next -> Next should be back to start)
    HalfEdgeHandle he_next = mesh.next(he);
    HalfEdgeHandle he_next_next = mesh.next(he_next);
    HalfEdgeHandle he_loop = mesh.next(he_next_next);

    assertTest(he == he_loop, "HalfEdge Loop Closed (Triangle is valid)");
}

// --- TEST 2: HANDLE FACTORY & BOUNDS ---
void testHandleFactory() {
    std::cout << "\n--- TEST 2: Handle Factory & Bounds ---" << std::endl;

    Mesh mesh;
    mesh.addVertex(0, 0, 0);
    mesh.addVertex(1, 1, 1);

    // 1. Valid Access
    VertexHandle v1 = mesh.getVertexHandle(1);
    assertTest(v1.isValid() && v1.idx() == 1, "getVertexHandle(1) returns valid handle");

    // 2. Invalid Access (Bounds Check)
    VertexHandle v_bad = mesh.getVertexHandle(999);
    assertTest(!v_bad.isValid(), "getVertexHandle(999) returns INVALID handle");

    VertexHandle v_neg = mesh.getVertexHandle(-5);
    assertTest(!v_neg.isValid(), "getVertexHandle(-5) returns INVALID handle");
}

// --- TEST 3: PROPERTY SYSTEM (TEMPLATES) ---
void testProperties() {
    std::cout << "\n--- TEST 3: Property System ---" << std::endl;

    Mesh mesh;
    VertexHandle v0 = mesh.addVertex(0, 0, 0);
    VertexHandle v1 = mesh.addVertex(1, 0, 0);

    // 1. Create Property (Float) - Scenario: Property added AFTER vertices
    // The property should auto-resize to size 2.
    PropertyCreationState state = mesh.addMeshProperty<float>(MeshComponent::Vertex, "Temperature");
    assertTest(state == PropertyCreationState::Created, "Property 'Temperature' Created");

    // 2. Access and Write Data
    auto propOpt = mesh.getMeshProperty<float>(MeshComponent::Vertex, "Temperature");
    assertTest(propOpt.has_value(), "Retrieved 'Temperature' Property");

    Property<float>& tempProp = propOpt->get();
    assertTest(tempProp.size() == 2, "Property Auto-Resized correctly");

    tempProp[0] = 36.6f;
    tempProp[1] = 100.0f;
    assertTest(std::abs(tempProp[0] - 36.6f) < 0.001f, "Read/Write Float Property");

    // 3. Add Vertex AFTER Property exists
    // The property should auto-grow.
    VertexHandle v2 = mesh.addVertex(0, 1, 0);
    assertTest(tempProp.size() == 3, "Property grew when new vertex added");
    assertTest(tempProp[2] == 0.0f, "New property slot initialized to default (0.0)");

    // 4. Different Type (std::string)
    mesh.addMeshProperty<std::string>(MeshComponent::Vertex, "Label");
    auto labelOpt = mesh.getMeshProperty<std::string>(MeshComponent::Vertex, "Label");
    labelOpt->get()[0] = "Start";
    assertTest(labelOpt->get()[0] == "Start", "Read/Write String Property");

    // 5. Raw Data Access (const vector&)
    const std::vector<float>& rawData = mesh.getPropertyData<float>(MeshComponent::Vertex, "Temperature");
    assertTest(rawData.size() == 3, "Raw Data Access Success");
    assertTest(rawData[1] == 100.0f, "Raw Data Content Correct");
}

// --- TEST 4: DEEP COPY & PROPERTY CLONING ---
void testDeepCopy() {
    std::cout << "\n--- TEST 4: Deep Copy & Cloning ---" << std::endl;

    Mesh m1;
    VertexHandle v0 = m1.addVertex(1, 1, 1);

    // Add property and set value
    m1.addMeshProperty<int>(MeshComponent::Vertex, "ID");
    auto prop1 = m1.getMeshProperty<int>(MeshComponent::Vertex, "ID");
    prop1->get()[0] = 42;

    // COPY CONSTRUCTOR (The big test)
    Mesh m2 = m1;

    // 1. Verify Topology Copy
    VertexHandle v0_2 = m2.getVertexHandle(0);
    assertTest(v0_2.isValid(), "Copied Mesh has vertices");

    // 2. Verify Property Copy
    auto prop2 = m2.getMeshProperty<int>(MeshComponent::Vertex, "ID");
    assertTest(prop2.has_value(), "Copied Mesh has 'ID' property");
    assertTest(prop2->get()[0] == 42, "Property Data Copied Successfully");

    // 3. Verify Independence (Deep Copy check)
    prop2->get()[0] = 99; // Change M2
    assertTest(prop1->get()[0] == 42, "Modifying Copy DOES NOT affect Original (Deep Copy Verified)");
}

// --- TEST 5: ERROR HANDLING ---
void testErrorHandling() {
    std::cout << "\n--- TEST 5: Error Handling ---" << std::endl;

    Mesh mesh;
    mesh.addMeshProperty<float>(MeshComponent::Vertex, "Weight");

    // 1. Type Mismatch
    // Asking for 'Weight' (float) as <int>
    auto badType = mesh.getMeshProperty<int>(MeshComponent::Vertex, "Weight");
    assertTest(!badType.has_value(), "Type Mismatch returns std::nullopt");

    // 2. Non-existent Property
    auto noProp = mesh.getMeshProperty<float>(MeshComponent::Vertex, "Ghost");
    assertTest(!noProp.has_value(), "Non-existent Property returns std::nullopt");

    // 3. Duplicate Creation
    PropertyCreationState state = mesh.addMeshProperty<float>(MeshComponent::Vertex, "Weight");
    assertTest(state == PropertyCreationState::AlreadyExists, "Creating duplicate property returns AlreadyExists");
}

// --- TEST 6: GLOBAL ITERATORS & STL ---
void testGlobalIterators() {
    std::cout << "\n--- TEST 6: Global Iterators & STL ---" << std::endl;

    Mesh mesh;

    // Create a simple geometry: 1 Triangle
    VertexHandle v0 = mesh.addVertex(0, 0, 0);
    VertexHandle v1 = mesh.addVertex(1, 0, 0);
    VertexHandle v2 = mesh.addVertex(0, 1, 0);
    FaceHandle f0 = mesh.addTriangle(v0, v1, v2);

    // 1. Test standard range-based for loop (Vertices)
    int vertex_count = 0;
    for (VertexHandle v : mesh.vertices()) {
        assertTest(v.isValid(), "Vertex Iterator returns valid handle");
        vertex_count++;
    }
    assertTest(vertex_count == 3, "Range-based loop iterated over exactly 3 vertices");

    // 2. Test standard range-based for loop (Faces)
    int face_count = 0;
    for (FaceHandle f : mesh.faces()) {
        assertTest(f.isValid(), "Face Iterator returns valid handle");
        face_count++;
    }
    assertTest(face_count == 1, "Range-based loop iterated over exactly 1 face");

    // 3. Test STL Compatibility (std::count_if)
    // Let's use the STL to count how many vertices are on the Y-axis (x == 0)
    int y_axis_count = std::count_if(
        mesh.vertices().begin(),
        mesh.vertices().end(),
        [&mesh](VertexHandle v) {
            Point p = mesh.getVertexCopy(v);
            return std::abs(p.x - 0.0f) < 0.0001f; // Check if x is roughly 0
        }
    );
    assertTest(y_axis_count == 2, "STL std::count_if works perfectly (found 2 vertices on Y-axis)");

    // 4. Test STL Compatibility (std::for_each)
    // Let's use STL to modify a property across all vertices
    mesh.addMeshProperty<int>(MeshComponent::Vertex, "Visited");
    auto visited_prop = mesh.getMeshProperty<int>(MeshComponent::Vertex, "Visited");

    std::for_each(
        mesh.vertices().begin(),
        mesh.vertices().end(),
        [&visited_prop](VertexHandle v) {
            visited_prop->get()[v.idx()] = 1; // Mark as visited
        }
    );

    assertTest(visited_prop->get()[v0.idx()] == 1 &&
        visited_prop->get()[v1.idx()] == 1 &&
        visited_prop->get()[v2.idx()] == 1,
        "STL std::for_each successfully modified property data");
}


void testVertexCirculator() {
    std::cout << "\n--- TEST 7: Vertex Circulator ---" << std::endl;

    Geometry::Mesh mesh;

    // Create a closed Tetrahedron (4 vertices, 4 triangles)
    Geometry::VertexHandle v0 = mesh.addVertex(0, 0, 0);
    Geometry::VertexHandle v1 = mesh.addVertex(1, 0, 0);
    Geometry::VertexHandle v2 = mesh.addVertex(0, 1, 0);
    Geometry::VertexHandle v3 = mesh.addVertex(0, 0, 1); // The peak

    // Base
    mesh.addTriangle(v0, v1, v2);

    // 3 Sides
    mesh.addTriangle(v0, v3, v1);
    mesh.addTriangle(v1, v3, v2);
    mesh.addTriangle(v2, v3, v0);

    // Circulate around the peak (v3). 
    int neighbor_count = 0;
    bool found_v0 = false, found_v1 = false, found_v2 = false;

    for (Geometry::VertexHandle neighbor : mesh.surroundingVertices(v3)) {
        neighbor_count++;
        if (neighbor == v0) found_v0 = true;
        if (neighbor == v1) found_v1 = true;
        if (neighbor == v2) found_v2 = true;
    }

    assertTest(neighbor_count == 3, "Circulator found exactly 3 neighbors");
    assertTest(found_v0 && found_v1 && found_v2, "Circulator found the correct neighbor vertices");
}

void testVertexCirculator1() {
    std::cout << "\n--- TEST 8: Vertex Circulator (Open Mesh) ---" << std::endl;

    Geometry::Mesh mesh;

    // Create an open plane (2 triangles, has boundaries)
    Geometry::VertexHandle v0 = mesh.addVertex(0, 0, 0);
    Geometry::VertexHandle v1 = mesh.addVertex(1, 0, 0);
    Geometry::VertexHandle v2 = mesh.addVertex(0, 1, 0);
    Geometry::VertexHandle v3 = mesh.addVertex(1, 1, 0);

    mesh.addTriangle(v0, v1, v2);
    mesh.addTriangle(v1, v3, v2);

    // Stitch the boundaries together!
    mesh.finalizeBoundaries();

    // Circulate around v1 (on the boundary edge)
    int neighbor_count = 0;
    bool found_v0 = false, found_v2 = false, found_v3 = false;

    for (Geometry::VertexHandle neighbor : mesh.surroundingVertices(v1)) {
        neighbor_count++;
        if (neighbor == v0) found_v0 = true;
        if (neighbor == v2) found_v2 = true;
        if (neighbor == v3) found_v3 = true;
    }

    assertTest(neighbor_count == 3, "Circulator found exactly 3 neighbors on boundary");
    assertTest(found_v0 && found_v2 && found_v3, "Circulator found the correct neighbor vertices");
}

void testEigen() {
    Eigen::SparseMatrix<float> testMatrix(10, 10);

}

int main() {
    testEigen();
    std::cout << "========================================" << std::endl;
    std::cout << "   TITAN MESH ENGINE - INTEGRATION TEST   " << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        testCoreGeometry();
        testHandleFactory();
        testProperties();
        testDeepCopy();
        testErrorHandling();
        testGlobalIterators();
        testVertexCirculator();
        testVertexCirculator1();
        std::cout << "\n========================================" << std::endl;
        std::cout << "   ALL TESTS PASSED SUCCESSFULLY!     " << std::endl;
        std::cout << "========================================" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "\n[EXCEPTION] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}