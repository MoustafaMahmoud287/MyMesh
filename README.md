# MyMesh

**MyMesh** is a high-performance, template-based C++ library for 3D mesh processing. Developed as a research-oriented project, it implements core geometric data structures and algorithms described in the *Polygon Mesh Processing* curriculum.

## 🛠 Technical Foundation
Unlike simple face-vertex lists, **MyMesh** is built for complex topological queries and research-grade geometric computing.

* **Half-Edge Data Structure:** Optimized for constant-time neighborhood traversals, including one-ring iterations, boundary detection, and vertex-face adjacency.
* **Modern C++ Architecture:** Leverages **C++17** standards, utilizing smart pointers for robust memory management and templates for scalar type flexibility (supporting both `float` and `double` precision).
* **High-Performance Design:** Focused on memory efficiency and cache-friendly mesh traversals to handle complex manifold geometries.

## 🚀 Current Implementation Status
This library is under active development, focusing on the fundamental building blocks of mesh analysis:

- [x] **Core Connectivity:** Manifold mesh representation using a robust Half-Edge structure.
- [x] **Adjacency Queries:** Efficient iterators for vertex-to-edge, edge-to-face, and face-to-face navigation.
- [/] **I/O System (In Progress):** Custom `.obj` file parser and exporter is currently under development to ensure robust handling of large datasets.
- [ ] **Upcoming:** Laplacian Smoothing and Differential Geometry operators (Mean/Gaussian curvature).

## 📂 Project Structure
The repository is organized following professional C++ project standards:

* `include/`: Public API headers and `.tpp` template implementation files.
* `src/`: Core library logic and non-template utilities.
* `examples/`: Demonstration programs showing how to query and manipulate a manifold mesh.

## ⚙️ Building the Project
The library is designed to be build-system agnostic, with primary development performed in **Microsoft Visual Studio 2022**.

### Prerequisites
* C++17 compatible compiler (MSVC, GCC, or Clang).

### Setup in Visual Studio
1. Clone the repository.
2. Add the `include/` directory to your project's **Additional Include Directories**.
3. Set the C++ Language Standard to **ISO C++17**.
4. Include `MyMesh.h` in your source files to begin processing.
