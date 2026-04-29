#include <iostream>
#include <iomanip>
#include <MyMesh/Hardware/HardwareProfiler.h>

using namespace MyMesh::Hardware;

int main() {
    std::cout << "=================================================\n";
    std::cout << "      MYMESH HARDWARE PROFILER - DIAGNOSTICS     \n";
    std::cout << "=================================================\n\n";

    // 1. Run the profiler (Triggers auto-tuner on first boot, reads .dat on second)
    std::cout << "[Main] Booting Hardware Profiler...\n";
    FullGPUSpec spec = HardwareProfiler::loadProfile(0); // Test device 0

    // 2. Check if the profile successfully locked onto the silicon
    if (!spec.m_gpu_profile.is_valid) {
        std::cerr << "\n[Main] CRITICAL ERROR: GPU Profile is invalid. Check NVIDIA drivers.\n";
        return -1;
    }

    // 3. Print the Raw Physical Specs
    std::cout << "\n--- PHYSICAL HARDWARE ---\n";
    std::cout << "GPU Device ID         : " << spec.m_gpu_profile.id << "\n";
    std::cout << "Compute Capability    : " << spec.m_gpu_profile.compute_capability_major
        << "." << spec.m_gpu_profile.compute_capability_minor << "\n";
    std::cout << "Max Threads/Block     : " << spec.m_gpu_profile.max_threads_per_block << "\n";
    std::cout << "Total Physical VRAM   : " << spec.m_gpu_profile.total_vram_mb << " MB\n";

    // 4. Print the Calculated Engine Routing Limits
    std::cout << "\n--- ENGINE ROUTING LIMITS ---\n";

    // Format the Min Limit (PCIe Tax Threshold)
    size_t min_bytes = spec.m_gpu_limits.MinSizeToRunGPU;
    double min_kb = min_bytes / 1024.0;
    double min_mb = min_bytes / (1024.0 * 1024.0);

    std::cout << "PCIe Tax Threshold (Min) : " << min_bytes << " Bytes ";
    if (min_mb >= 1.0) {
        std::cout << "(" << std::fixed << std::setprecision(2) << min_mb << " MB)\n";
    }
    else {
        std::cout << "(" << std::fixed << std::setprecision(2) << min_kb << " KB)\n";
    }

    // Format the Max Limit (Arena Safety Ceiling)
    std::cout << "Safe Arena Ceiling (Max) : "
        << (spec.m_gpu_limits.MaxAlocatedGPUMemory / (1024 * 1024)) << " MB\n";

    std::cout << "\n=================================================\n";
    std::cout << "              PROFILER TEST COMPLETE             \n";
    std::cout << "=================================================\n";

    return 0;
}