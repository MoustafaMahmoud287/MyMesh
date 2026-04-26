#pragma once

#include "HardwareProfilerInternals.h"

namespace MyMesh {
    namespace Hardware {

        class HardwareProfiler {
        public:
     
            static FullGPUSpec loadProfile(int device_id = 0);

        private:

            inline static const std::string CONFIG_PATH = "config/hardware_profile";

            static GPUProfile detectAndSaveProfile(int device_id = 0);
            static LimitsProfile estimateLimits(int device_id = 0);
               
        };

    }
}