#define CV_CPU_SIMD_FILENAME "/home/smart/Desktop/sasika/hl2ss-forks/viewer/opencv-4.x/modules/core/src/matmul.simd.hpp"
#define CV_CPU_DISPATCH_MODE NEON_DOTPROD
#include "opencv2/core/private/cv_cpu_include_simd_declarations.hpp"

#define CV_CPU_DISPATCH_MODES_ALL NEON_DOTPROD, BASELINE

#undef CV_CPU_SIMD_FILENAME
