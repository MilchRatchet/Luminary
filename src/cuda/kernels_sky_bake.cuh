#ifndef CU_KERNELS_SKY_BAKE_H
#define CU_KERNELS_SKY_BAKE_H

#include <cuda_runtime_api.h>

#include "utils.cuh"

__global__ void sky_compute_multiscattering_lut(float4* multiscattering_tex_lower, float4* multiscattering_tex_higher);
__global__ void sky_compute_transmittance_lut(float4* transmittance_tex_lower, float4* transmittance_tex_higher);
__global__ void sky_hdri_compute_hdri_lut(float4* dst);

#endif /* CU_KERNELS_SKY_BAKE_H */
