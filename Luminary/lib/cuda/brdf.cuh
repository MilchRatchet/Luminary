#ifndef CU_BRDF_H
#define CU_BRDF_H

#include "utils.cuh"
#include "math.cuh"
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

__device__
float Smith_G1_GGX(const float alpha2, const float NdotS2) {
    return 2.0f / (sqrtf(((alpha2 * (1.0f - NdotS2)) + NdotS2) / NdotS2) + 1.0f);
}

__device__
float Smith_G2_over_G1_height_correlated(const float alpha, const float alpha2, const float NdotL, const float NdotV) {
    const float G1V = Smith_G1_GGX(alpha2, NdotV * NdotV);
    const float G1L = Smith_G1_GGX(alpha2, NdotL * NdotL);
    return G1L / (G1V + G1L - G1V * G1L);
}

__device__
vec3 sample_GGX_VNDF(const vec3 v, const float alpha, const float random1, const float random2) {
    vec3 v_hemi;

    v_hemi.x = alpha * v.x;
    v_hemi.y = alpha * v.y;
    v_hemi.z = v.z;

    const float length_squared = v_hemi.x * v_hemi.x + v_hemi.y * v_hemi.y;
    vec3 T1;

    if (length_squared == 0.0f) {
        T1.x = 1.0f;
        T1.y = 0.0f;
        T1.z = 0.0f;
    } else {
        const float length = 1.0f / sqrtf(length_squared);
        T1.x = -v_hemi.y * length;
        T1.y = v_hemi.x * length;
        T1.z = 0.0f;
    }

    const vec3 T2 = cross_product(v_hemi, T1);

    const float r = sqrtf(random1);
    const float phi = 2.0f * PI * random2;
    const float t1 = r * cosf(phi);
    const float s = 0.5f * (1.0f + v_hemi.z);
    const float t2 = lerp(sqrtf(1.0f - t1 * t1), r * sinf(phi), s);

    vec3 normal_hemi;

    const float scalar = sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));
    normal_hemi.x = alpha * (t1 * T1.x + t2 * T2.x + scalar * v_hemi.x);
    normal_hemi.y = alpha * (t1 * T1.y + t2 * T2.y + scalar * v_hemi.y);
    normal_hemi.z = fmaxf(0.0f, t1 * T1.z + t2 * T2.z + scalar * v_hemi.z);

    return normalize_vector(normal_hemi);
}

#endif /* CU_BRDF_H */
