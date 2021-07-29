#ifndef CU_BRDF_H
#define CU_BRDF_H

#include "utils.cuh"
#include "random.cuh"
#include "math.cuh"
#include <cuda_runtime_api.h>

__device__
float shadowed_F90(const RGBF f0) {
    const float t = 1.0f / 0.04f;
    return fminf(1.0f, t * luminance(f0));
}

__device__
RGBF Fresnel_Schlick(const RGBF f0, const float f90, const float NdotS) {
    RGBF result;

    const float t = powf(1.0f - NdotS, 5.0f);

    result.r = lerp(f0.r, f90, t);
    result.g = lerp(f0.g, f90, t);
    result.b = lerp(f0.b, f90, t);

    return result;
}

__device__
float Smith_G1_GGX(const float alpha2, const float NdotS2) {
    return 2.0f / (sqrtf(((alpha2 * (1.0f - NdotS2)) + NdotS2) / NdotS2) + 1.0f);
}

__device__
float Smith_G2_over_G1_height_correlated(const float alpha2, const float NdotL, const float NdotV) {
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
    const float phi = random2;
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

__device__
vec3 specular_BRDF(RGBF &record, uint32_t &light_sample_id, const vec3 normal, const vec3 V, const Light light, const float light_sample, float light_sample_probability, const int light_count, const RGBAF albedo, const float roughness, const float metallic, const float beta, const float gamma, const float specular_probability) {
    const float alpha = roughness * roughness;

    const Quaternion rotation_to_z = get_rotation_to_z_canonical(normal);

    float weight = 1.0f;

    const vec3 V_local = rotate_vector_by_quaternion(V, rotation_to_z);
    vec3 H_local;

    if (alpha < eps) {
        H_local.x = 0.0f;
        H_local.y = 0.0f;
        H_local.z = 1.0f;

        light_sample_id = ANY_LIGHT;
    } else {
        const vec3 S_local = rotate_vector_by_quaternion(
            normalize_vector(sample_ray_from_angles_and_vector(beta * light.radius, gamma, light.pos)),
            rotation_to_z);

        if (light_sample < light_sample_probability) {
            H_local.x = S_local.x + V_local.x;
            H_local.y = S_local.y + V_local.y;
            H_local.z = S_local.z + V_local.z;

            H_local = normalize_vector(H_local);

            weight = (1.0f/light_sample_probability) * light.radius * light_count;
        } else {
            H_local = sample_GGX_VNDF(V_local, alpha, beta, gamma);

            light_sample_id = ANY_LIGHT;

            weight = (1.0f/(1.0f - light_sample_probability));
        }
    }

    const vec3 ray_local = reflect_vector(scale_vector(V_local, -1.0f), H_local);

    const float HdotR = fmaxf(eps, fminf(1.0f, dot_product(H_local, ray_local)));
    const float NdotR = fmaxf(eps, fminf(1.0f, ray_local.z));
    const float NdotV = fmaxf(eps, fminf(1.0f, V_local.z));

    vec3 ray = normalize_vector(rotate_vector_by_quaternion(ray_local, inverse_quaternion(rotation_to_z)));

    RGBF specular_f0;
    specular_f0.r = lerp(0.04f, albedo.r, metallic);
    specular_f0.g = lerp(0.04f, albedo.g, metallic);
    specular_f0.b = lerp(0.04f, albedo.b, metallic);

    const RGBF F = Fresnel_Schlick(specular_f0, shadowed_F90(specular_f0), HdotR);

    const float milchs_energy_recovery = lerp(1.0f, 1.51f + 1.51f * NdotV, roughness);

    weight *= milchs_energy_recovery * Smith_G2_over_G1_height_correlated(alpha * alpha, NdotR, NdotV) / specular_probability;

    record.r *= F.r * weight;
    record.g *= F.g * weight;
    record.b *= F.b * weight;

    return ray;
}

__device__
vec3 diffuse_BRDF(RGBF &record, uint32_t &light_sample_id, const vec3 normal, const vec3 V, const Light light, const float light_sample, const float light_sample_probability, const int light_count, const RGBAF albedo, const float roughness, const float metallic, const float beta, const float gamma, const float specular_probability) {
    float weight = 1.0f;

    const float alpha = acosf(sqrtf(beta));

    vec3 ray = normalize_vector(sample_ray_from_angles_and_vector(alpha * light.radius, gamma, light.pos));

    if (light_sample < light_sample_probability) {
        weight = (1.0f/light_sample_probability) * light.radius * light_count;
    } else {
        ray = sample_ray_from_angles_and_vector(alpha, gamma, normal);

        light_sample_id = ANY_LIGHT;

        weight = (1.0f/(1.0f - light_sample_probability));
    }

    vec3 H;
    H.x = V.x + ray.x;
    H.y = V.y + ray.y;
    H.z = V.z + ray.z;
    H = normalize_vector(H);

    const float half_angle = fmaxf(eps, fminf(dot_product(H, ray),1.0f));
    const float energyFactor = lerp(1.0f, 1.0f/1.51f, roughness);

    const float FD90MinusOne = 0.5f * roughness + 2.0f * half_angle * half_angle * roughness - 1.0f;

    const float angle = fmaxf(eps, fminf(dot_product(normal, ray),1.0f));
    const float previous_angle = fmaxf(eps, fminf(dot_product(V, normal),1.0f));

    const float FDL = 1.0f + (FD90MinusOne * __powf(1.0f - angle, 5.0f));
    const float FDV = 1.0f + (FD90MinusOne * __powf(1.0f - previous_angle, 5.0f));

    weight *= FDL * FDV * energyFactor * (1.0f - metallic) / (1.0f - specular_probability);

    record.r *= albedo.r * weight;
    record.g *= albedo.g * weight;
    record.b *= albedo.b * weight;

    return ray;
}

__device__
Light sample_light(const vec3 position, const int light_count, uint32_t &light_sample_id, const float r) {
    #ifdef LIGHTS_AT_NIGHT_ONLY
        const uint32_t light_index = (device_sun.y < NIGHT_THRESHOLD && light_count > 0) ? 1 + (uint32_t)(r * light_count) : 0;
    #else
        const uint32_t light_index = (uint32_t)(r * light_count);
    #endif

    light_sample_id = light_index;

    const float4 light_data = __ldg((float4*)(device_scene.lights + light_index));
    vec3 light_pos;
    light_pos.x = light_data.x;
    light_pos.y = light_data.y;
    light_pos.z = light_data.z;
    light_pos = sub_vector(light_pos, position);
    const float d = get_length(light_pos) + eps;

    Light light;
    light.pos = normalize_vector(light_pos);
    light.radius = fminf(1.0f, asinf(light_data.w / d) * 2.0f / PI);

    return light;
}

#endif /* CU_BRDF_H */
