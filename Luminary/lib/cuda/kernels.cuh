#ifndef CU_KERNELS_H
#define CU_KERNELS_H

#include "scene.h"
#include "primitives.h"
#include "image.h"
#include "raytrace.h"
#include "mesh.h"
#include "SDL/SDL.h"
#include "cuda/utils.cuh"
#include "cuda/math.cuh"
#include "cuda/sky.cuh"
#include "cuda/brdf.cuh"
#include "cuda/bvh.cuh"
#include "cuda/directives.cuh"
#include "cuda/random.cuh"
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <thread>
#include <immintrin.h>

__device__
Sample get_starting_ray(Sample sample) {
  vec3 default_ray;

  default_ray.x = device_scene.camera.focal_length * (-device_scene.camera.fov + device_step * sample.index.x + device_offset_x * sample_blue_noise(sample.index.x,sample.index.y,sample.random_index,8) * 2.0f);
  default_ray.y = device_scene.camera.focal_length * (device_vfov - device_step * sample.index.y - device_offset_y * sample_blue_noise(sample.index.x,sample.index.y,sample.random_index,9) * 2.0f);
  default_ray.z = -device_scene.camera.focal_length;

  const float alpha = sample_blue_noise(sample.index.x,sample.index.y,sample.random_index, 0) * 2.0f * PI;
  const float beta  = sample_blue_noise(sample.index.x,sample.index.y,sample.random_index, 1) * device_scene.camera.aperture_size;

  vec3 point_on_aperture;
  point_on_aperture.x = cosf(alpha) * beta;
  point_on_aperture.y = sinf(alpha) * beta;
  point_on_aperture.z = 0.0f;

  default_ray = vec_diff(default_ray, point_on_aperture);
  point_on_aperture = rotate_vector_by_quaternion(point_on_aperture, device_camera_rotation);

  sample.ray = normalize_vector(rotate_vector_by_quaternion(default_ray, device_camera_rotation));
  sample.origin.x = device_scene.camera.pos.x + point_on_aperture.x;
  sample.origin.y = device_scene.camera.pos.y + point_on_aperture.y;
  sample.origin.z = device_scene.camera.pos.z + point_on_aperture.z;

  return sample;
}

__global__
void generate_samples() {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  int pixel_index = id;
  id *= device_samples_per_sample;
  int sample_offset = 0;
  int iterations_for_pixel = device_diffuse_samples;

  while (id < device_samples_length) {
    Sample sample;

    sample.state.y = (iterations_for_pixel < device_iterations_per_sample) ? iterations_for_pixel : device_iterations_per_sample;
    iterations_for_pixel -= sample.state.y;

    const int x = pixel_index % device_width;
    const int y = pixel_index / device_width;

    sample.index.x = x;
    sample.index.y = y;

    sample.random_index = device_temporal_frames * device_diffuse_samples + iterations_for_pixel * device_reflection_depth;

    sample = get_starting_ray(sample);

    sample.state.x = 0b11 << 8;
    sample.record.r = 1.0f;
    sample.record.g = 1.0f;
    sample.record.b = 1.0f;
    sample.result.r = 0.0f;
    sample.result.g = 0.0f;
    sample.result.b = 0.0f;

    device_samples[id + sample_offset] = sample;

    sample_offset++;

    if (sample_offset == device_samples_per_sample) {
      sample_offset = 0;
      id += device_samples_per_sample * blockDim.x * gridDim.x;
      pixel_index += blockDim.x * gridDim.x;
      iterations_for_pixel = device_diffuse_samples;
    }
  }
}

__device__
int write_albedo_buffer(RGBF value, Sample sample) {
  if (device_denoiser && (sample.state.x & 0x0200)) {
    const int buffer_index = sample.index.x + sample.index.y * device_width;
    atomicAdd((float*)(device_albedo_buffer + buffer_index) + 0, value.r / device_diffuse_samples);
    atomicAdd((float*)(device_albedo_buffer + buffer_index) + 1, value.g / device_diffuse_samples);
    atomicAdd((float*)(device_albedo_buffer + buffer_index) + 2, value.b / device_diffuse_samples);
  }

  return sample.state.x & 0x01ff;
}

__device__
int write_albedo_buffer(RGBAF value, Sample sample) {
  if (device_denoiser && (sample.state.x & 0x0200)) {
    const int buffer_index = sample.index.x + sample.index.y * device_width;
    atomicAdd((float*)(device_albedo_buffer + buffer_index) + 0, value.r / device_diffuse_samples);
    atomicAdd((float*)(device_albedo_buffer + buffer_index) + 1, value.g / device_diffuse_samples);
    atomicAdd((float*)(device_albedo_buffer + buffer_index) + 2, value.b / device_diffuse_samples);
  }

  return sample.state.x & 0x01ff;
}

__device__
Sample write_result(Sample sample) {
  sample.state.y--;

  sample.state.x |= 0x0200;

  sample.state.x &= 0xff00;

  if (sample.state.y <= 0) {
    sample.state.x &= 0x0200;
    atomicSub((int32_t*)&device_sample_offset, 1);
  }

  sample.record.r = 1.0f;
  sample.record.g = 1.0f;
  sample.record.b = 1.0f;

  return get_starting_ray(sample);
}

__device__
float get_light_angle(Light light, vec3 pos) {
    const float d = get_length(vec_diff(pos, light.pos)) + eps;
    return fminf(PI/2.0f,asinf(light.radius / d));
}

__global__
void shade_samples() {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  while (id < device_samples_length) {
    Sample sample;

    while (1) {
      if (id >= device_samples_length) return;
      sample = device_samples[id];
      if (sample.state.x & 0x0100) break;
      id += blockDim.x * gridDim.x;
    }

    if (sample.hit_id == 0xffffffff) {
      RGBF sky = get_sky_color(sample.ray);

      sample.state.x = write_albedo_buffer(sky, sample);

      sample.result.r += sky.r * sample.record.r;
      sample.result.g += sky.g * sample.record.g;
      sample.result.b += sky.b * sample.record.b;

      device_samples[id] = write_result(sample);
      id += blockDim.x * gridDim.x;
      continue;
    }

    sample.origin.x += sample.ray.x * sample.depth;
    sample.origin.y += sample.ray.y * sample.depth;
    sample.origin.z += sample.ray.z * sample.depth;

    const float4* hit_address = (float4*)(device_scene.triangles + sample.hit_id);

    const float4 t1 = __ldg(hit_address);
    const float4 t2 = __ldg(hit_address + 1);
    const float4 t3 = __ldg(hit_address + 2);
    const float4 t4 = __ldg(hit_address + 3);
    const float4 t5 = __ldg(hit_address + 4);
    const float4 t6 = __ldg(hit_address + 5);
    const float4 t7 = __ldg(hit_address + 6);

    vec3 vertex;
    vertex.x = t1.x;
    vertex.y = t1.y;
    vertex.z = t1.z;

    vec3 edge1;
    edge1.x = t1.w;
    edge1.y = t2.x;
    edge1.z = t2.y;

    vec3 edge2;
    edge2.x = t2.z;
    edge2.y = t2.w;
    edge2.z = t3.x;

    vec3 normal = get_coordinates_in_triangle(vertex, edge1, edge2, sample.origin);

    const float lambda = normal.x;
    const float mu = normal.y;

    vec3 vertex_normal;
    vertex_normal.x = t3.y;
    vertex_normal.y = t3.z;
    vertex_normal.z = t3.w;

    vec3 edge1_normal;
    edge1_normal.x = t4.x;
    edge1_normal.y = t4.y;
    edge1_normal.z = t4.z;

    vec3 edge2_normal;
    edge2_normal.x = t4.w;
    edge2_normal.y = t5.x;
    edge2_normal.z = t5.y;

    normal = lerp_normals(vertex_normal, edge1_normal, edge2_normal, lambda, mu);

    UV vertex_texture;
    vertex_texture.u = t5.z;
    vertex_texture.v = t5.w;

    UV edge1_texture;
    edge1_texture.u = t6.x;
    edge1_texture.v = t6.y;

    UV edge2_texture;
    edge2_texture.u = t6.z;
    edge2_texture.v = t6.w;

    const UV tex_coords = lerp_uv(vertex_texture, edge1_texture, edge2_texture, lambda, mu);

    vec3 face_normal;
    face_normal.x = t7.x;
    face_normal.y = t7.y;
    face_normal.z = t7.z;

    const int texture_object = __float_as_int(t7.w);

    const ushort4 maps = __ldg((ushort4*)(device_texture_assignments + texture_object));

    float roughness;
    float metallic;
    float intensity;

    if (maps.z) {
        const float4 material_f = tex2D<float4>(device_material_atlas[maps.z], tex_coords.u, 1.0f - tex_coords.v);

        roughness = (1.0f - material_f.x) * (1.0f - material_f.x);
        metallic = material_f.y;
        intensity = material_f.z * 255.0f;
    } else {
        roughness = 0.81f;
        metallic = 0.0f;
        intensity = 1.0f;
    }

    if (maps.y) {
        #ifdef LIGHTS_AT_NIGHT_ONLY
        if (device_sun.y < NIGHT_THRESHOLD) {
        #endif
        const float4 illuminance_f = tex2D<float4>(device_illuminance_atlas[maps.y], tex_coords.u, 1.0f - tex_coords.v);

        RGBF emission;
        emission.r = illuminance_f.x;
        emission.g = illuminance_f.y;
        emission.b = illuminance_f.z;

        sample.state.x = write_albedo_buffer(emission, sample);

        sample.result.r += emission.r * intensity * sample.record.r;
        sample.result.g += emission.g * intensity * sample.record.g;
        sample.result.b += emission.b * intensity * sample.record.b;

        #ifdef FIRST_LIGHT_ONLY
        const double max_result = fmaxf(sample.result.r, fmaxf(sample.result.g, sample.result.b));
        if (max_result > eps) {
            device_samples[id] = write_result(sample);
            id += blockDim.x * gridDim.x;
            continue;
        }
        #endif

        #ifdef LIGHTS_AT_NIGHT_ONLY
        }
        #endif
    }

    RGBAF albedo;

    if (maps.x) {
        const float4 albedo_f = tex2D<float4>(device_albedo_atlas[maps.x], tex_coords.u, 1.0f - tex_coords.v);
        albedo.r = albedo_f.x;
        albedo.g = albedo_f.y;
        albedo.b = albedo_f.z;
        albedo.a = albedo_f.w;
    } else {
        albedo.r = 0.9f;
        albedo.g = 0.9f;
        albedo.b = 0.9f;
        albedo.a = 1.0f;
    }

    sample.state.x = write_albedo_buffer(albedo, sample);

    if (sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 40) > albedo.a) {
      sample.origin.x += 2.0f * eps * sample.ray.x;
      sample.origin.y += 2.0f * eps * sample.ray.y;
      sample.origin.z += 2.0f * eps * sample.ray.z;

      sample.record.r *= (albedo.r * albedo.a + 1.0f - albedo.a);
      sample.record.g *= (albedo.g * albedo.a + 1.0f - albedo.a);
      sample.record.b *= (albedo.b * albedo.a + 1.0f - albedo.a);
    } else {
        const float specular_probability = lerp(0.5f, 1.0f - eps, metallic);

        if (dot_product(normal, face_normal) < 0.0f) {
            face_normal = scale_vector(face_normal, -1.0f);
        }

        const vec3 V = scale_vector(sample.ray, -1.0f);

        if (dot_product(face_normal, V) < 0.0f) {
            normal = scale_vector(normal, -1.0f);
            face_normal = scale_vector(face_normal, -1.0f);
        }

        sample.origin.x += face_normal.x * (eps * 8.0f);
        sample.origin.y += face_normal.y * (eps * 8.0f);
        sample.origin.z += face_normal.z * (eps * 8.0f);

        const float light_sample = sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 50);

        float light_angle;
        vec3 light_source;
        #ifdef LIGHTS_AT_NIGHT_ONLY
        const int light_count = (device_sun.y < NIGHT_THRESHOLD) ? device_scene.lights_length - 1 : 1;
        #else
        const int light_count = device_scene.lights_length;
        #endif

        const float light_sample_probability = 1.0f - 1.0f/(light_count + 1);

        if (light_sample < light_sample_probability) {
            #ifdef LIGHTS_AT_NIGHT_ONLY
                const uint32_t light = (device_sun.y < NIGHT_THRESHOLD && light_count > 0) ? 1 + (uint32_t)(sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 51) * light_count) : 0;
            #else
                const uint32_t light = (uint32_t)(sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 51) * light_count);
            #endif

            const float4 light_data = __ldg((float4*)(device_scene.lights + light));
            vec3 light_pos;
            light_pos.x = light_data.x;
            light_pos.y = light_data.y;
            light_pos.z = light_data.z;
            light_pos = vec_diff(light_pos, sample.origin);
            light_source = normalize_vector(light_pos);
            const float d = get_length(light_pos) + eps;
            light_angle = fminf(PI/2.0f,asinf(light_data.w / d)) * 2.0f / PI;
        }

        if (sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 10) < specular_probability) {
            const float alpha = roughness * roughness;

            const float beta = sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 2);
            const float gamma = 2.0f * 3.1415926535f * sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 3);

            const Quaternion rotation_to_z = get_rotation_to_z_canonical(normal);

            float weight = 1.0f;

            const vec3 V_local = rotate_vector_by_quaternion(V, rotation_to_z);
            vec3 H_local;

            if (alpha < eps) {
                H_local.x = 0.0f;
                H_local.y = 0.0f;
                H_local.z = 1.0f;
            } else {
                const vec3 S_local = rotate_vector_by_quaternion(
                    normalize_vector(sample_ray_from_angles_and_vector(beta * light_angle, gamma, light_source)),
                    rotation_to_z);

                if (light_sample < light_sample_probability && S_local.z > 0.0f) {
                    H_local.x = S_local.x + V_local.x;
                    H_local.y = S_local.y + V_local.y;
                    H_local.z = S_local.z + V_local.z;

                    H_local = normalize_vector(H_local);

                    weight = (1.0f/light_sample_probability) * light_angle * light_count;
                } else {
                    H_local = sample_GGX_VNDF(V_local, alpha, sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 4), sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 5));

                    if (S_local.z > 0.0f) weight = (1.0f/(1.0f-light_sample_probability));
                }
            }

            const vec3 ray_local = reflect_vector(scale_vector(V_local, -1.0f), H_local);

            const float HdotR = fmaxf(eps, fminf(1.0f, dot_product(H_local, ray_local)));
            const float NdotR = fmaxf(eps, fminf(1.0f, ray_local.z));
            const float NdotV = fmaxf(eps, fminf(1.0f, V_local.z));

            sample.ray = normalize_vector(rotate_vector_by_quaternion(ray_local, inverse_quaternion(rotation_to_z)));

            vec3 specular_f0;
            specular_f0.x = lerp(0.04f, albedo.r, metallic);
            specular_f0.y = lerp(0.04f, albedo.g, metallic);
            specular_f0.z = lerp(0.04f, albedo.b, metallic);

            const vec3 F = Fresnel_Schlick(specular_f0, shadowed_F90(specular_f0), HdotR);

            const float milchs_energy_recovery = lerp(1.0f, 1.51f + 1.51f * NdotV, roughness);

            weight *= milchs_energy_recovery * Smith_G2_over_G1_height_correlated(alpha * alpha, NdotR, NdotV) / specular_probability;

            sample.record.r *= F.x * weight;
            sample.record.g *= F.y * weight;
            sample.record.b *= F.z * weight;
        }
        else
        {
            float weight = 1.0f;

            const float alpha = acosf(sqrtf(sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 2)));
            const float gamma = 2.0f * PI * sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 3);

            sample.ray = normalize_vector(sample_ray_from_angles_and_vector(alpha * light_angle, gamma, light_source));
            const float light_feasible = dot_product(sample.ray, normal);

            if (light_sample < light_sample_probability && light_feasible >= 0.0f) {
                weight = (1.0f/light_sample_probability) * light_angle * light_count;
            } else {
                sample.ray = sample_ray_from_angles_and_vector(alpha, gamma, normal);

                if (light_feasible >=0.0f) weight = (1.0f/(1.0f-light_sample_probability));
            }

            vec3 H;
            H.x = V.x + sample.ray.x;
            H.y = V.y + sample.ray.y;
            H.z = V.z + sample.ray.z;
            H = normalize_vector(H);

            const float half_angle = fmaxf(eps, fminf(dot_product(H, sample.ray),1.0f));
            const float energyFactor = lerp(1.0f, 1.0f/1.51f, roughness);

            const float FD90MinusOne = 0.5f * roughness + 2.0f * half_angle * half_angle * roughness - 1.0f;

            const float angle = fmaxf(eps, fminf(dot_product(normal, sample.ray),1.0f));
            const float previous_angle = fmaxf(eps, fminf(dot_product(V, normal),1.0f));

            const float FDL = 1.0f + (FD90MinusOne * __powf(1.0f - angle, 5.0f));
            const float FDV = 1.0f + (FD90MinusOne * __powf(1.0f - previous_angle, 5.0f));

            weight *= FDL * FDV * energyFactor * (1.0f - metallic) / (1.0f - specular_probability);

            sample.record.r *= albedo.r * weight;
            sample.record.g *= albedo.g * weight;
            sample.record.b *= albedo.b * weight;
        }
    }

    #ifdef WEIGHT_BASED_EXIT
    const double max_record = fmaxf(sample.record.r, fmaxf(sample.record.g, sample.record.b));
    if (max_record < CUTOFF ||
    (max_record < PROBABILISTIC_CUTOFF && sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 20) > (max_record - CUTOFF)/(CUTOFF-PROBABILISTIC_CUTOFF)))
    {
      sample.state.x = (sample.state.x & 0xff00) + device_reflection_depth;
    }
    #endif

    #ifdef LOW_QUALITY_LONG_BOUNCES
    if (sample.state.x & 0x00ff >= MIN_BOUNCES && sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 21) < 1.0f/device_reflection_depth) {
      sample.state.x = (sample.state.x & 0xff00) + device_reflection_depth;
    }
    #endif

    sample.random_index++;
    sample.state.x++;

    if ((sample.state.x & 0x00ff) >= device_reflection_depth) {
      sample = write_result(sample);
    }

    device_samples[id] = sample;

    id += blockDim.x * gridDim.x;
  }
}

__global__
void finalize_samples() {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  int pixel_index = id;
  id *= device_samples_per_sample;
  int sample_offset = 0;

  RGBF pixel;
  pixel.r = 0.0f;
  pixel.g = 0.0f;
  pixel.b = 0.0f;

  while (id < device_samples_length) {
    Sample sample = device_samples[id + sample_offset];

    pixel.r += sample.result.r;
    pixel.g += sample.result.g;
    pixel.b += sample.result.b;

    sample_offset++;

    if (sample_offset == device_samples_per_sample) {
      const float weight = device_scene.camera.exposure/device_diffuse_samples;

      RGBF temporal_pixel = device_frame[pixel_index];
      pixel.r = (pixel.r * weight + temporal_pixel.r * device_temporal_frames) / (device_temporal_frames + 1);
      pixel.g = (pixel.g * weight + temporal_pixel.g * device_temporal_frames) / (device_temporal_frames + 1);
      pixel.b = (pixel.b * weight + temporal_pixel.b * device_temporal_frames) / (device_temporal_frames + 1);

      device_frame[pixel_index] = pixel;

      pixel.r = 0.0f;
      pixel.g = 0.0f;
      pixel.b = 0.0f;

      sample_offset = 0;
      id += device_samples_per_sample * blockDim.x * gridDim.x;
      pixel_index += blockDim.x * gridDim.x;
    }
  }
}


#endif /* CU_KERNELS_H */
