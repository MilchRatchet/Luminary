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
#include "cuda/memory.cuh"
#include "cuda/ocean.cuh"
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
Sample get_starting_ray(Sample sample, const int id) {
  vec3 default_ray;

  default_ray.x = device_scene.camera.focal_length * (-device_scene.camera.fov + device_step * sample.index.x + device_offset_x * sample_blue_noise(sample.index.x,sample.index.y,sample.random_index,8, id) * 2.0f);
  default_ray.y = device_scene.camera.focal_length * (device_vfov - device_step * sample.index.y - device_offset_y * sample_blue_noise(sample.index.x,sample.index.y,sample.random_index,9, id) * 2.0f);
  default_ray.z = -device_scene.camera.focal_length;

  const float alpha = sample_blue_noise(sample.index.x,sample.index.y,sample.random_index, 0, id) * 2.0f * PI;
  const float beta  = sample_blue_noise(sample.index.x,sample.index.y,sample.random_index, 1, id) * device_scene.camera.aperture_size;

  vec3 point_on_aperture = get_vector(cosf(alpha) * beta, sinf(alpha) * beta, 0.0f);

  default_ray = sub_vector(default_ray, point_on_aperture);
  point_on_aperture = rotate_vector_by_quaternion(point_on_aperture, device_camera_rotation);

  sample.ray = normalize_vector(rotate_vector_by_quaternion(default_ray, device_camera_rotation));
  sample.origin = add_vector(device_scene.camera.pos, point_on_aperture);

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

    const unsigned short x = (unsigned short)(pixel_index % device_width);
    const unsigned short y = (unsigned short)(pixel_index / device_width);

    sample.index.x = x;
    sample.index.y = y;

    sample.random_index = device_temporal_frames * device_diffuse_samples + iterations_for_pixel;

    sample = get_starting_ray(sample, id + sample_offset);

    sample.state.x = (0b111 << 8) | (sample_offset << 11);
    sample.record = get_color(1.0f, 1.0f, 1.0f);
    sample.result = get_color(0.0f, 0.0f, 0.0f);
    sample.albedo_buffer = get_color(0.0f, 0.0f, 0.0f);

    device_active_samples[id + sample_offset] = sample;

    curandStateXORWOW_t state;
    curand_init(sample.random_index + x + y * device_width + clock(), 0, 0, &state);
    device_sample_randoms[id + sample_offset] = state;

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
Sample write_albedo_buffer(RGBF value, Sample sample) {
  if (device_denoiser && (sample.state.x & 0x0200)) {
    sample.albedo_buffer = add_color(sample.albedo_buffer, value);
  }

  sample.state.x &= 0xfdff;

  return sample;
}

__device__
Sample write_albedo_buffer(RGBAF value, Sample sample) {
  if (device_denoiser && (sample.state.x & 0x0200)) {
    sample.albedo_buffer.r += value.r;
    sample.albedo_buffer.g += value.g;
    sample.albedo_buffer.b += value.b;
  }

  sample.state.x &= 0xfdff;

  return sample;
}

__device__
Sample write_result(Sample sample, int id) {
  sample.state.y--;

  sample.random_index -= (sample.state.x & 0x00ff) + 1;

  sample.state.x |= 0x0600;

  sample.state.x &= 0xff00;

  if (sample.state.y <= 0) {
    sample.state.x &= 0xfe00;
  }

  sample.record = get_color(1.0f, 1.0f, 1.0f);

  return get_starting_ray(sample, id);
}

__device__
float get_light_angle(Light light, vec3 pos) {
    const float d = get_length(sub_vector(pos, light.pos)) + eps;
    return fminf(PI/2.0f,asinf(light.radius / d));
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 8)
void shade_samples() {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int store_id = id;

  while (id < device_samples_length) {
    Sample sample = load_active_sample_no_temporal_hint(device_active_samples + id);

    if (!(sample.state.x & 0x0100)) return;

    __prefetch_global_l1(device_scene.triangles + sample.hit_id);

    float ocean_intersection;
    if (device_scene.ocean.active) {
      ocean_intersection = get_intersection_ocean(sample.origin, sample.ray, sample.depth);
    } else {
      ocean_intersection = FLT_MAX;
    }

    vec3 normal;
    RGBAF albedo;
    float roughness;
    float metallic;
    float intensity;
    vec3 face_normal;

    if (ocean_intersection < sample.depth) {
      sample.origin = add_vector(sample.origin, scale_vector(sample.ray, ocean_intersection));

      normal = get_ocean_normal(sample.origin, fmaxf(0.1f * eps, ocean_intersection * 0.1f / device_width));

      albedo = device_scene.ocean.albedo;

      face_normal = normal;

      roughness = 0.01f;
      metallic = 0.0f;

      if (device_scene.ocean.emissive) {
        RGBF emission = get_color(albedo.r, albedo.g, albedo.b);
        intensity = 2.0f;

        sample = write_albedo_buffer(emission, sample);

        if (!isnan(sample.record.r) && !isinf(sample.record.r) && !isnan(sample.record.g) && !isinf(sample.record.g) && !isnan(sample.record.b) && !isinf(sample.record.b)) {
          sample.result.r += emission.r * intensity * sample.record.r;
          sample.result.g += emission.g * intensity * sample.record.g;
          sample.result.b += emission.b * intensity * sample.record.b;
        }
      }

    } else if (sample.hit_id == 0xffffffff) {
      RGBF sky = get_sky_color(sample.ray);

      sample = write_albedo_buffer(sky, sample);

      if (!(sample.state.x & 0x0400)) {
        sky.r = (sky.r > 1.0f) ? 0.0f : sky.r;
        sky.g = (sky.g > 1.0f) ? 0.0f : sky.g;
        sky.b = (sky.b > 1.0f) ? 0.0f : sky.b;
      }

      if (!isnan(sample.record.r) && !isinf(sample.record.r) && !isnan(sample.record.g) && !isinf(sample.record.g) && !isnan(sample.record.b) && !isinf(sample.record.b)) {
        sample.result.r += sky.r * sample.record.r;
        sample.result.g += sky.g * sample.record.g;
        sample.result.b += sky.b * sample.record.b;
      }

      sample = write_result(sample, id);
      goto store_back_sample;
    } else {
      sample.origin = add_vector(sample.origin, scale_vector(sample.ray, sample.depth));

      const float4* hit_address = (float4*)(device_scene.triangles + sample.hit_id);

      const float4 t1 = __ldg(hit_address);
      const float4 t2 = __ldg(hit_address + 1);
      const float4 t3 = __ldg(hit_address + 2);
      const float4 t4 = __ldg(hit_address + 3);
      const float4 t5 = __ldg(hit_address + 4);
      const float4 t6 = __ldg(hit_address + 5);
      const float4 t7 = __ldg(hit_address + 6);

      vec3 vertex = get_vector(t1.x, t1.y, t1.z);
      vec3 edge1 = get_vector(t1.w, t2.x, t2.y);
      vec3 edge2 = get_vector(t2.z, t2.w, t3.x);

      normal = get_coordinates_in_triangle(vertex, edge1, edge2, sample.origin);

      const float lambda = normal.x;
      const float mu = normal.y;

      vec3 vertex_normal = get_vector(t3.y, t3.z, t3.w);
      vec3 edge1_normal = get_vector(t4.x, t4.y, t4.z);
      vec3 edge2_normal = get_vector(t4.w, t5.x, t5.y);

      normal = lerp_normals(vertex_normal, edge1_normal, edge2_normal, lambda, mu);

      UV vertex_texture = get_UV(t5.z, t5.w);
      UV edge1_texture = get_UV(t6.x, t6.y);
      UV edge2_texture = get_UV(t6.z, t6.w);

      const UV tex_coords = lerp_uv(vertex_texture, edge1_texture, edge2_texture, lambda, mu);

      face_normal = get_vector(t7.x, t7.y, t7.z);

      const int texture_object = __float_as_int(t7.w);

      const ushort4 maps = __ldg((ushort4*)(device_texture_assignments + texture_object));

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

          RGBF emission = get_color(illuminance_f.x, illuminance_f.y, illuminance_f.z);

          sample = write_albedo_buffer(emission, sample);

          if (sample.state.x & 0x0400) {
            if (!isnan(sample.record.r) && !isinf(sample.record.r) && !isnan(sample.record.g) && !isinf(sample.record.g) && !isnan(sample.record.b) && !isinf(sample.record.b)) {
              sample.result.r += emission.r * intensity * sample.record.r;
              sample.result.g += emission.g * intensity * sample.record.g;
              sample.result.b += emission.b * intensity * sample.record.b;
            }
          }

          #ifdef FIRST_LIGHT_ONLY
          const double max_emission = intensity * fmaxf(emission.r, fmaxf(emission.g, emission.b));
          if (max_emission > eps) {
              sample = write_result(sample, id);
              goto store_back_sample;
          }
          #endif

          #ifdef LIGHTS_AT_NIGHT_ONLY
          }
          #endif
      }

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
    }

    sample = write_albedo_buffer(albedo, sample);

    if (sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 40, id) > albedo.a) {
      sample.origin = add_vector(sample.origin, scale_vector(sample.ray, 2.0f * eps));

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

        sample.origin = add_vector(sample.origin, scale_vector(face_normal, 8.0f * eps));

        const float light_sample = sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 50, id);

        Light light;
        #ifdef LIGHTS_AT_NIGHT_ONLY
          const int light_count = (device_sun.y < NIGHT_THRESHOLD) ? device_scene.lights_length - 1 : 1;
        #else
          const int light_count = device_scene.lights_length;
        #endif

        const float light_sample_probability = 1.0f - 1.0f/(light_count + 1);

        if (light_sample < light_sample_probability) {
          light = sample_light(sample, light_count, sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 51, id));
        }

        __prefetch_global_l1(device_active_samples + id + blockDim.x * gridDim.x);

        const float gamma = 2.0f * PI * sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 3, id);
        const float beta = sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 2, id);

        if (sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 10, id) < specular_probability) {
          sample = specular_BRDF(sample, normal, V, light, light_sample, light_sample_probability, light_count, albedo, roughness, metallic, beta, gamma, specular_probability);
        }
        else
        {
          sample = diffuse_BRDF(sample, normal, V, light, light_sample, light_sample_probability, light_count, albedo, roughness, metallic, beta, gamma, specular_probability);
        }
    }

    #ifdef WEIGHT_BASED_EXIT
    const double max_record = fmaxf(sample.record.r, fmaxf(sample.record.g, sample.record.b));
    if (max_record < CUTOFF ||
    (max_record < PROBABILISTIC_CUTOFF && sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 20, id) > (max_record - CUTOFF)/(CUTOFF-PROBABILISTIC_CUTOFF)))
    {
      sample.state.x = (sample.state.x & 0xff00) + device_reflection_depth;
    }
    #endif

    #ifdef LOW_QUALITY_LONG_BOUNCES
    if (sample.state.x & 0x00ff >= MIN_BOUNCES && sample_blue_noise(sample.index.x, sample.index.y, sample.random_index, 21, id) < 1.0f/device_reflection_depth) {
      sample.state.x = (sample.state.x & 0xff00) + device_reflection_depth;
    }
    #endif

    sample.random_index++;
    sample.state.x++;

    if ((sample.state.x & 0x00ff) >= device_reflection_depth) {
      sample = write_result(sample, id);
    }

    store_back_sample:;

    if (sample.state.x & 0x0100) {
      if (store_id < id) __stcs((unsigned short*)(device_active_samples + id) + 12, 0);
      store_active_sample_no_temporal_hint(sample, device_active_samples + store_id);
      store_id += blockDim.x * gridDim.x;
    } else {
      const unsigned int address = (sample.index.x + sample.index.y * device_width) * device_samples_per_sample + (sample.state.x >> 11);
      store_finished_sample_no_temporal_hint(sample, device_finished_samples + address);
      atomicSub((int32_t*)&device_sample_offset, 1);
      __stcs((unsigned short*)(device_active_samples + id) + 12, 0);
    }

    id += blockDim.x * gridDim.x;
  }
}

__global__
void finalize_samples() {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

  int pixel_index = id;
  id *= device_samples_per_sample;
  int sample_offset = 0;

  RGBF pixel = get_color(0.0f, 0.0f, 0.0f);
  RGBF albedo = get_color(0.0f, 0.0f, 0.0f);

  while (id < device_samples_length) {
    Sample_Result sample = load_finished_sample_no_temporal_hint(device_finished_samples + id + sample_offset);

    pixel = add_color(pixel, sample.result);
    albedo = add_color(albedo, sample.albedo_buffer);

    sample_offset++;

    if (sample_offset == device_samples_per_sample) {
      const float weight = device_scene.camera.exposure/device_diffuse_samples;

      if (isnan(pixel.r) || isinf(pixel.r)) pixel.r = 0.0f;
      if (isnan(pixel.g) || isinf(pixel.g)) pixel.g = 0.0f;
      if (isnan(pixel.b) || isinf(pixel.b)) pixel.b = 0.0f;

      if (device_temporal_frames) {
        RGBF temporal_pixel = device_frame[pixel_index];
        pixel.r = (pixel.r * weight + temporal_pixel.r * device_temporal_frames) / (device_temporal_frames + 1);
        pixel.g = (pixel.g * weight + temporal_pixel.g * device_temporal_frames) / (device_temporal_frames + 1);
        pixel.b = (pixel.b * weight + temporal_pixel.b * device_temporal_frames) / (device_temporal_frames + 1);
      } else {
        pixel.r *= weight;
        pixel.g *= weight;
        pixel.b *= weight;
      }

      device_frame[pixel_index] = pixel;

      if (device_denoiser) {
        device_albedo_buffer[pixel_index] = albedo;
      }

      pixel = get_color(0.0f, 0.0f, 0.0f);
      albedo = get_color(0.0f, 0.0f, 0.0f);

      sample_offset = 0;
      id += device_samples_per_sample * blockDim.x * gridDim.x;
      pixel_index += blockDim.x * gridDim.x;
    }
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10)
void special_shade_samples() {
  unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int store_id = id;

  while (id < device_samples_length) {
    Sample sample;

    if (device_shading_mode == SHADING_ALBEDO) {
      sample = load_active_sample_no_temporal_hint(device_active_samples + id);
      if (!(sample.state.x & 0x0100)) return;

      if (sample.hit_id == 0xffffffff) {
        sample.result = get_sky_color(sample.ray);
      } else {
        sample.origin = add_vector(sample.origin, scale_vector(sample.ray, sample.depth));

        const float4* hit_address = (float4*)(device_scene.triangles + sample.hit_id);
        const float4 t1 = __ldg(hit_address);
        const float4 t2 = __ldg(hit_address + 1);
        const float4 t3 = __ldg(hit_address + 2);
        const float4 t5 = __ldg(hit_address + 4);
        const float4 t6 = __ldg(hit_address + 5);
        const float4 t7 = __ldg(hit_address + 6);

        vec3 vertex = get_vector(t1.x, t1.y, t1.z);
        vec3 edge1 = get_vector(t1.w, t2.x, t2.y);
        vec3 edge2 = get_vector(t2.z, t2.w, t3.x);

        vec3 normal = get_coordinates_in_triangle(vertex, edge1, edge2, sample.origin);

        const float lambda = normal.x;
        const float mu = normal.y;

        UV vertex_texture = get_UV(t5.z, t5.w);
        UV edge1_texture = get_UV(t6.x, t6.y);
        UV edge2_texture = get_UV(t6.z, t6.w);

        const UV tex_coords = lerp_uv(vertex_texture, edge1_texture, edge2_texture, lambda, mu);

        const int texture_object = __float_as_int(t7.w);

        const ushort4 maps = __ldg((ushort4*)(device_texture_assignments + texture_object));

        sample.result = get_color(0.0f, 0.0f, 0.0f);

        if (maps.y) {
          const float4 illuminance_f = tex2D<float4>(device_illuminance_atlas[maps.y], tex_coords.u, 1.0f - tex_coords.v);
          sample.result = add_color(sample.result, get_color(illuminance_f.x, illuminance_f.y, illuminance_f.z));
        }
        if (maps.x) {
          const float4 albedo_f = tex2D<float4>(device_albedo_atlas[maps.x], tex_coords.u, 1.0f - tex_coords.v);
          sample.result = add_color(sample.result, get_color(albedo_f.x, albedo_f.y, albedo_f.z));
        } else {
          sample.result = add_color(sample.result, get_color(0.9f, 0.9f, 0.9f));
        }
      }
    } else if (device_shading_mode == SHADING_DEPTH) {
      sample = load_active_sample_no_temporal_hint(device_active_samples + id);
      if (!(sample.state.x & 0x0100)) return;

      const float value = __saturatef((1.0f/sample.depth) * 2.0f);
      sample.result = get_color(value, value, value);
    } else if (device_shading_mode == SHADING_NORMAL) {
      sample = load_active_sample_no_temporal_hint(device_active_samples + id);
      if (!(sample.state.x & 0x0100)) return;

      if (sample.hit_id == 0xffffffff) {
        sample.result = get_color(0.0f, 0.0f, 0.0f);
      } else {
        sample.origin = add_vector(sample.origin, scale_vector(sample.ray, sample.depth));

        const float4* hit_address = (float4*)(device_scene.triangles + sample.hit_id);

        const float4 t1 = __ldg(hit_address);
        const float4 t2 = __ldg(hit_address + 1);
        const float4 t3 = __ldg(hit_address + 2);
        const float4 t4 = __ldg(hit_address + 3);
        const float4 t5 = __ldg(hit_address + 4);

        vec3 vertex = get_vector(t1.x, t1.y, t1.z);
        vec3 edge1 = get_vector(t1.w, t2.x, t2.y);
        vec3 edge2 = get_vector(t2.z, t2.w, t3.x);

        vec3 normal = get_coordinates_in_triangle(vertex, edge1, edge2, sample.origin);

        const float lambda = normal.x;
        const float mu = normal.y;

        vec3 vertex_normal = get_vector(t3.y, t3.z, t3.w);
        vec3 edge1_normal = get_vector(t4.x, t4.y, t4.z);
        vec3 edge2_normal = get_vector(t4.w, t5.x, t5.y);

        normal = lerp_normals(vertex_normal, edge1_normal, edge2_normal, lambda, mu);

        sample.result = get_color(__saturatef(normal.x), __saturatef(normal.y), __saturatef(normal.z));
      }
    }

    sample.albedo_buffer = get_color(0.0f, 0.0f, 0.0f);

    sample.random_index++;

    sample.state.y--;

    if (sample.state.y <= 0) {
      sample.state.x &= 0;
    }

    sample = get_starting_ray(sample, id);

    if (sample.state.x & 0x0100) {
      if (store_id < id) __stcs((unsigned short*)(device_active_samples + id) + 12, 0);
      store_active_sample_no_temporal_hint(sample, device_active_samples + store_id);
      store_id += blockDim.x * gridDim.x;
    } else {
      const unsigned int address = (sample.index.x + sample.index.y * device_width) * device_samples_per_sample + (sample.state.x >> 11);
      store_finished_sample_no_temporal_hint(sample, device_finished_samples + address);
      atomicSub((int32_t*)&device_sample_offset, 1);
      __stcs((unsigned short*)(device_active_samples + id) + 12, 0);
    }

    id += blockDim.x * gridDim.x;
  }
}


#endif /* CU_KERNELS_H */
