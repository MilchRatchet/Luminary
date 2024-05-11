#include "raytrace.h"

#include <cuda_runtime_api.h>
#include <math.h>
#include <optix.h>
// This header must be included exactly be one translation unit
#include <optix_function_table_definition.h>
#include <stdio.h>
#include <string.h>

#include "buffer.h"
#include "ceb.h"
#include "denoise.h"
#include "device.h"
#include "light.h"
#include "optixrt.h"
#include "optixrt_particle.h"
#include "png.h"
#include "sky_defines.h"
#include "stars.h"
#include "struct_interleaving.h"
#include "texture.h"
#include "utils.h"

////////////////////////////////////////////////////////////////////
// OptiX Log Callback
////////////////////////////////////////////////////////////////////

static void _raytrace_optix_log_callback(unsigned int level, const char* tag, const char* message, void* cbdata) {
  (void) cbdata;

  switch (level) {
    case 1:
      print_crash("[OptiX Log Message][%s] %s", tag, message);
      break;
    case 2:
      print_error("[OptiX Log Message][%s] %s", tag, message);
      break;
    case 3:
      print_warn("[OptiX Log Message][%s] %s", tag, message);
      break;
    default:
      print_info("[OptiX Log Message][%s] %s", tag, message);
      break;
  }
}

////////////////////////////////////////////////////////////////////
// Math utility functions
////////////////////////////////////////////////////////////////////

static vec3 rotate_vector_by_quaternion(const vec3 v, const Quaternion q) {
  vec3 result;

  vec3 u;
  u.x = q.x;
  u.y = q.y;
  u.z = q.z;

  const float s = q.w;

  const float dot_uv = u.x * v.x + u.y * v.y + u.z * v.z;
  const float dot_uu = u.x * u.x + u.y * u.y + u.z * u.z;

  vec3 cross;

  cross.x = u.y * v.z - u.z * v.y;
  cross.y = u.z * v.x - u.x * v.z;
  cross.z = u.x * v.y - u.y * v.x;

  result.x = 2.0f * dot_uv * u.x + ((s * s) - dot_uu) * v.x + 2.0f * s * cross.x;
  result.y = 2.0f * dot_uv * u.y + ((s * s) - dot_uu) * v.y + 2.0f * s * cross.y;
  result.z = 2.0f * dot_uv * u.z + ((s * s) - dot_uu) * v.z + 2.0f * s * cross.z;

  return result;
}

/*
 * Computes a rotation quaternion from euler angles.
 * @param rotation Euler angles defining the rotation.
 * @result Rotation quaternion
 */
static Quaternion get_rotation_quaternion(const vec3 rotation) {
  const float alpha = rotation.x;
  const float beta  = rotation.y;
  const float gamma = rotation.z;

  const float cy = cosf(gamma * 0.5f);
  const float sy = sinf(gamma * 0.5f);
  const float cp = cosf(beta * 0.5f);
  const float sp = sinf(beta * 0.5f);
  const float cr = cosf(alpha * 0.5f);
  const float sr = sinf(alpha * 0.5f);

  Quaternion q;
  q.w = cr * cp * cy + sr * sp * sy;
  q.x = sr * cp * cy - cr * sp * sy;
  q.y = cr * sp * cy + sr * cp * sy;
  q.z = cr * cp * sy - sr * sp * cy;

  return q;
}

static vec3 angles_to_direction(const float altitude, const float azimuth) {
  vec3 dir;
  dir.x = cosf(azimuth) * cosf(altitude);
  dir.y = sinf(altitude);
  dir.z = sinf(azimuth) * cosf(altitude);

  return dir;
}

////////////////////////////////////////////////////////////////////
// Scene content related functions
////////////////////////////////////////////////////////////////////

/*
 * Computes value in halton sequence.
 * @param index Index in halton sequence.
 * @param base Base of halton sequence.
 * @result Value in halton sequence of base at index.
 */
static float halton(int index, int base) {
  float fraction = 1.0f;
  float result   = 0.0f;

  while (index > 0) {
    fraction /= base;
    result += fraction * (index % base);
    index = index / base;
  }

  return result;
}

static void raytrace_load_moon_textures(RaytraceInstance* instance) {
  uint64_t info = 0;

  void* moon_albedo_data;
  int64_t moon_albedo_data_length;
  ceb_access("moon_albedo.png", &moon_albedo_data, &moon_albedo_data_length, &info);

  if (info) {
    crash_message("Failed to load moon_albedo texture.");
  }

  void* moon_normal_data;
  int64_t moon_normal_data_length;
  ceb_access("moon_normal.png", &moon_normal_data, &moon_normal_data_length, &info);

  if (info) {
    crash_message("Failed to load moon_normal texture.");
  }

  TextureRGBA moon_albedo_tex = png_load(moon_albedo_data, moon_albedo_data_length, "moon_albedo.png");
  TextureRGBA moon_normal_tex = png_load(moon_normal_data, moon_normal_data_length, "moon_normal.png");

  texture_create_atlas(&instance->sky_moon_albedo_tex, &moon_albedo_tex, 1);
  texture_create_atlas(&instance->sky_moon_normal_tex, &moon_normal_tex, 1);
}

static void raytrace_load_bluenoise_texture(RaytraceInstance* instance) {
  uint64_t info = 0;

  void* bluenoise_1D_data;
  int64_t bluenoise_1D_data_length;
  ceb_access("bluenoise_1D.bin", &bluenoise_1D_data, &bluenoise_1D_data_length, &info);

  if (info) {
    crash_message("Failed to load bluenoise_1D texture.");
  }

  void* bluenoise_2D_data;
  int64_t bluenoise_2D_data_length;
  ceb_access("bluenoise_2D.bin", &bluenoise_2D_data, &bluenoise_2D_data_length, &info);

  if (info) {
    crash_message("Failed to load bluenoise_2D texture.");
  }

  device_buffer_malloc(instance->bluenoise_1D, 1, bluenoise_1D_data_length);
  device_buffer_upload(instance->bluenoise_1D, bluenoise_1D_data);

  device_buffer_malloc(instance->bluenoise_2D, 1, bluenoise_2D_data_length);
  device_buffer_upload(instance->bluenoise_2D, bluenoise_2D_data);
}

/*
 * Same as in math.cuh
 */
static float tent_filter_importance_sample(const float x) {
  if (x > 0.5f) {
    return 1.0f - sqrtf(2.0f) * sqrtf(1.0f - x);
  }
  else {
    return -1.0f + sqrtf(2.0f) * sqrtf(x);
  }
}

static float jitter_value(const float x) {
  return 0.5f + tent_filter_importance_sample(x);
}

void raytrace_update_ray_emitter(RaytraceInstance* instance) {
  RayEmitter emitter = instance->emitter;

  const Quaternion q = emitter.camera_rotation;
  const vec3 offset  = instance->scene.camera.pos;

  emitter.camera_rotation = get_rotation_quaternion(instance->scene.camera.rotation);

  // The following is legacy code that is only used for temporal reprojection.

  memset(&emitter.view_space, 0, sizeof(Mat4x4));

  emitter.view_space.f11 = 1.0f - (q.y * q.y * 2.0f + q.z * q.z * 2.0f);
  emitter.view_space.f12 = q.x * q.y * 2.0f + q.w * q.z * 2.0f;
  emitter.view_space.f13 = q.x * q.z * 2.0f - q.w * q.y * 2.0f;
  emitter.view_space.f21 = q.x * q.y * 2.0f - q.w * q.z * 2.0f;
  emitter.view_space.f22 = 1.0f - (q.x * q.x * 2.0f + q.z * q.z * 2.0f);
  emitter.view_space.f23 = q.y * q.z * 2.0f + q.w * q.x * 2.0f;
  emitter.view_space.f31 = q.x * q.z * 2.0f + q.w * q.y * 2.0f;
  emitter.view_space.f32 = q.y * q.z * 2.0f - q.w * q.x * 2.0f;
  emitter.view_space.f33 = 1.0f - (q.x * q.x * 2.0f + q.y * q.y * 2.0f);

  emitter.view_space.f14 = -(offset.x * emitter.view_space.f11 + offset.y * emitter.view_space.f12 + offset.z * emitter.view_space.f13);
  emitter.view_space.f24 = -(offset.x * emitter.view_space.f21 + offset.y * emitter.view_space.f22 + offset.z * emitter.view_space.f23);
  emitter.view_space.f34 = -(offset.x * emitter.view_space.f31 + offset.y * emitter.view_space.f32 + offset.z * emitter.view_space.f33);
  emitter.view_space.f44 = 1.0f;

  memset(&emitter.projection, 0, sizeof(Mat4x4));

  emitter.step = 2.0f * (instance->scene.camera.fov / instance->width);
  emitter.vfov = emitter.step * instance->height / 2.0f;

  const float z_far  = instance->scene.camera.far_clip_distance;
  const float z_near = 1.0f;

  emitter.projection.f11 = 1.0f / instance->scene.camera.fov;
  emitter.projection.f22 = 1.0f / emitter.vfov;
  emitter.projection.f33 = -(z_far + z_near) / (z_far - z_near);
  emitter.projection.f43 = -1.0f;
  emitter.projection.f34 = -(2.0f * z_far * z_near) / (z_far - z_near);

  emitter.jitter.prev_x = emitter.jitter.x;
  emitter.jitter.prev_y = emitter.jitter.y;
  emitter.jitter.x      = jitter_value(halton(instance->temporal_frames, 2));
  emitter.jitter.y      = jitter_value(halton(instance->temporal_frames, 3));

  instance->emitter = emitter;
  device_update_symbol(emitter, emitter);
}

void raytrace_update_toy_rotation(RaytraceInstance* instance) {
  instance->scene.toy.computed_rotation = get_rotation_quaternion(instance->scene.toy.rotation);
}

/*
 * Positions toy behind the camera.
 * @param instance RaytraceInstance to be used.
 */
static void toy_flashlight_set_position(RaytraceInstance* instance) {
  const Quaternion q = get_rotation_quaternion(instance->scene.camera.rotation);

  vec3 offset;
  offset.x = 0.0f;
  offset.y = -0.5f;
  offset.z = 0.0f;

  offset = rotate_vector_by_quaternion(offset, q);

  instance->scene.toy.position.x = instance->scene.camera.pos.x + offset.x;
  instance->scene.toy.position.y = instance->scene.camera.pos.y + offset.y;
  instance->scene.toy.position.z = instance->scene.camera.pos.z + offset.z;
}

/*
 * Positions toy in front of the camera.
 * @param instance RaytraceInstance to be used.
 */
void raytrace_center_toy_at_camera(RaytraceInstance* instance) {
  const Quaternion q = get_rotation_quaternion(instance->scene.camera.rotation);

  vec3 offset;
  offset.x = 0.0f;
  offset.y = 0.0f;
  offset.z = -1.0f;

  offset = rotate_vector_by_quaternion(offset, q);

  offset.x *= 3.0f * instance->scene.toy.scale;
  offset.y *= 3.0f * instance->scene.toy.scale;
  offset.z *= 3.0f * instance->scene.toy.scale;

  instance->scene.toy.position.x = instance->scene.camera.pos.x + offset.x;
  instance->scene.toy.position.y = instance->scene.camera.pos.y + offset.y;
  instance->scene.toy.position.z = instance->scene.camera.pos.z + offset.z;
}

/*
 * Computes sun position.
 * @param scene Scene which contains the sun properties.
 */
static void update_special_lights(const Scene scene) {
  vec3 sun              = angles_to_direction(scene.sky.altitude, scene.sky.azimuth);
  const float scale_sun = 1.0f / (sqrtf(sun.x * sun.x + sun.y * sun.y + sun.z * sun.z));
  sun.x *= scale_sun * SKY_SUN_DISTANCE;
  sun.y *= scale_sun * SKY_SUN_DISTANCE;
  sun.z *= scale_sun * SKY_SUN_DISTANCE;
  sun.y -= SKY_EARTH_RADIUS;
  sun.x -= scene.sky.geometry_offset.x;
  sun.y -= scene.sky.geometry_offset.y;
  sun.z -= scene.sky.geometry_offset.z;

  device_update_symbol(sun_pos, sun);

  vec3 moon              = angles_to_direction(scene.sky.moon_altitude, scene.sky.moon_azimuth);
  const float scale_moon = 1.0f / (sqrtf(moon.x * moon.x + moon.y * moon.y + moon.z * moon.z));
  moon.x *= scale_moon * SKY_MOON_DISTANCE;
  moon.y *= scale_moon * SKY_MOON_DISTANCE;
  moon.z *= scale_moon * SKY_MOON_DISTANCE;
  moon.y -= SKY_EARTH_RADIUS;
  moon.x -= scene.sky.geometry_offset.x;
  moon.y -= scene.sky.geometry_offset.y;
  moon.z -= scene.sky.geometry_offset.z;

  device_update_symbol(moon_pos, moon);
}

////////////////////////////////////////////////////////////////////
// Raytrace main functions
////////////////////////////////////////////////////////////////////

void raytrace_build_structures(RaytraceInstance* instance) {
  if (instance->scene.sky.hdri_active && !instance->scene.sky.hdri_initialized) {
    sky_hdri_generate_LUT(instance);
  }

  if (!instance->particles_instance.optix.initialized) {
    device_particle_generate(instance);
    optixrt_particle_init(instance);
  }

  if (instance->bvh_type == BVH_LUMINARY && !instance->luminary_bvh_initialized) {
    bvh_init(instance);
  }
}

void raytrace_execute(RaytraceInstance* instance) {
  device_update_symbol(temporal_frames, instance->temporal_frames);

  device_generate_tasks();

  if (instance->shading_mode) {
    device_execute_debug_kernels(instance);
  }
  else {
    device_execute_main_kernels(instance, 0);
    for (int i = 1; i <= instance->max_ray_depth; i++) {
      device_execute_main_kernels(instance, i);
    }
  }

  device_handle_accumulation(instance);

  gpuErrchk(cudaDeviceSynchronize());
}

static void _raytrace_gather_device_info(RaytraceInstance* instance) {
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  instance->device_info.global_mem_size = prop.totalGlobalMem;

  const int major = prop.major;
  const int minor = prop.minor;

  if (major < 6) {
    crash_message("Pre Pascal architecture GPU detected. This GPU is not supported.");
  }

  switch (major) {
    case 6: {
      if (minor == 0 || minor == 1 || minor == 2) {
        instance->device_info.arch            = DEVICE_ARCH_PASCAL;
        instance->device_info.rt_core_version = 0;
      }
      else {
        instance->device_info.arch            = DEVICE_ARCH_UNKNOWN;
        instance->device_info.rt_core_version = 0;
      }
    } break;
    case 7: {
      if (minor == 0 || minor == 2) {
        instance->device_info.arch            = DEVICE_ARCH_VOLTA;
        instance->device_info.rt_core_version = 0;
      }
      else if (minor == 5) {
        instance->device_info.arch            = DEVICE_ARCH_TURING;
        instance->device_info.rt_core_version = 1;

        // TU116 and TU117 do not have RT cores, these can be detected by searching for GTX in the name
        for (int i = 0; i < 256; i++) {
          if (prop.name[i] == 'G') {
            instance->device_info.rt_core_version = 0;
          }
        }
      }
      else {
        instance->device_info.arch            = DEVICE_ARCH_UNKNOWN;
        instance->device_info.rt_core_version = 0;
      }
    } break;
    case 8: {
      if (minor == 0) {
        // GA100 has no RT cores
        instance->device_info.arch            = DEVICE_ARCH_AMPERE;
        instance->device_info.rt_core_version = 0;
      }
      else if (minor == 6 || minor == 7) {
        instance->device_info.arch            = DEVICE_ARCH_AMPERE;
        instance->device_info.rt_core_version = 2;
      }
      else if (minor == 9) {
        instance->device_info.arch            = DEVICE_ARCH_ADA;
        instance->device_info.rt_core_version = 3;
      }
      else {
        instance->device_info.arch            = DEVICE_ARCH_UNKNOWN;
        instance->device_info.rt_core_version = 0;
      }
    } break;
    case 9: {
      if (minor == 0) {
        instance->device_info.arch            = DEVICE_ARCH_HOPPER;
        instance->device_info.rt_core_version = 0;
      }
      else {
        instance->device_info.arch            = DEVICE_ARCH_UNKNOWN;
        instance->device_info.rt_core_version = 0;
      }
    } break;
    default:
      instance->device_info.arch            = DEVICE_ARCH_UNKNOWN;
      instance->device_info.rt_core_version = 0;
      break;
  }

  if (instance->device_info.arch == DEVICE_ARCH_UNKNOWN) {
    warn_message(
      "Luminary failed to identify architecture of CUDA compute capability %d.%d. Some features may not be working.", major, minor);
  }

  log_message("===========DEVICE INFO===========");
  log_message("GLOBAL MEM SIZE:   %zu", instance->device_info.global_mem_size);
  log_message("ARCH:              %d", instance->device_info.arch);
  log_message("RT CORE VERSION:   %d", instance->device_info.rt_core_version);
  log_message("=================================");
}

void raytrace_init(RaytraceInstance** _instance, General general, TextureAtlas tex_atlas, Scene* scene, CommandlineOptions options) {
  RaytraceInstance* instance = (RaytraceInstance*) calloc(1, sizeof(RaytraceInstance));

  _raytrace_gather_device_info(instance);

  general.width  = (options.width) ? options.width : general.width;
  general.height = (options.height) ? options.height : general.height;

  instance->width         = general.width;
  instance->height        = general.height;
  instance->output_width  = general.width;
  instance->output_height = general.height;

  instance->user_selected_x = 0xFFFF;
  instance->user_selected_y = 0xFFFF;

  if (general.denoiser == DENOISING_UPSCALING) {
    if (instance->width * instance->height > 18144000) {
      error_message(
        "Internal resolution is too high for denoising with upscaling! The maximum is ~18144000 pixels. Upscaling is turned off!");
      general.denoiser = DENOISING_ON;
    }
    else {
      instance->output_width *= 2;
      instance->output_height *= 2;
    }
  }

  OPTIX_CHECK(optixInit());
  OPTIX_CHECK(optixDeviceContextCreate((CUcontext) 0, (OptixDeviceContextOptions*) 0, &instance->optix_ctx));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(instance->optix_ctx, _raytrace_optix_log_callback, (void*) 0, 3));

  optixrt_compile_kernel(instance->optix_ctx, (char*) "optix_kernels.ptx", &(instance->optix_kernel), options);
  optixrt_compile_kernel(instance->optix_ctx, (char*) "optix_kernels_trace_particle.ptx", &(instance->particles_instance.kernel), options);
  optixrt_compile_kernel(instance->optix_ctx, (char*) "optix_kernels_geometry.ptx", &(instance->optix_kernel_geometry), options);
  optixrt_compile_kernel(instance->optix_ctx, (char*) "optix_kernels_volume.ptx", &(instance->optix_kernel_volume), options);

  instance->max_ray_depth   = general.max_ray_depth;
  instance->offline_samples = general.samples;
  instance->denoiser        = general.denoiser;

  instance->tex_atlas = tex_atlas;

  instance->scene        = *scene;
  instance->settings     = general;
  instance->shading_mode = 0;
  instance->aov_mode     = options.aov_mode;
  instance->accum_mode   = TEMPORAL_ACCUMULATION;
  instance->bvh_type     = BVH_OPTIX;

  instance->ris_settings.initial_reservoir_size         = 16;
  instance->ris_settings.light_candidate_pool_size_log2 = 14;
  instance->ris_settings.num_light_rays                 = 1;

  instance->atmo_settings.base_density           = scene->sky.base_density;
  instance->atmo_settings.ground_visibility      = scene->sky.ground_visibility;
  instance->atmo_settings.mie_density            = scene->sky.mie_density;
  instance->atmo_settings.mie_falloff            = scene->sky.mie_falloff;
  instance->atmo_settings.mie_diameter           = scene->sky.mie_diameter;
  instance->atmo_settings.ozone_absorption       = scene->sky.ozone_absorption;
  instance->atmo_settings.ozone_density          = scene->sky.ozone_density;
  instance->atmo_settings.ozone_layer_thickness  = scene->sky.ozone_layer_thickness;
  instance->atmo_settings.rayleigh_density       = scene->sky.rayleigh_density;
  instance->atmo_settings.rayleigh_falloff       = scene->sky.rayleigh_falloff;
  instance->atmo_settings.multiscattering_factor = scene->sky.multiscattering_factor;

  device_buffer_init(&instance->trace_tasks);
  device_buffer_init(&instance->trace_counts);
  device_buffer_init(&instance->trace_results);
  device_buffer_init(&instance->task_counts);
  device_buffer_init(&instance->task_offsets);
  device_buffer_init(&instance->ior_stack);
  device_buffer_init(&instance->frame_buffer);
  device_buffer_init(&instance->frame_temporal);
  device_buffer_init(&instance->frame_variance);
  device_buffer_init(&instance->frame_accumulate);
  device_buffer_init(&instance->frame_output);
  device_buffer_init(&instance->frame_direct_buffer);
  device_buffer_init(&instance->frame_direct_accumulate);
  device_buffer_init(&instance->frame_indirect_buffer);
  device_buffer_init(&instance->frame_indirect_accumulate);
  device_buffer_init(&instance->albedo_buffer);
  device_buffer_init(&instance->normal_buffer);
  device_buffer_init(&instance->records);
  device_buffer_init(&instance->buffer_8bit);
  device_buffer_init(&instance->raydir_buffer);
  device_buffer_init(&instance->trace_result_buffer);
  device_buffer_init(&instance->state_buffer);
  device_buffer_init(&instance->light_candidates);
  device_buffer_init(&instance->cloud_noise);
  device_buffer_init(&instance->sky_ms_luts);
  device_buffer_init(&instance->sky_tm_luts);
  device_buffer_init(&instance->sky_hdri_luts);
  device_buffer_init(&instance->sky_moon_albedo_tex);
  device_buffer_init(&instance->sky_moon_normal_tex);
  device_buffer_init(&instance->bsdf_energy_lut);
  device_buffer_init(&instance->bluenoise_1D);
  device_buffer_init(&instance->bluenoise_2D);

  device_buffer_malloc(instance->buffer_8bit, sizeof(XRGB8), instance->width * instance->height);

  device_malloc((void**) &(instance->scene.materials), sizeof(PackedMaterial) * instance->scene.materials_count);
  device_malloc(
    (void**) &(instance->scene.triangles), INTERLEAVED_ALLOCATION_SIZE(sizeof(Triangle)) * instance->scene.triangle_data.triangle_count);
  device_malloc((void**) &(instance->scene.triangle_lights), sizeof(TriangleLight) * instance->scene.triangle_lights_count);
  device_malloc((void**) &(instance->scene.triangle_data.vertex_buffer), instance->scene.triangle_data.vertex_count * 4 * sizeof(float));
  device_malloc(
    (void**) &(instance->scene.triangle_data.index_buffer), instance->scene.triangle_data.triangle_count * 4 * sizeof(uint32_t));
  device_malloc((void**) &(instance->ris_settings.presampled_triangle_lights), sizeof(TriangleLight) * RIS_MAX_CANDIDATE_POOL_SIZE);

  Triangle* triangles_interleaved =
    (Triangle*) malloc(INTERLEAVED_ALLOCATION_SIZE(sizeof(Triangle)) * instance->scene.triangle_data.triangle_count);
  struct_triangles_interleave(triangles_interleaved, scene->triangles, instance->scene.triangle_data.triangle_count);

  gpuErrchk(cudaMemcpy(
    instance->scene.materials, scene->materials, sizeof(PackedMaterial) * instance->scene.materials_count, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(
    instance->scene.triangles, triangles_interleaved,
    INTERLEAVED_ALLOCATION_SIZE(sizeof(Triangle)) * instance->scene.triangle_data.triangle_count, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(
    instance->scene.triangle_lights, scene->triangle_lights, sizeof(TriangleLight) * instance->scene.triangle_lights_count,
    cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(
    instance->scene.triangle_data.vertex_buffer, scene->triangle_data.vertex_buffer,
    instance->scene.triangle_data.vertex_count * 4 * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(
    instance->scene.triangle_data.index_buffer, scene->triangle_data.index_buffer,
    instance->scene.triangle_data.triangle_count * 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));

  free(triangles_interleaved);

  device_update_symbol(aov_mode, instance->aov_mode);

  raytrace_load_moon_textures(instance);
  raytrace_load_bluenoise_texture(instance);

  device_sky_generate_LUTs(instance);
  device_cloud_noise_generate(instance);
  bsdf_compute_energy_lut(instance);
  stars_generate(instance);

  raytrace_allocate_buffers(instance);
  device_camera_post_init(instance);
  raytrace_update_device_pointers(instance);
  raytrace_prepare(instance);

  optixrt_init(instance, options);

  instance->snap_resolution = SNAP_RESOLUTION_RENDER;

  *_instance = instance;
}

void raytrace_reset(RaytraceInstance* instance) {
  device_camera_post_clear(instance);

  // If denoising was set to on but no denoise setup exists, then we do not create one either now
  const int allocate_denoise = instance->settings.denoiser && !(instance->denoiser && !instance->denoise_setup);

  if (instance->denoise_setup) {
    denoise_free(instance);
  }

  instance->width         = instance->settings.width;
  instance->height        = instance->settings.height;
  instance->output_width  = instance->settings.width;
  instance->output_height = instance->settings.height;
  instance->max_ray_depth = instance->settings.max_ray_depth;
  instance->denoiser      = instance->settings.denoiser;

  if (instance->denoiser == DENOISING_UPSCALING) {
    if (instance->width * instance->height > 18144000) {
      error_message(
        "Internal resolution is too high for denoising with upscaling! The maximum is ~18144000 pixels. Upscaling is turned off!");
      instance->denoiser = DENOISING_ON;
    }
    else {
      instance->output_width *= 2;
      instance->output_height *= 2;
    }
  }

  raytrace_allocate_buffers(instance);
  device_camera_post_init(instance);
  raytrace_update_device_pointers(instance);
  raytrace_prepare(instance);

  if (allocate_denoise) {
    denoise_create(instance);
  }

  log_message("Reset raytrace instance.");
}

/*
 * Updates and uploades all dynamic data. Must be called at least once before using trace.
 * @param instance RaytraceInstance to be used.
 */
void raytrace_prepare(RaytraceInstance* instance) {
  if (instance->scene.toy.flashlight_mode) {
    toy_flashlight_set_position(instance);
  }
  raytrace_update_device_scene(instance);
  device_update_symbol(shading_mode, instance->shading_mode);
  device_update_symbol(output_variable, instance->output_variable);
  update_special_lights(instance->scene);
  raytrace_update_ray_emitter(instance);
  device_update_symbol(accum_mode, instance->accum_mode);
  device_update_symbol(ris_settings, instance->ris_settings);
  device_update_symbol(user_selected_x, instance->user_selected_x);
  device_update_symbol(user_selected_y, instance->user_selected_y);
  raytrace_build_structures(instance);
}

/*
 * Allocates all pixel count related buffers.
 * @param instance RaytraceInstance to be used.
 */
void raytrace_allocate_buffers(RaytraceInstance* instance) {
  const unsigned int amount        = instance->width * instance->height;
  const unsigned int output_amount = instance->output_width * instance->output_height;

  device_update_symbol(width, instance->width);
  device_update_symbol(height, instance->height);
  device_update_symbol(output_width, instance->output_width);
  device_update_symbol(output_height, instance->output_height);
  device_update_symbol(max_ray_depth, instance->max_ray_depth);
  device_update_symbol(denoiser, instance->denoiser);

  device_buffer_malloc(instance->frame_buffer, sizeof(RGBF), amount);
  device_buffer_malloc(instance->frame_temporal, sizeof(RGBF), amount);
  device_buffer_malloc(instance->frame_variance, sizeof(float), amount);
  device_buffer_malloc(instance->frame_accumulate, sizeof(RGBF), amount);
  device_buffer_malloc(instance->frame_output, sizeof(RGBF), output_amount);
  device_buffer_malloc(instance->records, sizeof(RGBF), amount);

  if (instance->denoiser || instance->aov_mode) {
    device_buffer_malloc(instance->albedo_buffer, sizeof(RGBF), amount);
    device_buffer_malloc(instance->normal_buffer, sizeof(RGBF), amount);
  }

  if (instance->aov_mode) {
    device_buffer_malloc(instance->frame_direct_buffer, sizeof(RGBF), amount);
    device_buffer_malloc(instance->frame_direct_accumulate, sizeof(RGBF), amount);
    device_buffer_malloc(instance->frame_indirect_buffer, sizeof(RGBF), amount);
    device_buffer_malloc(instance->frame_indirect_accumulate, sizeof(RGBF), amount);
  }

  const int thread_count      = device_get_thread_count();
  const int pixels_per_thread = 1 + ((amount + thread_count - 1) / thread_count);
  const int max_task_count    = pixels_per_thread * thread_count;

  device_update_symbol(pixels_per_thread, pixels_per_thread);

  device_buffer_malloc(instance->trace_tasks, sizeof(TraceTask), max_task_count);
  device_buffer_malloc(instance->trace_counts, sizeof(uint16_t), thread_count);

  device_buffer_malloc(instance->trace_results, sizeof(TraceResult), max_task_count);
  device_buffer_malloc(instance->task_counts, sizeof(uint16_t), 7 * thread_count);
  device_buffer_malloc(instance->task_offsets, sizeof(uint16_t), 6 * thread_count);

  device_buffer_malloc(instance->ior_stack, sizeof(uint32_t), amount);
  device_buffer_malloc(instance->raydir_buffer, sizeof(vec3), amount);
  device_buffer_malloc(instance->trace_result_buffer, sizeof(TraceResult), amount);
  device_buffer_malloc(instance->state_buffer, sizeof(uint8_t), amount);

  device_buffer_malloc(instance->light_candidates, sizeof(uint32_t), RIS_MAX_CANDIDATE_POOL_SIZE);

  cudaMemset(device_buffer_get_pointer(instance->trace_result_buffer), 0, sizeof(TraceResult) * amount);
}

void raytrace_update_device_pointers(RaytraceInstance* instance) {
  DevicePointers ptrs;

  ptrs.trace_tasks               = (TraceTask*) device_buffer_get_pointer(instance->trace_tasks);
  ptrs.trace_counts              = (uint16_t*) device_buffer_get_pointer(instance->trace_counts);
  ptrs.trace_results             = (TraceResult*) device_buffer_get_pointer(instance->trace_results);
  ptrs.task_counts               = (uint16_t*) device_buffer_get_pointer(instance->task_counts);
  ptrs.task_offsets              = (uint16_t*) device_buffer_get_pointer(instance->task_offsets);
  ptrs.ior_stack                 = (uint32_t*) device_buffer_get_pointer(instance->ior_stack);
  ptrs.frame_buffer              = (RGBF*) device_buffer_get_pointer(instance->frame_buffer);
  ptrs.frame_temporal            = (RGBF*) device_buffer_get_pointer(instance->frame_temporal);
  ptrs.frame_variance            = (float*) device_buffer_get_pointer(instance->frame_variance);
  ptrs.frame_accumulate          = (RGBF*) device_buffer_get_pointer(instance->frame_accumulate);
  ptrs.frame_direct_buffer       = (RGBF*) device_buffer_get_pointer(instance->frame_direct_buffer);
  ptrs.frame_direct_accumulate   = (RGBF*) device_buffer_get_pointer(instance->frame_direct_accumulate);
  ptrs.frame_indirect_buffer     = (RGBF*) device_buffer_get_pointer(instance->frame_indirect_buffer);
  ptrs.frame_indirect_accumulate = (RGBF*) device_buffer_get_pointer(instance->frame_indirect_accumulate);
  ptrs.frame_output              = (RGBF*) device_buffer_get_pointer(instance->frame_output);
  ptrs.albedo_buffer             = (RGBF*) device_buffer_get_pointer(instance->albedo_buffer);
  ptrs.normal_buffer             = (RGBF*) device_buffer_get_pointer(instance->normal_buffer);
  ptrs.records                   = (RGBF*) device_buffer_get_pointer(instance->records);
  ptrs.buffer_8bit               = (XRGB8*) device_buffer_get_pointer(instance->buffer_8bit);
  ptrs.albedo_atlas              = (DeviceTexture*) device_buffer_get_pointer(instance->tex_atlas.albedo);
  ptrs.luminance_atlas           = (DeviceTexture*) device_buffer_get_pointer(instance->tex_atlas.luminance);
  ptrs.material_atlas            = (DeviceTexture*) device_buffer_get_pointer(instance->tex_atlas.material);
  ptrs.normal_atlas              = (DeviceTexture*) device_buffer_get_pointer(instance->tex_atlas.normal);
  ptrs.cloud_noise               = (DeviceTexture*) device_buffer_get_pointer(instance->cloud_noise);
  ptrs.raydir_buffer             = (vec3*) device_buffer_get_pointer(instance->raydir_buffer);
  ptrs.trace_result_buffer       = (TraceResult*) device_buffer_get_pointer(instance->trace_result_buffer);
  ptrs.state_buffer              = (uint8_t*) device_buffer_get_pointer(instance->state_buffer);
  ptrs.light_candidates          = (uint32_t*) device_buffer_get_pointer(instance->light_candidates);
  ptrs.sky_tm_luts               = (DeviceTexture*) device_buffer_get_pointer(instance->sky_tm_luts);
  ptrs.sky_ms_luts               = (DeviceTexture*) device_buffer_get_pointer(instance->sky_ms_luts);
  ptrs.sky_hdri_luts             = (DeviceTexture*) device_buffer_get_pointer(instance->sky_hdri_luts);
  ptrs.sky_moon_albedo_tex       = (DeviceTexture*) device_buffer_get_pointer(instance->sky_moon_albedo_tex);
  ptrs.sky_moon_normal_tex       = (DeviceTexture*) device_buffer_get_pointer(instance->sky_moon_normal_tex);
  ptrs.bsdf_energy_lut           = (DeviceTexture*) device_buffer_get_pointer(instance->bsdf_energy_lut);
  ptrs.bluenoise_1D              = (uint16_t*) device_buffer_get_pointer(instance->bluenoise_1D);
  ptrs.bluenoise_2D              = (uint32_t*) device_buffer_get_pointer(instance->bluenoise_2D);

  device_update_symbol(ptrs, ptrs);
  log_message("Updated device pointers.");
}

void raytrace_free_work_buffers(RaytraceInstance* instance) {
  gpuErrchk(cudaFree(instance->scene.materials));
  gpuErrchk(cudaFree(instance->scene.triangles));
  gpuErrchk(cudaFree(instance->scene.triangle_lights));
  gpuErrchk(cudaFree(instance->ris_settings.presampled_triangle_lights));
  device_buffer_free(instance->ior_stack);
  device_buffer_free(instance->trace_tasks);
  device_buffer_free(instance->trace_counts);
  device_buffer_free(instance->trace_results);
  device_buffer_free(instance->task_counts);
  device_buffer_free(instance->task_offsets);
  device_buffer_free(instance->frame_buffer);
  device_buffer_free(instance->frame_direct_buffer);
  device_buffer_free(instance->frame_indirect_buffer);
  device_buffer_free(instance->frame_temporal);
  device_buffer_free(instance->frame_variance);
  device_buffer_free(instance->records);
  device_buffer_free(instance->raydir_buffer);
  device_buffer_free(instance->trace_result_buffer);
  device_buffer_free(instance->state_buffer);

  gpuErrchk(cudaDeviceSynchronize());
}

void raytrace_free_output_buffers(RaytraceInstance* instance) {
  device_buffer_free(instance->frame_accumulate);

  if (instance->denoiser || instance->aov_mode) {
    device_buffer_free(instance->albedo_buffer);
    device_buffer_free(instance->normal_buffer);
  }

  if (instance->aov_mode) {
    device_buffer_free(instance->frame_direct_accumulate);
    device_buffer_free(instance->frame_indirect_accumulate);
  }

  device_camera_post_clear(instance);

  free(instance);
}

void raytrace_init_8bit_frame(RaytraceInstance* instance, const unsigned int width, const unsigned int height) {
  device_buffer_malloc(instance->buffer_8bit, sizeof(XRGB8), width * height);
  raytrace_update_device_pointers(instance);
}

void raytrace_free_8bit_frame(RaytraceInstance* instance) {
  device_buffer_free(instance->buffer_8bit);
}

void raytrace_update_device_scene(RaytraceInstance* instance) {
  raytrace_update_toy_rotation(instance);
  device_update_symbol(scene, instance->scene);
}

DeviceBuffer* raytrace_get_accumulate_buffer(RaytraceInstance* instance, OutputVariable output_variable) {
  switch (output_variable) {
    default:
    case OUTPUT_VARIABLE_BEAUTY:
      return instance->frame_accumulate;
    case OUTPUT_VARIABLE_ALBEDO_GUIDANCE:
      return instance->albedo_buffer;
    case OUTPUT_VARIABLE_NORMAL_GUIDANCE:
      return instance->normal_buffer;
    case OUTPUT_VARIABLE_DIRECT_LIGHTING:
      return instance->frame_direct_accumulate;
    case OUTPUT_VARIABLE_INDIRECT_LIGHTING:
      return instance->frame_indirect_accumulate;
  }
}

const char* raytrace_get_output_variable_name(OutputVariable output_variable) {
  switch (output_variable) {
    default:
    case OUTPUT_VARIABLE_BEAUTY:
      return "beauty";
    case OUTPUT_VARIABLE_ALBEDO_GUIDANCE:
      return "albedo";
    case OUTPUT_VARIABLE_NORMAL_GUIDANCE:
      return "normal";
    case OUTPUT_VARIABLE_DIRECT_LIGHTING:
      return "direct";
    case OUTPUT_VARIABLE_INDIRECT_LIGHTING:
      return "indirect";
  }
}
