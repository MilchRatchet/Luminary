#include "raytrace.h"

#include <cuda_runtime_api.h>

#include "buffer.h"
#include "device.h"
#include "utils.h"

#define gpuErrchk(ans)                                         \
  {                                                            \
    if (ans != cudaSuccess) {                                  \
      crash_message("GPUassert: %s", cudaGetErrorString(ans)); \
    }                                                          \
  }

/*
 * Computes whether light resampling is used.
 * @param instance RaytraceInstance
 */
void raytrace_update_light_resampling_active(RaytraceInstance* instance) {
  instance->light_resampling =
    instance->scene_gpu.material.lights_active || (instance->scene_gpu.toy.emissive && instance->scene_gpu.toy.active);

  device_update_symbol(light_resampling, instance->light_resampling);
}

void raytracing_init(RaytraceInstance** _instance, General general, TextureAtlas tex_atlas, Scene* scene) {
  RaytraceInstance* instance = (RaytraceInstance*) calloc(1, sizeof(RaytraceInstance));

  instance->width         = general.width;
  instance->height        = general.height;
  instance->output_width  = general.width;
  instance->output_height = general.height;

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

  instance->max_ray_depth   = general.max_ray_depth;
  instance->offline_samples = general.samples;
  instance->denoiser        = general.denoiser;
  instance->reservoir_size  = general.reservoir_size;

  instance->tex_atlas = tex_atlas;

  instance->scene_gpu          = *scene;
  instance->settings           = general;
  instance->shading_mode       = 0;
  instance->accum_mode         = TEMPORAL_ACCUMULATION;
  instance->spatial_samples    = 5;
  instance->spatial_iterations = 2;

  instance->atmo_settings.base_density           = scene->sky.base_density;
  instance->atmo_settings.ground_visibility      = scene->sky.ground_visibility;
  instance->atmo_settings.mie_density            = scene->sky.mie_density;
  instance->atmo_settings.mie_falloff            = scene->sky.mie_falloff;
  instance->atmo_settings.mie_g                  = scene->sky.mie_g;
  instance->atmo_settings.ozone_absorption       = scene->sky.ozone_absorption;
  instance->atmo_settings.ozone_density          = scene->sky.ozone_density;
  instance->atmo_settings.ozone_layer_thickness  = scene->sky.ozone_layer_thickness;
  instance->atmo_settings.rayleigh_density       = scene->sky.rayleigh_density;
  instance->atmo_settings.rayleigh_falloff       = scene->sky.rayleigh_falloff;
  instance->atmo_settings.multiscattering_factor = scene->sky.multiscattering_factor;

  device_buffer_init(&instance->light_trace);
  device_buffer_init(&instance->bounce_trace);
  device_buffer_init(&instance->light_trace_count);
  device_buffer_init(&instance->bounce_trace_count);
  device_buffer_init(&instance->trace_results);
  device_buffer_init(&instance->task_counts);
  device_buffer_init(&instance->task_offsets);
  device_buffer_init(&instance->light_sample_history);
  device_buffer_init(&instance->frame_output);
  device_buffer_init(&instance->frame_temporal);
  device_buffer_init(&instance->frame_buffer);
  device_buffer_init(&instance->frame_variance);
  device_buffer_init(&instance->albedo_buffer);
  device_buffer_init(&instance->normal_buffer);
  device_buffer_init(&instance->light_records);
  device_buffer_init(&instance->bounce_records);
  device_buffer_init(&instance->buffer_8bit);
  device_buffer_init(&instance->randoms);
  device_buffer_init(&instance->raydir_buffer);
  device_buffer_init(&instance->trace_result_buffer);
  device_buffer_init(&instance->state_buffer);
  device_buffer_init(&instance->light_samples_1);
  device_buffer_init(&instance->light_samples_2);
  device_buffer_init(&instance->light_eval_data);
  device_buffer_init(&instance->cloud_noise);
  device_buffer_init(&instance->sky_ms_luts);
  device_buffer_init(&instance->sky_tm_luts);

  device_buffer_malloc(instance->buffer_8bit, sizeof(XRGB8), instance->width * instance->height);

  device_malloc((void**) &(instance->scene_gpu.texture_assignments), sizeof(TextureAssignment) * scene->materials_length);
  device_malloc((void**) &(instance->scene_gpu.triangles), sizeof(Triangle) * instance->scene_gpu.triangles_length);
  device_malloc((void**) &(instance->scene_gpu.traversal_triangles), sizeof(TraversalTriangle) * instance->scene_gpu.triangles_length);
  device_malloc((void**) &(instance->scene_gpu.nodes), sizeof(Node8) * instance->scene_gpu.nodes_length);
  device_malloc((void**) &(instance->scene_gpu.triangle_lights), sizeof(TriangleLight) * instance->scene_gpu.triangle_lights_length);

  gpuErrchk(cudaMemcpy(
    instance->scene_gpu.texture_assignments, scene->texture_assignments, sizeof(TextureAssignment) * scene->materials_length,
    cudaMemcpyHostToDevice));
  gpuErrchk(
    cudaMemcpy(instance->scene_gpu.triangles, scene->triangles, sizeof(Triangle) * scene->triangles_length, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(
    instance->scene_gpu.traversal_triangles, scene->traversal_triangles, sizeof(TraversalTriangle) * scene->triangles_length,
    cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(instance->scene_gpu.nodes, scene->nodes, sizeof(Node8) * scene->nodes_length, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(
    instance->scene_gpu.triangle_lights, scene->triangle_lights, sizeof(TriangleLight) * scene->triangle_lights_length,
    cudaMemcpyHostToDevice));

  device_update_symbol(texture_assignments, instance->scene_gpu.texture_assignments);

  sky_generate_LUTs(instance);
  clouds_generate(instance);

  raytrace_update_light_resampling_active(instance);
  allocate_buffers(instance);
  device_bloom_allocate_mips(instance);
  update_device_pointers(instance);
  device_initialize_random_generators();
  prepare_trace(instance);
  update_temporal_matrix(instance);

  instance->snap_resolution = SNAP_RESOLUTION_RENDER;

  *_instance = instance;
}
