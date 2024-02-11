#ifndef CU_KERNELS_H
#define CU_KERNELS_H

#include <cuda_runtime_api.h>

#include "utils.cuh"

__global__ void generate_trace_tasks();
__global__ void balance_trace_tasks();
__global__ void preprocess_trace_tasks();
__global__ void process_sky_inscattering_tasks();
__global__ void postprocess_trace_tasks();
__global__ void convert_RGBF_to_XRGB8(
  const RGBF* source, XRGB8* dest, const int width, const int height, const int ld, const OutputVariable output_variable);
__global__ void restir_candidates_pool_generation();
__global__ void process_trace_tasks();
__global__ void process_toy_tasks();
__global__ void process_debug_toy_tasks();
__global__ void initialize_randoms();
__global__ void process_debug_sky_tasks();
__global__ void process_sky_tasks();
__global__ void process_debug_ocean_tasks();
__global__ void process_ocean_tasks();
__global__ void particle_process_debug_tasks();
__global__ void particle_process_tasks();
__global__ void particle_kernel_generate(
  const uint32_t count, float size, const float size_variation, float4* vertex_buffer, uint32_t* index_buffer, Quad* quads);
__global__ void process_debug_geometry_tasks();
__global__ void process_geometry_tasks();
__global__ void volume_process_tasks();
__global__ void volume_process_events();
__global__ void clouds_render_tasks();
__global__ void temporal_reprojection();
__global__ void temporal_accumulation_aov(const RGBF* buffer, RGBF* accumulate);
__global__ void temporal_accumulation();
__global__ void image_upsample(
  const RGBF* source, const int sw, const int sh, RGBF* target, const RGBF* base, const int tw, const int th, const float sa,
  const float sb);
__global__ void image_downsample(const RGBF* source, const int sw, const int sh, RGBF* target, const int tw, const int th);
__global__ void image_downsample_threshold(
  const RGBF* source, const int sw, const int sh, RGBF* target, const int tw, const int th, const float thresh);
__global__ void mipmap_generate_level_2D_RGBAF(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height);
__global__ void mipmap_generate_level_3D_RGBAF(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height, int depth);
__global__ void mipmap_generate_level_2D_RGBA16(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height);
__global__ void mipmap_generate_level_3D_RGBA16(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height, int depth);
__global__ void mipmap_generate_level_2D_RGBA8(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height);
__global__ void mipmap_generate_level_3D_RGBA8(cudaTextureObject_t src, cudaSurfaceObject_t dst, int width, int height, int depth);
__global__ void camera_lens_flare_halo(const RGBF* src, const int sw, const int sh, RGBF* target, const int tw, const int th);
__global__ void camera_lens_flare_ghosts(const RGBF* source, const int sw, const int sh, RGBF* target, const int tw, const int th);
__global__ void cloud_noise_generate_shape(const int dim, uint8_t* tex);
__global__ void cloud_noise_generate_detail(const int dim, uint8_t* tex);
__global__ void cloud_noise_generate_weather(const int dim, const float seed, uint8_t* tex);
__global__ void omm_level_0_format_4(uint8_t* dst, uint8_t* level_record);
__global__ void omm_refine_format_4(uint8_t* dst, const uint8_t* src, uint8_t* level_record, const uint32_t src_level);
__global__ void omm_gather_array_format_4(
  uint8_t* dst, const uint8_t* src, const uint32_t level, const uint8_t* level_record, const OptixOpacityMicromapDesc* desc);
__global__ void dmm_precompute_indices(uint32_t* dst);
__global__ void dmm_setup_vertex_directions(half* dst);
__global__ void dmm_build_level_3_format_64(uint8_t* dst, const uint32_t* mapping, const uint32_t count);

#endif /* CU_KERNELS_H */
