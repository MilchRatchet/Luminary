#ifndef LUMINARY_DEVICE_UTILS_H
#define LUMINARY_DEVICE_UTILS_H

#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>

#include "utils.h"

#define OPTIXRT_NUM_GROUPS 3
#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 2048

struct DeviceTexture;

////////////////////////////////////////////////////////////////////
// Kernel passing structs
////////////////////////////////////////////////////////////////////

struct ShadingTask {
  uint32_t hit_id;
  ushort2 index;
  vec3 position;  // (Origin if sky)
  vec3 ray;
} typedef ShadingTask;
LUM_STATIC_SIZE_ASSERT(ShadingTask, 32);

struct TraceTask {
  uint32_t padding;
  ushort2 index;
  vec3 origin;
  vec3 ray;
} typedef TraceTask;
LUM_STATIC_SIZE_ASSERT(TraceTask, 32);

struct TraceResult {
  float depth;
  uint32_t hit_id;
} typedef TraceResult;
LUM_STATIC_SIZE_ASSERT(TraceResult, 8);

////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////

struct DevicePointers {
  TraceTask* trace_tasks;
  uint16_t* trace_counts;
  TraceResult* trace_results;
  uint16_t* task_counts;
  uint16_t* task_offsets;
  uint32_t* ior_stack;
  float* frame_variance;
  RGBF* frame_accumulate;
  RGBF* frame_direct_buffer;
  RGBF* frame_direct_accumulate;
  RGBF* frame_indirect_buffer;
  RGBF* frame_indirect_accumulate;
  RGBF* frame_post;
  RGBF* frame_final;
  RGBF* albedo_buffer;
  RGBF* normal_buffer;
  RGBF* records;
  XRGB8* buffer_8bit;
  uint32_t* hit_id_history;
  uint8_t* state_buffer;
  const DeviceTexture* albedo_atlas;
  const DeviceTexture* luminance_atlas;
  const DeviceTexture* material_atlas;
  const DeviceTexture* normal_atlas;
  const DeviceTexture* cloud_noise;
  const DeviceTexture* sky_ms_luts;
  const DeviceTexture* sky_tm_luts;
  const DeviceTexture* sky_hdri_luts;
  const DeviceTexture* sky_moon_albedo_tex;
  const DeviceTexture* sky_moon_normal_tex;
  const DeviceTexture* bsdf_energy_lut;
  const uint16_t* bluenoise_1D;
  const uint32_t* bluenoise_2D;
} typedef DevicePointers;

struct DeviceConstantMemory {
  DevicePointers ptrs;
  Scene scene;
  RISSettings ris_settings;
  BridgeSettings bridge_settings;
  uint16_t user_selected_x;
  uint16_t user_selected_y;
  int max_ray_depth;
  int pixels_per_thread;
  int depth;
  float temporal_frames;
  int undersampling;
  int denoiser;
  int width;
  int height;
  int internal_width;
  int internal_height;
  vec3 sun_pos;
  vec3 moon_pos;
  int shading_mode;
  OutputVariable output_variable;
  RGBF* bloom_scratch;
  RayEmitter emitter;
  int accumulate;
  OptixTraversableHandle optix_bvh;
  OptixTraversableHandle optix_bvh_shadow;
  OptixTraversableHandle optix_bvh_light;
  OptixTraversableHandle optix_bvh_particles;
  Node8* bvh_nodes;
  TraversalTriangle* bvh_triangles;
  Quad* particle_quads;
  LightTreeNode8Packed* light_tree_nodes_8;
  uint2* light_tree_paths;
  float* bridge_lut;
} typedef DeviceConstantMemory;

#endif /* LUMINARY_DEVICE_UTILS_H */
