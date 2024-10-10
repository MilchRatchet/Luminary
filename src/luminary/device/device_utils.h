#ifndef LUMINARY_DEVICE_UTILS_H
#define LUMINARY_DEVICE_UTILS_H

#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>

#include "device_structs.h"
#include "utils.h"

#define LUMINARY_MAX_NUM_DEVICES 4
#define OPTIXRT_NUM_GROUPS 3
#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 2048

#define OPTIX_VALIDATION
#define NO_LUMINARY_BVH

#define STARS_GRID_LD 64

////////////////////////////////////////////////////////////////////
// Failure handles
////////////////////////////////////////////////////////////////////

#define CUDA_FAILURE_HANDLE(command)                                                                                                       \
  {                                                                                                                                        \
    const cudaError_t __cuda_err = command;                                                                                                \
    if (__cuda_err != cudaSuccess) {                                                                                                       \
      __RETURN_ERROR(                                                                                                                      \
        LUMINARY_ERROR_CUDA, "CUDA returned error \"%s\" (%s) in call (%s)", cudaGetErrorName(__cuda_err), cudaGetErrorString(__cuda_err), \
        #command);                                                                                                                         \
    }                                                                                                                                      \
  }

#define OPTIX_FAILURE_HANDLE(command)                                                                                                 \
  {                                                                                                                                   \
    const OptixResult __optix_err = command;                                                                                          \
    if (__optix_err != OPTIX_SUCCESS) {                                                                                               \
      __RETURN_ERROR(                                                                                                                 \
        LUMINARY_ERROR_OPTIX, "Optix returned error \"%s\"(%d) in call (%s)", optixGetErrorName(__optix_err), __optix_err, #command); \
    }                                                                                                                                 \
  }

#define OPTIX_FAILURE_HANDLE_LOG(command, log)                                                                                        \
  {                                                                                                                                   \
    const OptixResult __optix_err = command;                                                                                          \
                                                                                                                                      \
    if (__optix_err != OPTIX_SUCCESS) {                                                                                               \
      error_message("Optix returned message: %s", log);                                                                               \
      __RETURN_ERROR(                                                                                                                 \
        LUMINARY_ERROR_OPTIX, "Optix returned error \"%s\"(%d) in call (%s)", optixGetErrorName(__optix_err), __optix_err, #command); \
    }                                                                                                                                 \
  }

struct Mat3x3 {
  float f11;
  float f12;
  float f13;
  float f21;
  float f22;
  float f23;
  float f31;
  float f32;
  float f33;
} typedef Mat3x3;

struct Mat3x4 {
  float f11;
  float f12;
  float f13;
  float f14;
  float f21;
  float f22;
  float f23;
  float f24;
  float f31;
  float f32;
  float f33;
  float f34;
} typedef Mat3x4;

struct Quad {
  vec3 vertex;
  vec3 edge1;
  vec3 edge2;
  vec3 normal;
} typedef Quad;

enum GBufferFlags {
  G_BUFFER_FLAG_REFRACTION_IS_INSIDE = 0b1,
  G_BUFFER_FLAG_COLORED_DIELECTRIC   = 0b10,
  G_BUFFER_FLAG_USE_LIGHT_RAYS       = 0b100
} typedef GBufferFlags;

struct GBufferData {
  uint32_t instance_id;
  uint16_t tri_id;
  RGBAF albedo;
  RGBF emission;
  vec3 position;
  vec3 V;
  vec3 normal;
  float roughness;
  float metallic;
  uint8_t state;
  uint8_t flags;
  /* IOR of medium in direction of V. */
  float ior_in;
  /* IOR of medium on the other side. */
  float ior_out;
} typedef GBufferData;

////////////////////////////////////////////////////////////////////
// Kernel passing structs
////////////////////////////////////////////////////////////////////

struct ShadingTask {
  uint32_t instance_id;
  ushort2 index;
  vec3 position;  // (Origin if sky)
  vec3 ray;
} typedef ShadingTask;
LUM_STATIC_SIZE_ASSERT(ShadingTask, 0x20);

struct ShadingTaskAuxData {
  uint16_t tri_id;
  uint8_t state;
  uint8_t padding;
} typedef ShadingTaskAuxData;
LUM_STATIC_SIZE_ASSERT(ShadingTaskAuxData, 0x04);

struct TraceTask {
  uint8_t state;
  uint8_t padding;
  uint16_t padding1;
  ushort2 index;
  vec3 origin;
  vec3 ray;
} typedef TraceTask;
LUM_STATIC_SIZE_ASSERT(TraceTask, 0x20);

INTERLEAVED_STORAGE struct TraceResult {
  float depth;
  uint32_t instance_id;
  uint16_t tri_id;
} typedef TraceResult;

struct LightTreeNode8Packed {
  vec3 base_point;
  int8_t exp_x;
  int8_t exp_y;
  int8_t exp_z;
  int8_t exp_confidence;
  uint32_t child_ptr;
  uint32_t light_ptr;
  uint32_t rel_point_x[2];
  uint32_t rel_point_y[2];
  uint32_t rel_point_z[2];
  uint32_t rel_energy[2];
  uint32_t confidence_light[2];
} typedef LightTreeNode8Packed;
LUM_STATIC_SIZE_ASSERT(LightTreeNode8Packed, 0x40);

////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////

struct DevicePointers {
  TraceTask* trace_tasks;
  ShadingTaskAuxData* aux_data;
  uint16_t* trace_counts;
  INTERLEAVED_STORAGE TraceResult* trace_results;
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
  RGBF* records;
  XRGB8* buffer_8bit;
  uint32_t* hit_id_history;
  const DeviceTextureObject* albedo_atlas;
  const DeviceTextureObject* luminance_atlas;
  const DeviceTextureObject* material_atlas;
  const DeviceTextureObject* normal_atlas;
  const DeviceTextureObject* cloud_noise;
  const DeviceTextureObject* sky_ms_luts;
  const DeviceTextureObject* sky_tm_luts;
  const DeviceTextureObject* sky_hdri_luts;
  const DeviceTextureObject* sky_moon_albedo_tex;
  const DeviceTextureObject* sky_moon_normal_tex;
  const DeviceTextureObject* bsdf_energy_lut;
  const uint16_t* bluenoise_1D;
  const uint32_t* bluenoise_2D;
  const float* bridge_lut;
  const DeviceMaterialCompressed* materials;
  INTERLEAVED_STORAGE const DeviceTriangle* triangles;
  const DeviceInstancelet* instances;
  const DeviceTransform* instance_transforms;
  const void** bottom_level_light_trees;
  const uint2** bottom_level_light_paths;
  const void* top_level_light_tree;
  const uint2* top_level_light_paths;
  const Quad* particle_quads;
  const Star* stars;
  const uint32_t* stars_offsets;
} typedef DevicePointers;

struct DeviceConstantMemory {
  DevicePointers ptrs;
  DeviceRendererSettings settings;
  DeviceCamera camera;
  DeviceOcean ocean;
  DeviceSky sky;
  DeviceCloud cloud;
  DeviceFog fog;
  DeviceParticles particles;
  DeviceToy toy;
  uint16_t user_selected_x;
  uint16_t user_selected_y;
  // Warning: This used to be a float, I will from now on have to emulate the old behaviour whenever we do undersampling
  uint32_t sample_id;
  uint32_t depth;
  uint32_t undersampling;
  uint32_t pixels_per_thread;
  OptixTraversableHandle optix_bvh;
  OptixTraversableHandle optix_bvh_shadow;
  OptixTraversableHandle optix_bvh_light;
  OptixTraversableHandle optix_bvh_particles;
  uint32_t non_instanced_triangle_count;
  uint32_t max_task_count;

  /*
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

  RGBF* bloom_scratch;
  int accumulate;
  BVHNode8* bvh_nodes;
  TraversalTriangle* bvh_triangles;
  Quad* particle_quads;
  LightTreeNode8Packed* light_tree_nodes_8;
  uint2* light_tree_paths;
  float* bridge_lut;
  */
} typedef DeviceConstantMemory;

#endif /* LUMINARY_DEVICE_UTILS_H */
