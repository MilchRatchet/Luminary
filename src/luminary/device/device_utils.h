#ifndef LUMINARY_DEVICE_UTILS_H
#define LUMINARY_DEVICE_UTILS_H

#include "device_structs.h"
#include "utils.h"

#define LUMINARY_MAX_NUM_DEVICES 4
#define OPTIXRT_NUM_GROUPS 3
#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 2048

#define OPTIX_VALIDATION
#define NO_LUMINARY_BVH

/*
 * Annotation to signal that this pointer is a device allocation. Dereferencing this pointer on the host is undefined behaviour.
 */
#define DEVICE

/*
 * Annotation to signal that this function may only be called from within a valid device context.
 */
#define DEVICE_CTX_FUNC

#define STARS_GRID_LD 64

////////////////////////////////////////////////////////////////////
// Failure handles
////////////////////////////////////////////////////////////////////

#define CUDA_FAILURE_HANDLE(command)                                                                                               \
  {                                                                                                                                \
    const CUresult __cuda_err = (command);                                                                                         \
    if (__cuda_err != CUDA_SUCCESS) {                                                                                              \
      const char* __error_name   = (const char*) 0;                                                                                \
      const char* __error_string = (const char*) 0;                                                                                \
      cuGetErrorName(__cuda_err, &__error_name);                                                                                   \
      cuGetErrorString(__cuda_err, &__error_string);                                                                               \
      __RETURN_ERROR(LUMINARY_ERROR_CUDA, "CUDA returned error \"%s\" (%s) in call (%s)", __error_name, __error_string, #command); \
    }                                                                                                                              \
  }

#define OPTIX_FAILURE_HANDLE(command)                                                                                                 \
  {                                                                                                                                   \
    const OptixResult __optix_err = (command);                                                                                        \
    if (__optix_err != OPTIX_SUCCESS) {                                                                                               \
      __RETURN_ERROR(                                                                                                                 \
        LUMINARY_ERROR_OPTIX, "Optix returned error \"%s\"(%d) in call (%s)", optixGetErrorName(__optix_err), __optix_err, #command); \
    }                                                                                                                                 \
  }

#define OPTIX_FAILURE_HANDLE_LOG(command, log)                                                                                        \
  {                                                                                                                                   \
    const OptixResult __optix_err = (command);                                                                                        \
                                                                                                                                      \
    if (__optix_err != OPTIX_SUCCESS) {                                                                                               \
      error_message("Optix returned message: %s", log);                                                                               \
      __RETURN_ERROR(                                                                                                                 \
        LUMINARY_ERROR_OPTIX, "Optix returned error \"%s\"(%d) in call (%s)", optixGetErrorName(__optix_err), __optix_err, #command); \
    }                                                                                                                                 \
  }

////////////////////////////////////////////////////////////////////
// CUDA data types
////////////////////////////////////////////////////////////////////

// CUDA already defines these. We can query that by checking for C++.
#ifndef __cplusplus
struct ushort2 {
  uint16_t x;
  uint16_t y;
} typedef ushort2;

struct uint2 {
  uint32_t x;
  uint32_t y;
} typedef uint2;
#endif /* __cplusplus */

////////////////////////////////////////////////////////////////////
// Misc data types
////////////////////////////////////////////////////////////////////

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
  DEVICE TraceTask* trace_tasks;
  DEVICE ShadingTaskAuxData* aux_data;
  DEVICE uint16_t* trace_counts;  // TODO: Remove and reuse inside task_counts
  DEVICE INTERLEAVED_STORAGE TraceResult* trace_results;
  DEVICE uint16_t* task_counts;
  DEVICE uint16_t* task_offsets;
  DEVICE uint32_t* ior_stack;
  DEVICE float* frame_variance;
  DEVICE RGBF* frame_accumulate;
  DEVICE RGBF* frame_direct_buffer;
  DEVICE RGBF* frame_direct_accumulate;
  DEVICE RGBF* frame_indirect_buffer;
  DEVICE RGBF* frame_indirect_accumulate;
  DEVICE RGBF* frame_post;
  DEVICE RGBF* frame_final;
  DEVICE RGBF* records;
  DEVICE uint32_t* hit_id_history;
  DEVICE XRGB8* buffer_8bit;
  DEVICE const DeviceTextureObject* textures;
  DEVICE const DeviceTextureObject* cloud_noise;
  DEVICE const DeviceTextureObject* sky_ms_luts;
  DEVICE const DeviceTextureObject* sky_tm_luts;
  DEVICE const DeviceTextureObject* sky_hdri_luts;
  DEVICE const DeviceTextureObject* bsdf_energy_lut;
  DEVICE const uint16_t* bluenoise_1D;
  DEVICE const uint32_t* bluenoise_2D;
  DEVICE const float* bridge_lut;
  DEVICE const DeviceMaterialCompressed* materials;
  DEVICE INTERLEAVED_STORAGE const DeviceTriangle* triangles;
  DEVICE const DeviceInstancelet* instances;
  DEVICE const DeviceTransform* instance_transforms;
  DEVICE const uint32_t* light_instance_map;
  DEVICE const LightTreeNode8Packed** bottom_level_light_trees;
  DEVICE const uint2** bottom_level_light_paths;
  DEVICE const LightTreeNode8Packed* top_level_light_tree;
  DEVICE const uint2* top_level_light_paths;
  DEVICE const Quad* particle_quads;
  DEVICE const Star* stars;
  DEVICE const uint32_t* stars_offsets;
} typedef DevicePointers;

enum DeviceConstantMemoryMember {
  DEVICE_CONSTANT_MEMORY_MEMBER_PTRS      = 0,
  DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS  = 1,
  DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA    = 2,
  DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN     = 3,
  DEVICE_CONSTANT_MEMORY_MEMBER_SKY       = 4,
  DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD     = 5,
  DEVICE_CONSTANT_MEMORY_MEMBER_FOG       = 6,
  DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES = 7,
  DEVICE_CONSTANT_MEMORY_MEMBER_TOY       = 8,
  DEVICE_CONSTANT_MEMORY_MEMBER_TASK_META = 9,
  DEVICE_CONSTANT_MEMORY_MEMBER_OPTIX_BVH = 10,
  DEVICE_CONSTANT_MEMORY_MEMBER_TRI_COUNT = 11,
  DEVICE_CONSTANT_MEMORY_MEMBER_MOON_TEX  = 12,
  DEVICE_CONSTANT_MEMORY_MEMBER_DYNAMIC   = 13,

  DEVICE_CONSTANT_MEMORY_MEMBER_COUNT
} typedef DeviceConstantMemoryMember;

struct DeviceConstantMemory {
  // DEVICE_CONSTANT_MEMORY_MEMBER_PTRS
  DevicePointers ptrs;
  // DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS
  DeviceRendererSettings settings;
  // DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA
  DeviceCamera camera;
  // DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN
  DeviceOcean ocean;
  // DEVICE_CONSTANT_MEMORY_MEMBER_SKY
  DeviceSky sky;
  // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD
  DeviceCloud cloud;
  // DEVICE_CONSTANT_MEMORY_MEMBER_FOG
  DeviceFog fog;
  // DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES
  DeviceParticles particles;
  // DEVICE_CONSTANT_MEMORY_MEMBER_TOY
  DeviceToy toy;
  // DEVICE_CONSTANT_MEMORY_MEMBER_TASK_META
  uint32_t pixels_per_thread;
  uint32_t max_task_count;
  // DEVICE_CONSTANT_MEMORY_MEMBER_OPTIX_BVH
  OptixTraversableHandle optix_bvh;
  OptixTraversableHandle optix_bvh_shadow;
  OptixTraversableHandle optix_bvh_light;
  OptixTraversableHandle optix_bvh_particles;
  // DEVICE_CONSTANT_MEMORY_MEMBER_TRI_COUNT
  uint32_t non_instanced_triangle_count;
  // DEVICE_CONSTANT_MEMORY_MEMBER_MOON_TEX
  DeviceTextureObject moon_albedo_tex;
  DeviceTextureObject moon_normal_tex;
  // DEVICE_CONSTANT_MEMORY_MEMBER_DYNAMIC
  uint16_t user_selected_x;
  uint16_t user_selected_y;
  // Warning: This used to be a float, I will from now on have to emulate the old behaviour whenever we do undersampling
  uint32_t sample_id;
  uint32_t depth;
  uint32_t undersampling;
} typedef DeviceConstantMemory;

#endif /* LUMINARY_DEVICE_UTILS_H */
