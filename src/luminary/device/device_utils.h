#ifndef LUMINARY_DEVICE_UTILS_H
#define LUMINARY_DEVICE_UTILS_H

#include "device_structs.h"
#include "utils.h"

#define LUMINARY_MAX_NUM_DEVICES 4
#define THREADS_PER_BLOCK 128

// #define OPTIX_VALIDATION
#define NO_LUMINARY_BVH

/*
 * Annotation to signal that this pointer is a device allocation. Dereferencing this pointer on the host is undefined behaviour.
 */
#define DEVICE

/*
 * Annotation to signal that this function may only be called from within a valid device context.
 */
#define DEVICE_CTX_FUNC

#define RECOMMENDED_TASKS_PER_THREAD 8
#define MINIMUM_TASKS_PER_THREAD (RECOMMENDED_TASKS_PER_THREAD >> 1)
#define MAXIMUM_TASKS_PER_THREAD (RECOMMENDED_TASKS_PER_THREAD << 1)

#define STARS_GRID_LD 64
#define BSDF_LUT_SIZE 32

#define LIGHT_TREE_LIGHT_ID_NULL (0xFFFFFFFF)
#define LIGHT_TREE_ROOT_MAX_CHILD_COUNT 128
#define LIGHT_TREE_MAX_CHILDREN_PER_SECTION 8
#define LIGHT_TREE_ROOT_MAX_NUM_SECTIONS (LIGHT_TREE_ROOT_MAX_CHILD_COUNT / LIGHT_TREE_MAX_CHILDREN_PER_SECTION)
#define LIGHT_TREE_CHILDREN_PER_NODE 8
#define LIGHT_TREE_NODE_SECTION_REL_SIZE (sizeof(DeviceLightTreeRootSection) / sizeof(DeviceLightTreeRootHeader))
#define LIGHT_GEO_MAX_SAMPLES 8
#define LIGHT_GEO_MAX_BRIDGE_LENGTH 8
#define LIGHT_SUN_CAUSTICS_MAX_SAMPLES 128
#define LIGHT_NUM_MICROTRIANGLES 64

static_assert(
  LIGHT_TREE_ROOT_MAX_CHILD_COUNT % LIGHT_TREE_MAX_CHILDREN_PER_SECTION == 0,
  "Light tree root node max children must be a multiple of max children per root section.");

#define OMM_REFINEMENT_NEEDED_FLAG (0x80)

#define UNDERSAMPLING_FIRST_SAMPLE_MASK 0x80
#define UNDERSAMPLING_STAGE_MASK 0x7C
#define UNDERSAMPLING_STAGE_SHIFT 2
#define UNDERSAMPLING_ITERATION_MASK 0x03

#define SPECTRAL_MIN_WAVELENGTH 360
#define SPECTRAL_MAX_WAVELENGTH 830

#define IOR_AIR (1.0003f)

#define TEXTURE_OBJECT_INVALID (0xFFFFFFFFFFFFFFFFull)

#ifdef __cplusplus
#define LUM_RESTRICT __restrict__
#else
#define LUM_RESTRICT restrict
#endif

////////////////////////////////////////////////////////////////////
// Failure handles
////////////////////////////////////////////////////////////////////

#define CUDA_STALL_VALIDATION

#ifdef CUDA_STALL_VALIDATION

extern ThreadStatus* __cuda_stall_validation_macro_walltime;

#define CUDA_FAILURE_HANDLE(command)                                                                                               \
  {                                                                                                                                \
    thread_status_start(__cuda_stall_validation_macro_walltime, "_DEBUG_CUDA_STALL_VALIDATION");                                   \
    const CUresult __cuda_err = (command);                                                                                         \
    double __macro_time;                                                                                                           \
    thread_status_get_time(__cuda_stall_validation_macro_walltime, &__macro_time);                                                 \
    if (__macro_time > 0.01) {                                                                                                     \
      warn_message("CUDA API call (%s) stalled for %fs", #command, __macro_time);                                                  \
    }                                                                                                                              \
    if (__cuda_err != CUDA_SUCCESS) {                                                                                              \
      const char* __error_name   = (const char*) 0;                                                                                \
      const char* __error_string = (const char*) 0;                                                                                \
      cuGetErrorName(__cuda_err, &__error_name);                                                                                   \
      cuGetErrorString(__cuda_err, &__error_string);                                                                               \
      __RETURN_ERROR(LUMINARY_ERROR_CUDA, "CUDA returned error \"%s\" (%s) in call (%s)", __error_name, __error_string, #command); \
    }                                                                                                                              \
  }
#else /* CUDA_STALL_VALIDATION */
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
#endif /* !CUDA_STALL_VALIDATION */

#define OPTIX_FAILURE_HANDLE(command)                                                                                                 \
  {                                                                                                                                   \
    const OptixResult __optix_err = (command);                                                                                        \
    if (__optix_err != OPTIX_SUCCESS) {                                                                                               \
      __RETURN_ERROR(                                                                                                                 \
        LUMINARY_ERROR_OPTIX, "Optix returned error \"%s\"(%d) in call (%s)", optixGetErrorName(__optix_err), __optix_err, #command); \
    }                                                                                                                                 \
  }

#define OPTIX_FAILURE_HANDLE_LOG(__macro_command, __macro_log, __macro_log_size)                                                       \
  {                                                                                                                                    \
    const size_t __macro_actual_log_size = __macro_log_size;                                                                           \
                                                                                                                                       \
    const OptixResult __macro_optix_err = (__macro_command);                                                                           \
                                                                                                                                       \
    if (__macro_optix_err != OPTIX_SUCCESS) {                                                                                          \
      error_message("Optix returned message: %s", __macro_log);                                                                        \
      __RETURN_ERROR(                                                                                                                  \
        LUMINARY_ERROR_OPTIX, "Optix returned error \"%s\"(%d) in call (%s)", optixGetErrorName(__macro_optix_err), __macro_optix_err, \
        #__macro_command);                                                                                                             \
    }                                                                                                                                  \
    else {                                                                                                                             \
      log_message("Optix returned message: %s", __macro_log);                                                                          \
    }                                                                                                                                  \
    memset(__macro_log, 0, __macro_actual_log_size);                                                                                   \
    __macro_log_size = __macro_actual_log_size;                                                                                        \
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

struct float4 {
  float x;
  float y;
  float z;
  float w;
} typedef float4;
#endif /* __cplusplus */

////////////////////////////////////////////////////////////////////
// Misc data types
////////////////////////////////////////////////////////////////////

typedef uint16_t BFloat16;
typedef uint16_t UnsignedBFloat16;
typedef uint2 PackedRecord;
typedef uint2 PackedMISPayload;
typedef uint2 PackedRayDirection;

#define PACKED_RECORD_BLACK ((PackedRecord) make_uint2(0, 0))

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

struct AGXCustomParams {
  float slope;
  float power;
  float saturation;
} typedef AGXCustomParams;

struct TriangleHandle {
  uint32_t instance_id;
  uint32_t tri_id;
} typedef TriangleHandle;

struct VolumeDescriptor {
  VolumeType type;
  RGBF absorption;
  RGBF scattering;
  float max_scattering;
  float dist;
  float max_height;
  float min_height;
} typedef VolumeDescriptor;

enum MaterialFlag {
  MATERIAL_FLAG_BASE_SUBSTRATE_OPAQUE      = 0,
  MATERIAL_FLAG_BASE_SUBSTRATE_TRANSLUCENT = 1,
  MATERIAL_FLAG_BASE_SUBSTRATE_MASK        = 1,
  MATERIAL_FLAG_REFRACTION_IS_INSIDE       = 0b10,
  MATERIAL_FLAG_METALLIC                   = 0b100,
  MATERIAL_FLAG_COLORED_TRANSPARENCY       = 0b1000,
  MATERIAL_FLAG_VOLUME_SCATTERED           = 0b10000
} typedef MaterialFlag;

#define MATERIAL_IS_SUBSTRATE_OPAQUE(__internal_macro_flags) \
  (((__internal_macro_flags) & MATERIAL_FLAG_BASE_SUBSTRATE_MASK) == MATERIAL_FLAG_BASE_SUBSTRATE_OPAQUE)
#define MATERIAL_IS_SUBSTRATE_TRANSLUCENT(__internal_macro_flags) \
  (((__internal_macro_flags) & MATERIAL_FLAG_BASE_SUBSTRATE_MASK) == MATERIAL_FLAG_BASE_SUBSTRATE_TRANSLUCENT)

struct GBufferMetaData {
  uint32_t instance_id;
  float depth;
  uint16_t rel_hit_x_bfloat16;
  uint16_t rel_hit_y_bfloat16;
  uint16_t rel_hit_z_bfloat16;
  uint16_t material_id;
} typedef GBufferMetaData;
LUM_STATIC_SIZE_ASSERT(GBufferMetaData, 0x10);

////////////////////////////////////////////////////////////////////
// Light Importance Sampling Structs
////////////////////////////////////////////////////////////////////

struct DeviceLightTreeNode {
  uint16_t x;
  uint16_t y;
  uint16_t z;
  uint16_t padding;
  int8_t exp_x;
  int8_t exp_y;
  int8_t exp_z;
  int8_t exp_std_dev;
  uint8_t num_lights;
  uint8_t padding1;
  uint16_t padding2;
  uint32_t child_ptr;
  uint32_t light_ptr;
  uint8_t rel_mean_x[LIGHT_TREE_MAX_CHILDREN_PER_SECTION];
  uint8_t rel_mean_y[LIGHT_TREE_MAX_CHILDREN_PER_SECTION];
  uint8_t rel_mean_z[LIGHT_TREE_MAX_CHILDREN_PER_SECTION];
  uint8_t rel_std_dev[LIGHT_TREE_MAX_CHILDREN_PER_SECTION];
  uint8_t rel_power[LIGHT_TREE_MAX_CHILDREN_PER_SECTION];
} typedef DeviceLightTreeNode;
LUM_STATIC_SIZE_ASSERT(DeviceLightTreeNode, 0x40);

struct DeviceLightTreeRootHeader {
  uint16_t x;
  uint16_t y;
  uint16_t z;
  uint16_t num_root_lights;
  uint16_t padding;
  uint8_t num_sections;
  uint8_t padding1;
  int8_t exp_x;
  int8_t exp_y;
  int8_t exp_z;
  int8_t exp_std_dev;
} typedef DeviceLightTreeRootHeader;
LUM_STATIC_SIZE_ASSERT(DeviceLightTreeRootHeader, 0x10);

struct DeviceLightTreeRootSection {
  uint8_t rel_mean_x[LIGHT_TREE_MAX_CHILDREN_PER_SECTION];
  uint8_t rel_mean_y[LIGHT_TREE_MAX_CHILDREN_PER_SECTION];
  uint8_t rel_mean_z[LIGHT_TREE_MAX_CHILDREN_PER_SECTION];
  uint8_t rel_std_dev[LIGHT_TREE_MAX_CHILDREN_PER_SECTION];
  uint16_t rel_power[LIGHT_TREE_MAX_CHILDREN_PER_SECTION];
} typedef DeviceLightTreeRootSection;
LUM_STATIC_SIZE_ASSERT(DeviceLightTreeRootSection, 0x30);

struct MISPayload {
  float sampling_probability;
  vec3 origin;
} typedef MISPayload;

struct DeviceLightSceneData {
  float total_power;
  uint32_t num_lights;
} typedef DeviceLightSceneData;

#define MAX_NUM_INDIRECT_BUCKETS 3

////////////////////////////////////////////////////////////////////
// Task State
////////////////////////////////////////////////////////////////////

enum TaskStateBufferIndex {
  TASK_STATE_BUFFER_INDEX_PRESORT,
  TASK_STATE_BUFFER_INDEX_POSTSORT,

  TASK_STATE_BUFFER_INDEX_COUNT,

  TASK_STATE_BUFFER_INDEX_DIRECT_LIGHT = TASK_STATE_BUFFER_INDEX_PRESORT,

  TASK_STATE_BUFFER_INDEX_DIRECT_LIGHT_COUNT
} typedef TaskStateBufferIndex;

// TODO: This needs to change when adding SSS.
typedef uint32_t DeviceIORStack;

struct DeviceTask {
  uint16_t state;
  uint16_t volume_id;
  ushort2 index;
  vec3 origin;
  vec3 ray;
} typedef DeviceTask;
LUM_STATIC_SIZE_ASSERT(DeviceTask, 0x20);

struct DeviceTaskTrace {
  TriangleHandle handle;
  float depth;
  DeviceIORStack ior_stack;
} typedef DeviceTaskTrace;
LUM_STATIC_SIZE_ASSERT(DeviceTaskTrace, 0x10);

struct DeviceTaskThroughput {
  PackedRecord record;
  PackedMISPayload payload;
} typedef DeviceTaskThroughput;
LUM_STATIC_SIZE_ASSERT(DeviceTaskThroughput, 0x10);

struct DeviceTaskState {
  DeviceTask task;
  DeviceTaskTrace trace_result;
  DeviceTaskThroughput throughput;
} typedef DeviceTaskState;
LUM_STATIC_SIZE_ASSERT(DeviceTaskState, 0x40);

////////////////////////////////////////////////////////////////////
// Task Direct Lighting
////////////////////////////////////////////////////////////////////

struct DeviceTaskDirectLightGeo {
  uint32_t light_id;
  RGBF light_color;
  vec3 ray;
  float dist;
} typedef DeviceTaskDirectLightGeo;
LUM_STATIC_SIZE_ASSERT(DeviceTaskDirectLightGeo, 0x20);

struct DeviceTaskDirectLightSun {
  PackedRecord light_color;
  PackedRayDirection ray;
} typedef DeviceTaskDirectLightSun;
LUM_STATIC_SIZE_ASSERT(DeviceTaskDirectLightSun, 0x10);

struct DeviceTaskDirectLightAmbient {
  PackedRecord light_color;
  PackedRayDirection ray;
} typedef DeviceTaskDirectLightAmbient;
LUM_STATIC_SIZE_ASSERT(DeviceTaskDirectLightAmbient, 0x10);

struct DeviceTaskDirectLightBridges {
  uint32_t light_id;
  RGBF light_color;
  uint32_t seed;
  Quaternion16 rotation;
  float scale;
} typedef DeviceTaskDirectLightBridges;
LUM_STATIC_SIZE_ASSERT(DeviceTaskDirectLightBridges, 0x20);

struct DeviceTaskDirectLight {
  union {
    DeviceTaskDirectLightGeo geo;
    DeviceTaskDirectLightBridges bridges;
  };
  DeviceTaskDirectLightSun sun;
  DeviceTaskDirectLightAmbient ambient;
} typedef DeviceTaskDirectLight;
LUM_STATIC_SIZE_ASSERT(DeviceTaskDirectLight, 0x40);

////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////

struct DevicePointers {
  DEVICE DeviceTaskState* LUM_RESTRICT task_states;
  DEVICE DeviceTaskDirectLight* LUM_RESTRICT task_direct_light;
  DEVICE uint16_t* LUM_RESTRICT trace_counts;
  DEVICE uint16_t* LUM_RESTRICT task_counts;
  DEVICE uint16_t* LUM_RESTRICT task_offsets;
  DEVICE RGBF* LUM_RESTRICT frame_current_result;
  DEVICE RGBF* LUM_RESTRICT frame_direct_buffer;
  DEVICE RGBF* LUM_RESTRICT frame_direct_accumulate;
  DEVICE RGBF* LUM_RESTRICT frame_indirect_buffer;
  DEVICE float* LUM_RESTRICT frame_indirect_accumulate_red[MAX_NUM_INDIRECT_BUCKETS];
  DEVICE float* LUM_RESTRICT frame_indirect_accumulate_green[MAX_NUM_INDIRECT_BUCKETS];
  DEVICE float* LUM_RESTRICT frame_indirect_accumulate_blue[MAX_NUM_INDIRECT_BUCKETS];
  DEVICE RGBF* LUM_RESTRICT frame_final;
  DEVICE GBufferMetaData* LUM_RESTRICT gbuffer_meta;
  DEVICE const DeviceTextureObject* LUM_RESTRICT textures;
  DEVICE const uint16_t* LUM_RESTRICT bluenoise_1D;
  DEVICE const uint32_t* LUM_RESTRICT bluenoise_2D;
  DEVICE const float* LUM_RESTRICT bridge_lut;
  DEVICE const DeviceMaterialCompressed* LUM_RESTRICT materials;
  DEVICE INTERLEAVED_STORAGE const DeviceTriangle** LUM_RESTRICT triangles;
  DEVICE const uint32_t* LUM_RESTRICT triangle_counts;
  DEVICE const DeviceTransform* LUM_RESTRICT instance_transforms;
  DEVICE const uint32_t* LUM_RESTRICT instance_mesh_id;
  DEVICE const DeviceLightTreeRootHeader* LUM_RESTRICT light_tree_root;
  DEVICE const DeviceLightTreeNode* LUM_RESTRICT light_tree_nodes;
  DEVICE const TriangleHandle* LUM_RESTRICT light_tree_tri_handle_map;
  DEVICE const DeviceLightSceneData* LUM_RESTRICT light_scene_data;
  DEVICE const Quad* LUM_RESTRICT particle_quads;
  DEVICE const Star* LUM_RESTRICT stars;
  DEVICE const uint32_t* LUM_RESTRICT stars_offsets;
  DEVICE const float* LUM_RESTRICT spectral_cdf;
  DEVICE const DeviceCameraInterface* LUM_RESTRICT camera_interfaces;
  DEVICE const DeviceCameraMedium* LUM_RESTRICT camera_media;
  DEVICE uint32_t* LUM_RESTRICT abort_flag;  // Could be used for general execution flags in the future
} typedef DevicePointers;

struct DeviceExecutionState {
  // Warning: This used to be a float, I will from now on have to emulate the old behaviour whenever we do undersampling
  uint32_t sample_id;
  uint32_t tile_id;
  uint16_t user_selected_x;
  uint16_t user_selected_y;
  uint8_t depth;
  uint8_t undersampling;
  uint32_t aggregate_sample_count;
} typedef DeviceExecutionState;

struct DeviceExecutionConfiguration {
  uint32_t num_blocks;
  uint32_t num_tasks_per_thread;
} typedef DeviceExecutionConfiguration;

enum DeviceConstantMemoryMember {
  DEVICE_CONSTANT_MEMORY_MEMBER_PTRS,
  DEVICE_CONSTANT_MEMORY_MEMBER_SETTINGS,
  DEVICE_CONSTANT_MEMORY_MEMBER_CAMERA,
  DEVICE_CONSTANT_MEMORY_MEMBER_OCEAN,
  DEVICE_CONSTANT_MEMORY_MEMBER_SKY,
  DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD,
  DEVICE_CONSTANT_MEMORY_MEMBER_FOG,
  DEVICE_CONSTANT_MEMORY_MEMBER_PARTICLES,
  DEVICE_CONSTANT_MEMORY_MEMBER_OPTIX_BVH,
  DEVICE_CONSTANT_MEMORY_MEMBER_MOON_TEX,
  DEVICE_CONSTANT_MEMORY_MEMBER_SKY_LUT_TEX,
  DEVICE_CONSTANT_MEMORY_MEMBER_SKY_HDRI_TEX,
  DEVICE_CONSTANT_MEMORY_MEMBER_BSDF_LUT_TEX,
  DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD_NOISE_TEX,
  DEVICE_CONSTANT_MEMORY_MEMBER_SPECTRAL_LUT_TEX,
  DEVICE_CONSTANT_MEMORY_MEMBER_CONFIG,
  DEVICE_CONSTANT_MEMORY_MEMBER_STATE,

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
  // DEVICE_CONSTANT_MEMORY_MEMBER_OPTIX_BVH
  OptixTraversableHandle optix_bvh;
  OptixTraversableHandle optix_bvh_shadow;
  OptixTraversableHandle optix_bvh_particles;
  // DEVICE_CONSTANT_MEMORY_MEMBER_MOON_TEX
  DeviceTextureObject moon_albedo_tex;
  DeviceTextureObject moon_normal_tex;
  // DEVICE_CONSTANT_MEMORY_MEMBER_SKY_LUT_TEX
  DeviceTextureObject sky_lut_transmission_high_tex;
  DeviceTextureObject sky_lut_transmission_low_tex;
  DeviceTextureObject sky_lut_multiscattering_high_tex;
  DeviceTextureObject sky_lut_multiscattering_low_tex;
  // DEVICE_CONSTANT_MEMORY_MEMBER_SKY_HDRI_TEX
  DeviceTextureObject sky_hdri_color_tex;
  DeviceTextureObject sky_hdri_shadow_tex;
  // DEVICE_CONSTANT_MEMORY_MEMBER_BSDF_LUT_TEX
  DeviceTextureObject bsdf_lut_conductor;
  DeviceTextureObject bsdf_lut_glossy;
  DeviceTextureObject bsdf_lut_dielectric;
  DeviceTextureObject bsdf_lut_dielectric_inv;
  // DEVICE_CONSTANT_MEMORY_MEMBER_CLOUD_NOISE_TEX
  DeviceTextureObject cloud_noise_shape_tex;
  DeviceTextureObject cloud_noise_detail_tex;
  DeviceTextureObject cloud_noise_weather_tex;
  // DEVICE_CONSTANT_MEMORY_MEMBER_SPECTRAL_LUT_TEX
  DeviceTextureObject spectral_xy_lut_tex;
  DeviceTextureObject spectral_z_lut_tex;
  // DEVICE_CONSTANT_MEMORY_MEMBER_CONFIG
  DeviceExecutionConfiguration config;
  // DEVICE_CONSTANT_MEMORY_MEMBER_STATE
  DeviceExecutionState state;
} typedef DeviceConstantMemory;

#endif /* LUMINARY_DEVICE_UTILS_H */
