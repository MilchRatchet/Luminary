#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include "log.h"
#include "utils.h"

#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 1024

#define OPTIX_CHECK(call)                                                \
  {                                                                      \
    OptixResult res = call;                                              \
                                                                         \
    if (res != OPTIX_SUCCESS) {                                          \
      crash_message("Optix returned error %d in call (%s)", res, #call); \
    }                                                                    \
  }

#define gpuErrchk(ans)                                         \
  {                                                            \
    if (ans != cudaSuccess) {                                  \
      crash_message("GPUassert: %s", cudaGetErrorString(ans)); \
    }                                                          \
  }

#ifndef eps
#define eps 0.001f
#endif /* eps */

#ifndef PRIMITIVES_H
struct vec3 {
  float x;
  float y;
  float z;
} typedef vec3;

struct Quaternion {
  float w;
  float x;
  float y;
  float z;
} typedef Quaternion;
#endif

// state is 16 bits the depth and the last 16 bits the random_index

// ray_xz is horizontal angle
struct GeometryTask {
  vec3 position;
  float ray_y;
  float ray_xz;
  uint32_t hit_id;
  ushort2 index;
  uint32_t state;
} typedef GeometryTask;

struct SkyTask {
  vec3 ray;
  ushort2 index;
} typedef SkyTask;

// Magnitude of ray gives distance
struct OceanTask {
  vec3 position;
  float ray_y;
  float ray_xz;
  float distance;
  ushort2 index;
  uint32_t state;
} typedef OceanTask;

struct ToyTask {
  vec3 position;
  vec3 ray;
  ushort2 index;
  uint32_t state;
} typedef ToyTask;

struct FogTask {
  vec3 position;
  float ray_y;
  float ray_xz;
  float distance;
  ushort2 index;
  uint32_t state;
} typedef FogTask;

struct TraceTask {
  vec3 origin;
  vec3 ray;
  ushort2 index;
  uint32_t state;
} typedef TraceTask;

struct TraceResult {
  float depth;
  uint32_t hit_id;
} typedef TraceResult;

//===========================================================================================
// Bit Masks
//===========================================================================================

#define RANDOM_INDEX 0x0000ffff
#define DEPTH_LEFT 0xffff0000
#define SKY_HIT 0xffffffff
#define OCEAN_HIT 0xfffffffe
#define TOY_HIT 0xfffffffd
#define FOG_HIT 0xfffffffc
#define TOY_LIGHT 0x1
#define SUN_LIGHT 0x0
#define TYPE_CAMERA 0x0
#define TYPE_LIGHT 0x1
#define TYPE_BOUNCE 0x2
#define STATE_ALBEDO 0b1

//===========================================================================================
// Device Variables
//===========================================================================================

__constant__ int device_max_ray_depth;

__constant__ Scene device_scene;

__constant__ int device_pixels_per_thread;

__constant__ int device_iteration_type;

__constant__ TraceTask* device_trace_tasks;

__constant__ TraceTask* device_light_trace;

__constant__ TraceTask* device_bounce_trace;

__constant__ TraceResult* device_trace_results;

__constant__ uint16_t* device_trace_count;

__constant__ uint16_t* device_light_trace_count;

__constant__ uint16_t* device_bounce_trace_count;

// 0: GeoCount 1: OceanCount 2: SkyCount 3: ToyCount 4: FogCount
__constant__ uint16_t* device_task_counts;

// 0: GeoCount 1: OceanCount 2: SkyCount 3: ToyCount 4: FogCount
__constant__ uint16_t* device_task_offsets;

__constant__ uint32_t* device_light_sample_history;

__constant__ curandStateXORWOW_t* device_sample_randoms;

__constant__ int device_temporal_frames;

__constant__ int device_lights_active;

__constant__ RGBF* device_frame_buffer;

__constant__ RGBF* device_frame_temporal;

__constant__ RGBF* device_frame_output;

__constant__ RGBF* device_frame_variance;

__constant__ RGBF* device_frame_bias_cache;

__constant__ RGBF* device_records;

__constant__ RGBF* device_light_records;

__constant__ RGBF* device_bounce_records;

__constant__ int device_denoiser;

__constant__ RGBF* device_albedo_buffer;

__constant__ XRGB8* device_frame_8bit;

__constant__ int device_width;

__constant__ int device_height;

__constant__ int device_amount;

__constant__ float device_step;

__constant__ float device_vfov;

__constant__ Quaternion device_camera_rotation;

__constant__ cudaTextureObject_t* device_albedo_atlas;

__constant__ cudaTextureObject_t* device_illuminance_atlas;

__constant__ cudaTextureObject_t* device_material_atlas;

__constant__ TextureAssignment* device_texture_assignments;

__constant__ vec3 device_sun;

__constant__ RGBF device_default_material;

__constant__ int device_shading_mode;

__constant__ RGBF* device_bloom_scratch;

__constant__ Jitter device_jitter;

__device__ Mat4x4 device_view_space;

__device__ Mat4x4 device_projection;

__constant__ vec3* device_raydir_buffer;

__constant__ TraceResult* device_trace_result_buffer;

__constant__ TraceResult* device_trace_result_temporal;

__constant__ int device_accum_mode;

__constant__ uint8_t* device_state_buffer;

//===========================================================================================
// Functions
//===========================================================================================

__device__ int get_task_address_of_thread(const int thread_id, const int block_id, const int number) {
  const int warp_id       = (((thread_id & 0x60) >> 5) + block_id * (THREADS_PER_BLOCK / 32));
  const int thread_offset = (thread_id & 0x1f);
  return 32 * device_pixels_per_thread * warp_id + 32 * number + thread_offset;
}

__device__ int get_task_address(const int number) {
  return get_task_address_of_thread(threadIdx.x, blockIdx.x, number);
}

__device__ int is_first_ray() {
  return (device_iteration_type == TYPE_CAMERA);
}

#endif /* CU_UTILS_H */
