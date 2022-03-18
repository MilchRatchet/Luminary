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

#define SKY_EARTH_RADIUS 6371.0f
#define SKY_SUN_RADIUS 696340.0f
#define SKY_SUN_DISTANCE 149597870.0f

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

struct RestirSample {
  uint32_t id;
  float weight;
} typedef RestirSample;

struct RestirEvalData {
  vec3 position;
  vec3 normal;
} typedef RestirEvalData;

// state is 16 bits the depth and the last 16 bits the random_index

// TaskCounts: 0: GeoCount 1: OceanCount 2: SkyCount 3: ToyCount 4: FogCount

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
  vec3 origin;
  vec3 ray;
  ushort2 index;
  uint32_t state;
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

struct DevicePointers {
  TraceTask* light_trace;
  TraceTask* bounce_trace;
  uint16_t* light_trace_count;
  uint16_t* bounce_trace_count;
  TraceResult* trace_results;
  uint16_t* task_counts;
  uint16_t* task_offsets;
  uint32_t* light_sample_history;
  RGBF* frame_final;
  RGBF* frame_output;
  RGBF* frame_temporal;
  RGBF* frame_buffer;
  RGBF* frame_variance;
  RGBF* frame_bias_cache;
  RGBF* albedo_buffer;
  RGBF* light_records;
  RGBF* bounce_records;
  XRGB8* buffer_8bit;
  vec3* raydir_buffer;
  TraceResult* trace_result_buffer;
  TraceResult* trace_result_temporal;
  uint8_t* state_buffer;
  curandStateXORWOW_t* randoms;
  cudaTextureObject_t* albedo_atlas;
  cudaTextureObject_t* illuminance_atlas;
  cudaTextureObject_t* material_atlas;
  RestirSample* restir_samples;
  RestirEvalData* restir_eval_data;
} typedef DevicePointers;

//===========================================================================================
// Bit Masks
//===========================================================================================

#define RANDOM_INDEX 0x0000ffffu
#define DEPTH_LEFT 0xffff0000u
#define SKY_HIT 0xffffffffu
#define OCEAN_HIT 0xfffffffeu
#define TOY_HIT 0xfffffffdu
#define FOG_HIT 0xfffffffcu
#define DEBUG_LIGHT_HIT 0xfffffff0u
#define TRIANGLE_ID_LIMIT 0xefffffffu
#define LIGHT_ID_ANY 0xfffffff0u
#define LIGHT_ID_NONE 0xfffffff1u
#define TYPE_CAMERA 0x0u
#define TYPE_LIGHT 0x1u
#define TYPE_BOUNCE 0x2u
#define STATE_ALBEDO 0b1u
#define STATE_LIGHT_OCCUPIED 0b10u
#define STATE_BOUNCE_OCCUPIED 0b100u

//===========================================================================================
// Device Variables
//===========================================================================================

__constant__ DevicePointers device;

__constant__ int device_max_ray_depth;

__constant__ Scene device_scene;

__constant__ int device_pixels_per_thread;

__constant__ int device_iteration_type;

__constant__ TraceTask* device_trace_tasks;

__constant__ uint16_t* device_trace_count;

__constant__ RGBF* device_records;

__constant__ int device_temporal_frames;

__constant__ int device_denoiser;

__constant__ uint32_t device_reservoir_size;

__constant__ int device_width;

__constant__ int device_height;

__constant__ int device_amount;

__constant__ float device_step;

__constant__ float device_vfov;

__constant__ Quaternion device_camera_rotation;

__constant__ TextureAssignment* device_texture_assignments;

__constant__ vec3 device_sun;

__constant__ int device_shading_mode;

__constant__ RGBF* device_bloom_scratch;

__constant__ Jitter device_jitter;

__device__ Mat4x4 device_view_space;

__device__ Mat4x4 device_projection;

__constant__ int device_accum_mode;

//===========================================================================================
// Functions
//===========================================================================================

__device__ static uint32_t get_pixel_id(const int x, const int y) {
  return x + device_width * y;
}

__device__ static int get_task_address_of_thread(const int thread_id, const int block_id, const int number) {
  const int warp_id       = (((thread_id & 0x60) >> 5) + block_id * (THREADS_PER_BLOCK / 32));
  const int thread_offset = (thread_id & 0x1f);
  return 32 * device_pixels_per_thread * warp_id + 32 * number + thread_offset;
}

__device__ static int get_task_address(const int number) {
  return get_task_address_of_thread(threadIdx.x, blockIdx.x, number);
}

__device__ static int is_first_ray() {
  return (device_iteration_type == TYPE_CAMERA);
}

__device__ static int proper_light_sample(const uint32_t target_light, const uint32_t source_light) {
  return (device_iteration_type == TYPE_CAMERA || target_light == source_light || target_light == LIGHT_ID_ANY);
}

#endif /* CU_UTILS_H */
