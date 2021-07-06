#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include "utils.h"

#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 512

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        fprintf( stderr, "Optix returned error %d in call (%s) (line %d)\n", res, #call, __LINE__ ); \
        system("pause");                                                \
        exit(-1);                                                       \
      }                                                                 \
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

// state is first bit albedo buffer written, then 15 bits the depth and the last 16 bits the random_index

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
#define DEPTH_LEFT 0x7fff0000
#define ALBEDO_BUFFER_STATE 0x80000000
#define SKY_HIT 0xffffffff
#define OCEAN_HIT 0xfffffffe

//===========================================================================================
// Device Variables
//===========================================================================================

__constant__
int device_max_ray_depth;

__constant__
Scene device_scene;

__constant__
int device_pixels_per_thread;

__device__
int device_pixels_left;

__constant__
GeometryTask* device_geometry_tasks;

__constant__
SkyTask* device_sky_tasks;

__constant__
OceanTask* device_ocean_tasks;

__constant__
TraceTask* device_trace_tasks;

__constant__
TraceResult* device_trace_results;

// 0: TraceCount 1: GeoCount 2: OceanCount 3: SkyCount
__constant__
uint16_t* device_task_counts;

__constant__
curandStateXORWOW_t* device_sample_randoms;

__constant__
int device_temporal_frames;

__constant__
RGBF* device_frame_buffer;

__constant__
RGBF* device_frame_output;

__constant__
RGBF* device_records;

__constant__
RGBF* device_denoiser;

__constant__
RGBF* device_albedo_buffer;

__constant__
RGB8* device_frame_8bit;

__constant__
int device_width;

__constant__
int device_height;

__constant__
int device_amount;

__constant__
float device_step;

__constant__
float device_vfov;

__constant__
float device_offset_x;

__constant__
float device_offset_y;

__constant__
Quaternion device_camera_rotation;

__constant__
cudaTextureObject_t* device_albedo_atlas;

__constant__
cudaTextureObject_t* device_illuminance_atlas;

__constant__
cudaTextureObject_t* device_material_atlas;

__constant__
texture_assignment* device_texture_assignments;

__constant__
vec3 device_sun;

__constant__
RGBF device_default_material;

__constant__
int device_shading_mode;

__constant__
RGBF* device_bloom_scratch;

//===========================================================================================
// Functions
//===========================================================================================

__device__
int get_task_address(const int number) {
  const int warp_id = (((threadIdx.x & 0x60) >> 5) + blockIdx.x * (THREADS_PER_BLOCK / 32));
  const int thread_id = (threadIdx.x & 0x1f);
  return 32 * device_pixels_per_thread * warp_id + 32 * number + thread_id;
}


#endif /* CU_UTILS_H */
