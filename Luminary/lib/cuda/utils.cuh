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

struct Sample {
  vec3 origin;
  vec3 ray;
  ushort2 state; //x = (high 8bits) 1st bit (active?) 2nd bit (albedo buffer written?) 3rd bit (is light sample?) other 5bits give sample offset | (low 8 bits) depth, y = iterations left
  int random_index;
  RGBF record;
  RGBF result;
  RGBF albedo_buffer;
  ushort2 index;
  float depth;
  unsigned int hit_id;
} typedef Sample;

struct Sample_Result {
  RGBF result;
  RGBF albedo_buffer;
} typedef Sample_Result;

//===========================================================================================
// Device Variables
//===========================================================================================

__constant__
int device_reflection_depth;

__constant__
Scene device_scene;

__constant__
unsigned int device_samples_length;

__constant__
Sample* device_active_samples;

__constant__
Sample_Result* device_finished_samples;

__constant__
int device_iterations_per_sample;

__constant__
int device_samples_per_sample;

__device__
int device_sample_offset;

__constant__
curandStateXORWOW_t* device_sample_randoms;

__constant__
int device_temporal_frames;

__constant__
int device_diffuse_samples;

__constant__
RGBF* device_frame;

__constant__
RGBF* device_denoiser;

__constant__
RGBF* device_albedo_buffer;

__constant__
unsigned int device_width;

__constant__
unsigned int device_height;

__constant__
unsigned int device_amount;

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


#endif /* CU_UTILS_H */
