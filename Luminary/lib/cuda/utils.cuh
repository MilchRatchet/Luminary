#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include "utils.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
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

//===========================================================================================
// Device Variables
//===========================================================================================

__constant__
int device_reflection_depth;

__constant__
Scene device_scene;

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

__device__
curandStateXORWOW_t device_random;

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


#endif /* CU_UTILS_H */
