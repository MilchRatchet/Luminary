#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

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

__device__
int device_reflection_depth;

__device__
Scene device_scene;

__device__
int device_diffuse_samples;

__device__
RGBF* device_frame;

__device__
RGBF* device_denoiser;

__device__
RGBF* device_albedo_buffer;

__device__
Quaternion device_camera_space;

__device__
unsigned int device_width;

__device__
unsigned int device_height;

__device__
unsigned int device_amount;

__device__
float device_step;

__device__
float device_vfov;

__device__
float device_offset_x;

__device__
float device_offset_y;

__device__
Quaternion device_camera_rotation;

__device__
curandStateXORWOW_t device_random;

__device__
cudaTextureObject_t* device_albedo_atlas;

__device__
cudaTextureObject_t* device_illuminance_atlas;

__device__
cudaTextureObject_t* device_material_atlas;

__device__
texture_assignment* device_texture_assignments;

__device__
vec3 device_sun;


#endif /* CU_UTILS_H */
