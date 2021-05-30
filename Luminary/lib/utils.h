#ifndef UTILS_H
#define UTILS_H

#include "primitives.h"
#include "mesh.h"
#include "bvh.h"
#include "texture.h"

#ifndef PI
#define PI 3.141592653589f
#endif

#ifndef ONE_OVER_PI
#define ONE_OVER_PI 0.31830988618f
#endif

#ifndef LUMINARY_SHADING_MODES
#define LUMINARY_SHADING_MODES
#define SHADING_DEFAULT 0
#define SHADING_ALBEDO 1
#define SHADING_DEPTH 2
#define SHADING_NORMAL 3
#endif

struct Camera {
  vec3 pos;
  vec3 rotation;
  float fov;
  float focal_length;
  float aperture_size;
  float exposure;
} typedef Camera;

struct Light {
  vec3 pos;
  float radius;
} typedef Light;

struct Scene {
  Camera camera;
  float far_clip_distance;
  Triangle* triangles;
  Traversal_Triangle* traversal_triangles;
  unsigned int triangles_length;
  Node8* nodes;
  unsigned int nodes_length;
  uint16_t materials_length;
  texture_assignment* texture_assignments;
  float azimuth;
  float altitude;
  float sun_strength;
  Light* lights;
  unsigned int lights_length;
} typedef Scene;

struct raytrace_instance {
  unsigned int width;
  unsigned int height;
  RGBF* frame_buffer;
  RGBF* frame_buffer_gpu;
  RGB8* buffer_8bit_gpu;
  void* albedo_atlas;
  int albedo_atlas_length;
  void* illuminance_atlas;
  int illuminance_atlas_length;
  void* material_atlas;
  int material_atlas_length;
  int reflection_depth;
  int diffuse_samples;
  Scene scene_gpu;
  int denoiser;
  void* samples_gpu;
  void* samples_finished_gpu;
  int samples_per_sample;
  void* randoms_gpu;
  RGBF* albedo_buffer_gpu;
  int shading_mode;
} typedef raytrace_instance;

#endif /* UTILS_H */
