#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
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
  int auto_exposure;
  float alpha_cutoff;
} typedef Camera;

struct Light {
  vec3 pos;
  float radius;
} typedef Light;

struct Sky {
  float base_density;
  float rayleigh_falloff;
  float mie_falloff;
} typedef Sky;

struct Ocean {
  int active;
  int emissive;
  float height;
  float amplitude;
  float frequency;
  float choppyness;
  float speed;
  float time;
  RGBAF albedo;
} typedef Ocean;

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
  Ocean ocean;
  Sky sky;
} typedef Scene;

struct raytrace_instance {
  unsigned int width;
  unsigned int height;
  void* geometry_tasks_gpu;
  void* sky_tasks_gpu;
  void* ocean_tasks_gpu;
  void* trace_tasks_gpu;
  void* trace_results_gpu;
  void* task_counts_gpu;
  RGBF* frame_output;
  RGBF* frame_output_gpu;
  RGBF* frame_buffer_gpu;
  RGBF* frame_variance_gpu;
  RGBF* albedo_buffer_gpu;
  RGBF* records_gpu;
  RGB8* buffer_8bit_gpu;
  void* albedo_atlas;
  int albedo_atlas_length;
  void* illuminance_atlas;
  int illuminance_atlas_length;
  void* material_atlas;
  int material_atlas_length;
  int max_ray_depth;
  Scene scene_gpu;
  int denoiser;
  void* randoms_gpu;
  RGBF default_material;
  int shading_mode;
  RGBF* bloom_scratch_gpu;
} typedef raytrace_instance;

#define clamp(value, low, high) \
  { (value) = min((high), max((value), (low))); }

#endif /* UTILS_H */
