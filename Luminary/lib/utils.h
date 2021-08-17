#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>

#include "bvh.h"
#include "mesh.h"
#include "primitives.h"
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

#ifndef LUMINARY_TOY_SHAPES
#define LUMINARY_TOY_SHAPES
#define TOY_SPHERE 0
#endif

#ifndef LUMINARY_TONEMAPS
#define LUMINARY_TONEMAPS
#define TONEMAP_NONE 0
#define TONEMAP_ACES 1
#define TONEMAP_REINHARD 2
#define TONEMAP_UNCHARTED2 3
#define TONEMAP_CUSTOM 4
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
  float far_clip_distance;
  int tonemap;
} typedef Camera;

struct Light {
  vec3 pos;
  float radius;
} typedef Light;

struct Toy {
  int active;
  int shape;
  int emissive;
  vec3 position;
  vec3 rotation;
  float scale;
  float refractive_index;
  RGBAF albedo;
  RGBAF material;
  RGBAF emission;
} typedef Toy;

struct Sky {
  float azimuth;
  float altitude;
  float sun_strength;
  float base_density;
  float rayleigh_falloff;
  float mie_falloff;
} typedef Sky;

struct Ocean {
  int active;
  int emissive;
  int update;
  float height;
  float amplitude;
  float frequency;
  float choppyness;
  float speed;
  float time;
  RGBAF albedo;
  float refractive_index;
} typedef Ocean;

struct Scene {
  Camera camera;
  Triangle* triangles;
  Traversal_Triangle* traversal_triangles;
  unsigned int triangles_length;
  Node8* nodes;
  unsigned int nodes_length;
  uint16_t materials_length;
  texture_assignment* texture_assignments;
  Light* lights;
  unsigned int lights_length;
  Ocean ocean;
  Sky sky;
  Toy toy;
} typedef Scene;

struct RaytraceInstance {
  unsigned int width;
  unsigned int height;
  void* tasks_gpu;
  void* trace_results_gpu;
  void* task_counts_gpu;
  void* task_offsets_gpu;
  uint32_t* light_sample_history_gpu;
  RGBF* frame_output_gpu;
  RGBF* frame_buffer_gpu;
  RGBF* frame_variance_gpu;
  RGBF* frame_bias_cache_gpu;
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
  int offline_samples;
  Scene scene_gpu;
  int denoiser;
  int use_denoiser;
  int use_bloom;
  int temporal_frames;
  int lights_active;
  void* randoms_gpu;
  RGBF default_material;
  int shading_mode;
  RGBF* bloom_scratch_gpu;
} typedef RaytraceInstance;

#define clamp(value, low, high) \
  { (value) = min((high), max((value), (low))); }

#define ensure_capacity(ptr, count, length, size) \
  {                                               \
    if (count == length) {                        \
      length += 1;                                \
      length *= 2;                                \
      ptr = safe_realloc(ptr, size * length);     \
    }                                             \
  }

#endif /* UTILS_H */
