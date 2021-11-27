#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>

#if defined(__MSVC__) || defined(__CUDACC__) || defined(__clang__)
#include <intrin.h>
#endif

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
#define SHADING_HEAT 4
#define SHADING_WIREFRAME 5
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
#endif

#ifndef LUMINARY_FILTERS
#define LUMINARY_FILTERS
#define FILTER_NONE 0
#define FILTER_GRAY 1
#define FILTER_SEPIA 2
#define FILTER_GAMEBOY 3
#define FILTER_2BITGRAY 4
#define FILTER_CRT 5
#define FILTER_BLACKWHITE 6
#endif

#ifndef LUMINARY_SNAP_RESOLUTION
#define LUMINARY_SNAP_RESOLUTION
#define SNAP_RESOLUTION_WINDOW 0
#define SNAP_RESOLUTION_RENDER 1
#endif

#ifndef LUMINARY_ACCUM_MODES
#define LUMINARY_ACCUM_MODES
#define NO_ACCUMULATION 0
#define TEMPORAL_ACCUMULATION 1
#define TEMPORAL_REPROJECTION 2
#endif

struct General {
  int width;
  int height;
  int samples;
  int max_ray_depth;
  int denoiser;
  char** mesh_files;
  int mesh_files_count;
  int mesh_files_length;
  char* output_path;
} typedef General;

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
  int filter;
  int bloom;
  float bloom_strength;
  int dithering;
  float wasd_speed;
  float mouse_speed;
  int smooth_movement;
  float smoothing_factor;
  float temporal_blend_factor;
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
  RGBF sun_color;
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

struct Fog {
  int active;
  float scattering;
  float absorption;
  float anisotropy;
  float height;
  float dist;
  float falloff;
} typedef Fog;

struct Jitter {
  float x;
  float y;
  float prev_x;
  float prev_y;
} typedef Jitter;

struct Scene {
  Camera camera;
  Triangle* triangles;
  TraversalTriangle* traversal_triangles;
  unsigned int triangles_length;
  Node8* nodes;
  unsigned int nodes_length;
  uint16_t materials_length;
  TextureAssignment* texture_assignments;
  Light* lights;
  unsigned int lights_length;
  Ocean ocean;
  Sky sky;
  Toy toy;
  Fog fog;
} typedef Scene;

struct DeviceBuffer {
  void* device_pointer;
  size_t size;
  int allocated;
} typedef DeviceBuffer;

struct RaytraceInstance {
  unsigned int width;
  unsigned int height;
  RGBF* frame_final_device;
  DeviceBuffer* light_trace;
  DeviceBuffer* bounce_trace;
  DeviceBuffer* light_trace_count;
  DeviceBuffer* bounce_trace_count;
  DeviceBuffer* trace_results;
  DeviceBuffer* task_counts;
  DeviceBuffer* task_offsets;
  DeviceBuffer* light_sample_history;
  DeviceBuffer* frame_output;
  DeviceBuffer* frame_temporal;
  DeviceBuffer* frame_buffer;
  DeviceBuffer* frame_variance;
  DeviceBuffer* frame_bias_cache;
  DeviceBuffer* albedo_buffer;
  DeviceBuffer* light_records;
  DeviceBuffer* bounce_records;
  DeviceBuffer* buffer_8bit;
  DeviceBuffer* albedo_atlas;
  int albedo_atlas_length;
  DeviceBuffer* illuminance_atlas;
  int illuminance_atlas_length;
  DeviceBuffer* material_atlas;
  int material_atlas_length;
  int max_ray_depth;
  int offline_samples;
  Scene scene_gpu;
  int denoiser;
  int use_denoiser;
  int temporal_frames;
  int lights_active;
  DeviceBuffer* randoms;
  RGBF default_material;
  int shading_mode;
  RGBF** bloom_mips_gpu;
  int snap_resolution;
  General settings;
  void* denoise_setup;
  Jitter jitter;
  int accum_mode;
  DeviceBuffer* raydir_buffer;
  DeviceBuffer* trace_result_buffer;
  DeviceBuffer* trace_result_temporal;
  DeviceBuffer* state_buffer;
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

#if defined(__GNUC__)
#warning The function bsr may not work correctly on GCC as it was never tested.
#define bsr(input, output) \
  { output = 32 - __builtin_clz(input | 1) }
#elif defined(__MSVC__)
#define bsr(input, output) _BitScanReverse((DWORD*) &output, (DWORD) input | 1);
#elif defined(__clang__) || defined(__CUDACC__)
#define bsr(input, output) _BitScanReverse((unsigned long*) &output, (unsigned long) input | 1);
#else
#error No implementation of bsr is available for the given compiler. Consider adding an implementation.
#endif

#endif /* UTILS_H */
