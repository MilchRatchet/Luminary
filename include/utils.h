#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>

#if defined(__GNUC__) || defined(__clang__)
#elif defined(__MSVC__) || defined(__CUDACC__)
#include <intrin.h>
#endif

#include "bvh.h"
#include "log.h"
#include "mesh.h"
#include "primitives.h"
#include "texture.h"

// Flags variables as unused so that no warning is emitted
#define LUM_UNUSED(x) (void) (x);

#ifndef PI
#define PI 3.141592653589f
#endif

#ifndef ONE_OVER_PI
#define ONE_OVER_PI 0.31830988618f
#endif

#define LIGHT_ID_SUN 0xffffffffu
#define LIGHT_ID_TOY 0xfffffffeu
#define LIGHT_ID_NONE 0xfffffff1u

enum OutputImageFormat { IMGFORMAT_PNG = 0, IMGFORMAT_QOI = 1 } typedef OutputImageFormat;

enum ShadingMode {
  SHADING_DEFAULT   = 0,
  SHADING_ALBEDO    = 1,
  SHADING_DEPTH     = 2,
  SHADING_NORMAL    = 3,
  SHADING_HEAT      = 4,
  SHADING_WIREFRAME = 5
} typedef ShadingMode;

enum ToyShape { TOY_SPHERE = 0 } typedef ToyShape;

enum ToneMap { TONEMAP_NONE = 0, TONEMAP_ACES = 1, TONEMAP_REINHARD = 2, TONEMAP_UNCHARTED2 = 3 } typedef ToneMap;

enum Filter {
  FILTER_NONE       = 0,
  FILTER_GRAY       = 1,
  FILTER_SEPIA      = 2,
  FILTER_GAMEBOY    = 3,
  FILTER_2BITGRAY   = 4,
  FILTER_CRT        = 5,
  FILTER_BLACKWHITE = 6
} typedef Filter;

enum SnapResolution { SNAP_RESOLUTION_WINDOW = 0, SNAP_RESOLUTION_RENDER = 1 } typedef SnapResolution;

enum AccumMode { NO_ACCUMULATION = 0, TEMPORAL_ACCUMULATION = 1, TEMPORAL_REPROJECTION = 2 } typedef AccumMode;

enum MaterialFresnel { SCHLICK = 0, FDEZ_AGUERA = 1 } typedef MaterialFresnel;

enum MaterialDiffuse { LAMBERTIAN = 0, FROSTBITEDISNEY = 1 } typedef MaterialDiffuse;

struct DeviceBuffer {
  void* device_pointer;
  size_t size;
  int allocated;
} typedef DeviceBuffer;

struct General {
  int width;
  int height;
  int samples;
  int max_ray_depth;
  int reservoir_size;
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
  float far_clip_distance;
  int tonemap;
  int filter;
  int bloom;
  float bloom_strength;
  float bloom_threshold;
  int dithering;
  int purkinje;
  float purkinje_kappa1;
  float purkinje_kappa2;
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

struct Star {
  float altitude;
  float azimuth;
  float radius;
  float intensity;
} typedef Star;

struct Cloud {
  int active;
  int initialized;
  float offset_x;
  float offset_z;
  float height_max;
  float height_min;
  float density;
  int seed;
  uint8_t* shape_noise;
  uint8_t* detail_noise;
  uint8_t* weather_map;
  uint8_t* curl_noise;
  float forward_scattering;
  float backward_scattering;
  float lobe_lerp;
  float wetness;
  float powder;
  int shadow_steps;
  float noise_shape_scale;
  float noise_detail_scale;
  float noise_weather_scale;
  float noise_curl_scale;
  float coverage;
  float coverage_min;
  float anvil;
} typedef Cloud;

struct Sky {
  vec3 geometry_offset;
  RGBF sun_color;
  float azimuth;
  float altitude;
  float moon_azimuth;
  float moon_altitude;
  float moon_albedo;
  float sun_strength;
  float base_density;
  int ozone_absorption;
  int steps;
  int shadow_steps;
  Star* stars;
  int* stars_offsets;
  int settings_stars_count;
  int current_stars_count;
  int stars_seed;
  float stars_intensity;
  Cloud cloud;
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

struct GlobalMaterial {
  RGBF default_material;
  MaterialFresnel fresnel;
  MaterialDiffuse diffuse;
  int lights_active;
  int bvh_alpha_cutoff;
  float alpha_cutoff;
} typedef GlobalMaterial;

struct Scene {
  Camera camera;
  Triangle* triangles;
  TraversalTriangle* traversal_triangles;
  TriangleLight* triangle_lights;
  unsigned int triangles_length;
  unsigned int triangle_lights_length;
  Node8* nodes;
  unsigned int nodes_length;
  uint16_t materials_length;
  TextureAssignment* texture_assignments;
  Ocean ocean;
  Sky sky;
  Toy toy;
  Fog fog;
  GlobalMaterial material;
} typedef Scene;

struct RaytraceInstance {
  unsigned int width;
  unsigned int height;
  int realtime;
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
  DeviceBuffer* light_samples_1;
  DeviceBuffer* light_samples_2;
  DeviceBuffer* light_eval_data;
  int albedo_atlas_length;
  DeviceBuffer* illuminance_atlas;
  int illuminance_atlas_length;
  DeviceBuffer* material_atlas;
  int material_atlas_length;
  int max_ray_depth;
  int reservoir_size;
  int offline_samples;
  Scene scene_gpu;
  int denoiser;
  int temporal_frames;
  int spatial_samples;
  int spatial_iterations;
  DeviceBuffer* randoms;
  int shading_mode;
  RGBF** bloom_mips_gpu;
  int snap_resolution;
  OutputImageFormat image_format;
  General settings;
  void* denoise_setup;
  Jitter jitter;
  int accum_mode;
  DeviceBuffer* raydir_buffer;
  DeviceBuffer* trace_result_buffer;
  DeviceBuffer* trace_result_temporal;
  DeviceBuffer* state_buffer;
} typedef RaytraceInstance;

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif /* min */

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif /* max */

#undef assert
#define assert(ans, message, _abort)  \
  {                                   \
    if (!(ans)) {                     \
      if (_abort) {                   \
        crash_message("%s", message); \
      }                               \
      else {                          \
        error_message("%s", message); \
      }                               \
    }                                 \
  }

static inline void* ___s_realloc(void* ptr, const size_t size) {
  if (size == 0)
    return (void*) 0;
  void* new_ptr = realloc(ptr, size);
  assert((unsigned long long) new_ptr, "Reallocation failed!", 1);
  return new_ptr;
}

#define safe_realloc(ptr, size) ___s_realloc((ptr), (size))

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

#if defined(__GNUC__) || defined(__clang__)
#define bsr(input, output) \
  { output = 32 - __builtin_clz(input | 1); }
#elif defined(__MSVC__)
#define bsr(input, output) _BitScanReverse((DWORD*) &output, (DWORD) input | 1);
#elif defined(__CUDACC__)
#define bsr(input, output) _BitScanReverse((unsigned long*) &output, (unsigned long) input | 1);
#else
#error No implementation of bsr is available for the given compiler. Consider adding an implementation.
#endif

#endif /* UTILS_H */
