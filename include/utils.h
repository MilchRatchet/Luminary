#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime_api.h>
#include <stdlib.h>

#if defined(__GNUC__) || defined(__clang__)
#elif defined(__MSVC__) || defined(__CUDACC__)
#include <intrin.h>
#endif

#include "bvh.h"
#include "log.h"
#include "structs.h"

#define gpuErrchk(ans)                                         \
  {                                                            \
    if (ans != cudaSuccess) {                                  \
      crash_message("GPUassert: %s", cudaGetErrorString(ans)); \
    }                                                          \
  }

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

enum RayIterationType { TYPE_CAMERA = 0, TYPE_LIGHT = 1, TYPE_BOUNCE = 2 } typedef RayIterationType;

#define TEXTURE_NONE ((uint16_t) 0xffffu)

enum OutputImageFormat { IMGFORMAT_PNG = 0, IMGFORMAT_QOI = 1 } typedef OutputImageFormat;

enum ShadingMode {
  SHADING_DEFAULT   = 0,
  SHADING_ALBEDO    = 1,
  SHADING_DEPTH     = 2,
  SHADING_NORMAL    = 3,
  SHADING_HEAT      = 4,
  SHADING_WIREFRAME = 5,
  SHADING_LIGHTS    = 6
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

enum DenoisingMode { DENOISING_OFF = 0, DENOISING_ON = 1, DENOISING_UPSCALING = 2 } typedef DenoisingMode;

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
  DenoisingMode denoiser;
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
  float russian_roulette_bias;
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
  int flashlight_mode;
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
  float forward_scattering;
  float backward_scattering;
  float lobe_lerp;
  float wetness;
  float powder;
  int steps;
  int shadow_steps;
  float noise_shape_scale;
  float noise_detail_scale;
  float noise_weather_scale;
  float noise_curl_scale;
  float coverage;
  float coverage_min;
  float anvil;
  float mipmap_bias;
} typedef Cloud;

// Settings that affect the sky LUTs
struct AtmoSettings {
  float rayleigh_density;
  float mie_density;
  float ozone_density;
  float rayleigh_falloff;
  float mie_falloff;
  float mie_g;
  float ground_visibility;
  float ozone_layer_thickness;
  float base_density;
  int ozone_absorption;
  float multiscattering_factor;
} typedef AtmoSettings;

struct Sky {
  vec3 geometry_offset;
  float azimuth;
  float altitude;
  float moon_azimuth;
  float moon_altitude;
  float moon_albedo;
  float sun_strength;
  float base_density;
  int ozone_absorption;
  int steps;
  Star* stars;
  int* stars_offsets;
  int settings_stars_count;
  int current_stars_count;
  int stars_seed;
  float stars_intensity;
  Cloud cloud;
  float rayleigh_density;
  float mie_density;
  float ozone_density;
  float rayleigh_falloff;
  float mie_falloff;
  float mie_g;
  float ground_visibility;
  float ozone_layer_thickness;
  float multiscattering_factor;
  int lut_initialized;
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
  float anisotropy;
  RGBF scattering;
  RGBF absorption;
  float pollution;
  float absorption_strength;
} typedef Ocean;

struct Fog {
  int active;
  float density;
  float anisotropy;
  float height;
  float dist;
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
  int lights_active;
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

struct RayEmitter {
  Jitter jitter;
  Mat4x4 view_space;
  Mat4x4 projection;
  float step;
  float vfov;
  Quaternion camera_rotation;
} typedef RayEmitter;

struct TextureAtlas {
  DeviceBuffer* albedo;
  int albedo_length;
  DeviceBuffer* illuminance;
  int illuminance_length;
  DeviceBuffer* material;
  int material_length;
  DeviceBuffer* normal;
  int normal_length;
} typedef TextureAtlas;

struct RaytraceInstance {
  unsigned int width;
  unsigned int height;
  unsigned int output_width;
  unsigned int output_height;
  int realtime;
  RGBAhalf* frame_final_device;
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
  DeviceBuffer* albedo_buffer;
  DeviceBuffer* normal_buffer;
  DeviceBuffer* light_records;
  DeviceBuffer* bounce_records;
  DeviceBuffer* buffer_8bit;
  DeviceBuffer* light_samples_1;
  DeviceBuffer* light_samples_2;
  DeviceBuffer* light_eval_data;
  DeviceBuffer* cloud_noise;
  DeviceBuffer* sky_ms_luts;
  DeviceBuffer* sky_tm_luts;
  int max_ray_depth;
  int reservoir_size;
  int offline_samples;
  int light_resampling;
  Scene scene;
  DenoisingMode denoiser;
  int temporal_frames;
  int spatial_samples;
  int spatial_iterations;
  DeviceBuffer* randoms;
  int shading_mode;
  RGBAhalf** bloom_mips_gpu;
  int snap_resolution;
  OutputImageFormat image_format;
  int post_process_menu;
  General settings;
  AtmoSettings atmo_settings;
  void* denoise_setup;
  Jitter jitter;
  int accum_mode;
  RayEmitter emitter;
  DeviceBuffer* raydir_buffer;
  DeviceBuffer* trace_result_buffer;
  DeviceBuffer* state_buffer;
  TextureAtlas tex_atlas;
} typedef RaytraceInstance;

struct DevicePointers {
  TraceTask* light_trace;
  TraceTask* bounce_trace;
  uint16_t* light_trace_count;
  uint16_t* bounce_trace_count;
  TraceResult* trace_results;
  uint16_t* task_counts;
  uint16_t* task_offsets;
  uint32_t* light_sample_history;
  RGBAhalf* frame_output;
  RGBAhalf* frame_temporal;
  RGBAhalf* frame_buffer;
  RGBAhalf* frame_variance;
  RGBAhalf* albedo_buffer;
  RGBAhalf* normal_buffer;
  RGBAhalf* light_records;
  RGBAhalf* bounce_records;
  XRGB8* buffer_8bit;
  vec3* raydir_buffer;
  TraceResult* trace_result_buffer;
  uint8_t* state_buffer;
  uint32_t* randoms;
  cudaTextureObject_t* albedo_atlas;
  cudaTextureObject_t* illuminance_atlas;
  cudaTextureObject_t* material_atlas;
  cudaTextureObject_t* normal_atlas;
  cudaTextureObject_t* cloud_noise;
  cudaTextureObject_t* sky_ms_luts;
  cudaTextureObject_t* sky_tm_luts;
  LightSample* light_samples;
  LightEvalData* light_eval_data;
} typedef DevicePointers;

struct DeviceConstantMemory {
  DevicePointers ptrs;
  Scene scene;
  int max_ray_depth;
  int pixels_per_thread;
  int iteration_type;
  TraceTask* trace_tasks;
  uint16_t* trace_count;
  RGBAhalf* records;
  int temporal_frames;
  int denoiser;
  uint32_t reservoir_size;
  int spatial_samples;
  int light_resampling;
  int width;
  int height;
  int output_width;
  int output_height;
  TextureAssignment* texture_assignments;
  vec3 sun_pos;
  vec3 moon_pos;
  int shading_mode;
  RGBF* bloom_scratch;
  RayEmitter emitter;
  int accum_mode;
} typedef DeviceConstantMemory;

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
