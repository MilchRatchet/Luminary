#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>
#include <stdlib.h>

#if defined(__GNUC__) || defined(__clang__)
#elif defined(__MSVC__) || defined(__CUDACC__)
#include <intrin.h>
#endif

#include "log.h"
#include "structs.h"

#define gpuErrchk(ans)                                                                      \
  {                                                                                         \
    if (ans != cudaSuccess) {                                                               \
      crash_message("CUDA Error: %s (%s)", cudaGetErrorName(ans), cudaGetErrorString(ans)); \
    }                                                                                       \
  }

#define OPTIX_CHECK(call)                                                                                \
  {                                                                                                      \
    OptixResult res = call;                                                                              \
                                                                                                         \
    if (res != OPTIX_SUCCESS) {                                                                          \
      crash_message("Optix returned error \"%s\"(%d) in call (%s)", optixGetErrorName(res), res, #call); \
    }                                                                                                    \
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

#define OPTIXRT_NUM_GROUPS 3

enum RayIterationType { TYPE_CAMERA = 0, TYPE_LIGHT = 1, TYPE_BOUNCE = 2 } typedef RayIterationType;

#define TEXTURE_NONE ((uint16_t) 0xffffu)

enum OutputImageFormat { IMGFORMAT_PNG = 0, IMGFORMAT_QOI = 1 } typedef OutputImageFormat;

enum ShadingMode {
  SHADING_DEFAULT        = 0,
  SHADING_ALBEDO         = 1,
  SHADING_DEPTH          = 2,
  SHADING_NORMAL         = 3,
  SHADING_HEAT           = 4,
  SHADING_IDENTIFICATION = 5,
  SHADING_LIGHTS         = 6
} typedef ShadingMode;

enum ToyShape { TOY_SPHERE = 0, TOY_PLANE = 1 } typedef ToyShape;

enum ToneMap { TONEMAP_NONE = 0, TONEMAP_ACES = 1, TONEMAP_REINHARD = 2, TONEMAP_UNCHARTED2 = 3 } typedef ToneMap;

enum BVHType { BVH_LUMINARY = 0, BVH_OPTIX = 1 } typedef BVHType;

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

// Set of architectures supported by Luminary
enum DeviceArch {
  DEVICE_ARCH_UNKNOWN = 0,
  DEVICE_ARCH_PASCAL  = 1,
  DEVICE_ARCH_VOLTA   = 11,
  DEVICE_ARCH_TURING  = 2,
  DEVICE_ARCH_AMPERE  = 3,
  DEVICE_ARCH_ADA     = 4,
  DEVICE_ARCH_HOPPER  = 41
} typedef DeviceArch;

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
  float bloom_blend;
  int lens_flare;
  float lens_flare_threshold;
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

struct Toy {
  int active;
  ToyShape shape;
  int emissive;
  vec3 position;
  vec3 rotation;
  float scale;
  float refractive_index;
  RGBAF albedo;
  RGBAF material;
  RGBAF emission;
  int flashlight_mode;
  Quaternion computed_rotation;
} typedef Toy;

struct Star {
  float altitude;
  float azimuth;
  float radius;
  float intensity;
} typedef Star;

struct CloudLayer {
  int active;
  float height_max;
  float height_min;
  float coverage;
  float coverage_min;
  float type;
  float type_min;
  float wind_speed;
  float wind_angle;
} typedef CloudLayer;

struct Cloud {
  int active;
  int initialized;
  int atmosphere_scattering;
  CloudLayer low;
  CloudLayer mid;
  CloudLayer top;
  float offset_x;
  float offset_z;
  float density;
  int seed;
  float forward_scattering;
  float backward_scattering;
  float lobe_lerp;
  int steps;
  int shadow_steps;
  float noise_shape_scale;
  float noise_detail_scale;
  float noise_weather_scale;
  float mipmap_bias;
  int octaves;
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
  int hdri_initialized;
  int hdri_dim;
  int settings_hdri_dim;
  int hdri_active;
  int hdri_samples;
  vec3 hdri_origin;
  float hdri_mip_bias;
  int aerial_perspective;
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
  int colored_transparency;
} typedef GlobalMaterial;

struct Scene {
  Camera camera;
  Triangle* triangles;
  TriangleLight* triangle_lights;
  TriangleGeomData triangle_data;
  unsigned int triangle_lights_count;
  uint16_t materials_count;
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

struct ReSTIRSettings {
  int use_temporal_resampling;
  int use_spatial_resampling;
  int spatial_sample_count;
  int initial_reservoir_size;
} typedef ReSTIRSettings;

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
  RGBF* light_records;
  RGBF* bounce_records;
  XRGB8* buffer_8bit;
  vec3* raydir_buffer;
  TraceResult* trace_result_buffer;
  uint8_t* state_buffer;
  uint32_t* randoms;
  DeviceTexture* albedo_atlas;
  DeviceTexture* illuminance_atlas;
  DeviceTexture* material_atlas;
  DeviceTexture* normal_atlas;
  DeviceTexture* cloud_noise;
  DeviceTexture* sky_ms_luts;
  DeviceTexture* sky_tm_luts;
  DeviceTexture* sky_hdri_luts;
  LightSample* light_samples;
  LightEvalData* light_eval_data;
} typedef DevicePointers;

struct DeviceConstantMemory {
  DevicePointers ptrs;
  Scene scene;
  ReSTIRSettings restir;
  int max_ray_depth;
  int pixels_per_thread;
  int iteration_type;
  int depth;
  TraceTask* trace_tasks;
  uint16_t* trace_count;
  RGBF* records;
  int temporal_frames;
  int denoiser;
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
  OptixTraversableHandle optix_bvh;
  Node8* bvh_nodes;
  TraversalTriangle* bvh_triangles;
} typedef DeviceConstantMemory;

struct OptixBVH {
  int initialized;
  OptixTraversableHandle traversable;
  void* bvh_data;
  OptixPipeline pipeline;
  OptixShaderBindingTable shaders;
  DeviceConstantMemory* params;
  int force_dmm_usage;
} typedef OptixBVH;

struct DeviceInfo {
  size_t global_mem_size;
  DeviceArch arch;
  int rt_core_version;
} typedef DeviceInfo;

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
  DeviceBuffer* sky_hdri_luts;
  int max_ray_depth;
  int reservoir_size;
  int offline_samples;
  int light_resampling;
  Scene scene;
  DenoisingMode denoiser;
  int temporal_frames;
  DeviceBuffer* randoms;
  int shading_mode;
  RGBAhalf** bloom_mips_gpu;
  int bloom_mips_count;
  RGBAhalf** lens_flare_buffers_gpu;
  int snap_resolution;
  OutputImageFormat image_format;
  int post_process_menu;
  General settings;
  AtmoSettings atmo_settings;
  void* denoise_setup;
  Jitter jitter;
  int accum_mode;
  RayEmitter emitter;
  ReSTIRSettings restir;
  DeviceBuffer* raydir_buffer;
  DeviceBuffer* trace_result_buffer;
  DeviceBuffer* state_buffer;
  TextureAtlas tex_atlas;
  OptixDeviceContext optix_ctx;
  OptixBVH optix_bvh;
  BVHType bvh_type;
  int luminary_bvh_initialized;
  DeviceInfo device_info;
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

#endif /* UTILS_H */
