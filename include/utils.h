#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>

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
#define LUM_UNUSED(x) ((void) (x))

#ifndef PI
#define PI 3.141592653589f
#endif

#ifndef ONE_OVER_PI
#define ONE_OVER_PI 0.31830988618f
#endif

#define OPTIXRT_NUM_GROUPS 3

#define RESTIR_CANDIDATE_POOL_MAX (1 << 20)

enum RayIterationType { TYPE_CAMERA = 0, TYPE_LIGHT = 1, TYPE_BOUNCE = 2 } typedef RayIterationType;

enum LightID : uint32_t {
  LIGHT_ID_SUN               = 0xffffffffu,
  LIGHT_ID_TOY               = 0xfffffffeu,
  LIGHT_ID_NONE              = 0xfffffff1u,
  LIGHT_ID_ANY               = 0xfffffff0u,
  LIGHT_ID_TRIANGLE_ID_LIMIT = 0x7fffffffu
} typedef LightID;

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

enum OutputVariable {
  OUTPUT_VARIABLE_BEAUTY            = 0,
  OUTPUT_VARIABLE_ALBEDO_GUIDANCE   = 1,
  OUTPUT_VARIABLE_NORMAL_GUIDANCE   = 2,
  OUTPUT_VARIABLE_DIRECT_LIGHTING   = 3,
  OUTPUT_VARIABLE_INDIRECT_LIGHTING = 4,
  OUTPUT_VARIABLE_COUNT             = 5
} typedef OutputVariable;

enum ToyShape { TOY_SPHERE = 0, TOY_PLANE = 1 } typedef ToyShape;

enum ToneMap {
  TONEMAP_NONE       = 0,
  TONEMAP_ACES       = 1,
  TONEMAP_REINHARD   = 2,
  TONEMAP_UNCHARTED2 = 3,
  TONEMAP_AGX        = 4,
  TONEMAP_AGX_PUNCHY = 5,
  TONEMAP_AGX_CUSTOM = 6
} typedef ToneMap;

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

enum DenoisingMode { DENOISING_OFF = 0, DENOISING_ON = 1, DENOISING_UPSCALING = 2 } typedef DenoisingMode;

enum VolumeType { VOLUME_TYPE_FOG = 0, VOLUME_TYPE_OCEAN = 1, VOLUME_TYPE_PARTICLE } typedef VolumeType;

enum CameraApertureShape { CAMERA_APERTURE_ROUND = 0, CAMERA_APERTURE_BLADED = 1 } typedef CameraApertureShape;

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

struct CommandlineOptions {
  int aov_mode;
  int width;
  int height;
  int dmm_active;
  int omm_active;
} typedef CommandlineOptions;

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
  CameraApertureShape aperture_shape;
  int aperture_blade_count;
  float exposure;
  float max_exposure;
  float min_exposure;
  int auto_exposure;
  float far_clip_distance;
  ToneMap tonemap;
  float agx_custom_slope;
  float agx_custom_power;
  float agx_custom_saturation;
  Filter filter;
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
  float russian_roulette_threshold;
  int use_color_correction;
  RGBF color_correction;
  int do_firefly_clamping;
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
  RGBF emission;
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
  float droplet_diameter;
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
  float mie_diameter;
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
  float moon_tex_offset;
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
  float mie_diameter;
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

enum JerlovWaterType {
  JERLOV_WATER_TYPE_I   = 0,
  JERLOV_WATER_TYPE_IA  = 1,
  JERLOV_WATER_TYPE_IB  = 2,
  JERLOV_WATER_TYPE_II  = 3,
  JERLOV_WATER_TYPE_III = 4,
  JERLOV_WATER_TYPE_1C  = 5,
  JERLOV_WATER_TYPE_3C  = 6,
  JERLOV_WATER_TYPE_5C  = 7,
  JERLOV_WATER_TYPE_7C  = 8,
  JERLOV_WATER_TYPE_9C  = 9
} typedef JerlovWaterType;

struct Ocean {
  int active;
  float height;
  float amplitude;
  float frequency;
  float choppyness;
  float refractive_index;
  JerlovWaterType water_type;
} typedef Ocean;

struct Fog {
  int active;
  float density;
  float droplet_diameter;
  float height;
  float dist;
} typedef Fog;

struct Jitter {
  float x;
  float y;
  float prev_x;
  float prev_y;
} typedef Jitter;

enum LightSideMode { LIGHT_SIDE_MODE_BOTH = 0, LIGHT_SIDE_MODE_ONE_CW = 1, LIGHT_SIDE_MODE_ONE_CCW = 2 } typedef LightSideMode;

struct GlobalMaterial {
  RGBF default_material;
  int lights_active;
  float alpha_cutoff;
  int colored_transparency;
  int invert_roughness;
  int override_materials;
  LightSideMode light_side_mode;
} typedef GlobalMaterial;

struct Particles {
  int active;
  uint32_t seed;
  uint32_t count;
  RGBF albedo;
  float speed;
  float direction_altitude;
  float direction_azimuth;
  float phase_diameter;
  float scale;
  float size;
  float size_variation;
} typedef Particles;

struct Scene {
  Camera camera;
  Triangle* triangles;
  TriangleLight* triangle_lights;
  TriangleGeomData triangle_data;
  unsigned int triangle_lights_count;
  uint16_t materials_count;
  PackedMaterial* materials;
  Ocean ocean;
  Sky sky;
  Toy toy;
  Fog fog;
  GlobalMaterial material;
  Particles particles;
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
  DeviceBuffer* luminance;
  int luminance_length;
  DeviceBuffer* material;
  int material_length;
  DeviceBuffer* normal;
  int normal_length;
} typedef TextureAtlas;

struct ReSTIRSettings {
  int initial_reservoir_size;
  int light_candidate_pool_size_log2;
  TriangleLight* presampled_triangle_lights;
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
  uint32_t* ior_stack;
  RGBF* frame_buffer;
  RGBF* frame_temporal;
  float* frame_variance;
  RGBF* frame_accumulate;
  RGBF* frame_direct_buffer;
  RGBF* frame_direct_accumulate;
  RGBF* frame_indirect_buffer;
  RGBF* frame_indirect_accumulate;
  RGBF* frame_output;
  RGBF* albedo_buffer;
  RGBF* normal_buffer;
  RGBF* light_records;
  RGBF* bounce_records;
  RGBF* bounce_records_history;
  PackedGBufferData* packed_gbuffer_history;
  XRGB8* buffer_8bit;
  vec3* raydir_buffer;
  float* mis_buffer;
  TraceResult* trace_result_buffer;
  uint8_t* state_buffer;
  DeviceTexture* albedo_atlas;
  DeviceTexture* luminance_atlas;
  DeviceTexture* material_atlas;
  DeviceTexture* normal_atlas;
  DeviceTexture* cloud_noise;
  DeviceTexture* sky_ms_luts;
  DeviceTexture* sky_tm_luts;
  DeviceTexture* sky_hdri_luts;
  DeviceTexture* sky_moon_albedo_tex;
  DeviceTexture* sky_moon_normal_tex;
  DeviceTexture* bsdf_energy_lut;
  uint16_t* bluenoise_1D;
  uint32_t* bluenoise_2D;
  uint32_t* light_candidates;
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
  int width;
  int height;
  int output_width;
  int output_height;
  PackedMaterial* materials;
  vec3 sun_pos;
  vec3 moon_pos;
  int shading_mode;
  int aov_mode;
  OutputVariable output_variable;
  RGBF* bloom_scratch;
  RayEmitter emitter;
  int accum_mode;
  OptixTraversableHandle optix_bvh;
  OptixTraversableHandle optix_bvh_particles;
  Node8* bvh_nodes;
  TraversalTriangle* bvh_triangles;
  Quad* particle_quads;
} typedef DeviceConstantMemory;

struct OptixKernel {
  OptixPipeline pipeline;
  OptixShaderBindingTable shaders;
  DeviceConstantMemory* params;
} typedef OptixKernel;

struct OptixBVH {
  int initialized;
  size_t bvh_mem_size;
  OptixTraversableHandle traversable;
  void* bvh_data;
  int force_dmm_usage;
  int disable_omm;
} typedef OptixBVH;

struct ParticlesInstance {
  OptixKernel kernel;
  OptixBVH optix;
  uint32_t triangle_count;
  uint32_t vertex_count;
  DeviceBuffer* vertex_buffer;
  uint32_t index_count;
  DeviceBuffer* index_buffer;
  DeviceBuffer* quad_buffer;
} typedef ParticlesInstance;

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
  DeviceBuffer* ior_stack;
  DeviceBuffer* light_trace;
  DeviceBuffer* bounce_trace;
  DeviceBuffer* light_trace_count;
  DeviceBuffer* bounce_trace_count;
  DeviceBuffer* trace_results;
  DeviceBuffer* task_counts;
  DeviceBuffer* task_offsets;
  DeviceBuffer* light_sample_history;
  DeviceBuffer* frame_buffer;
  DeviceBuffer* frame_temporal;
  DeviceBuffer* frame_variance;
  DeviceBuffer* frame_accumulate;
  DeviceBuffer* frame_output;
  DeviceBuffer* frame_direct_buffer;
  DeviceBuffer* frame_direct_accumulate;
  DeviceBuffer* frame_indirect_buffer;
  DeviceBuffer* frame_indirect_accumulate;
  DeviceBuffer* albedo_buffer;
  DeviceBuffer* normal_buffer;
  DeviceBuffer* light_records;
  DeviceBuffer* bounce_records;
  DeviceBuffer* bounce_records_history;
  DeviceBuffer* buffer_8bit;
  DeviceBuffer* light_candidates;
  DeviceBuffer* cloud_noise;
  DeviceBuffer* sky_ms_luts;
  DeviceBuffer* sky_tm_luts;
  DeviceBuffer* sky_hdri_luts;
  DeviceBuffer* sky_moon_albedo_tex;
  DeviceBuffer* sky_moon_normal_tex;
  DeviceBuffer* bsdf_energy_lut;
  DeviceBuffer* bluenoise_1D;
  DeviceBuffer* bluenoise_2D;
  DeviceBuffer* mis_buffer;
  DeviceBuffer* packed_gbuffer_history;
  int max_ray_depth;
  int reservoir_size;
  int offline_samples;
  Scene scene;
  DenoisingMode denoiser;
  int temporal_frames;
  int shading_mode;
  RGBF** bloom_mips_gpu;
  int bloom_mips_count;
  RGBF** lens_flare_buffers_gpu;
  int snap_resolution;
  OutputImageFormat image_format;
  int post_process_menu;
  int aov_mode;
  OutputVariable output_variable;
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
  OptixKernel optix_kernel;
  OptixBVH optix_bvh;
  BVHType bvh_type;
  int luminary_bvh_initialized;
  ParticlesInstance particles_instance;
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
    if (count >= length) {                        \
      length = count;                             \
      length += 1;                                \
      length *= 2;                                \
      ptr = safe_realloc(ptr, size * length);     \
    }                                             \
  }

#endif /* UTILS_H */
