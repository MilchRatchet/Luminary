#ifndef UTILS_H
#define UTILS_H

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

#ifndef ONE_OVER_PI
#define ONE_OVER_PI 0.31830988618f
#endif

enum OutputImageFormat { IMGFORMAT_PNG = 0, IMGFORMAT_QOI = 1 } typedef OutputImageFormat;

enum OutputVariable {
  OUTPUT_VARIABLE_BEAUTY            = 0,
  OUTPUT_VARIABLE_DIRECT_LIGHTING   = 1,
  OUTPUT_VARIABLE_INDIRECT_LIGHTING = 2,
  OUTPUT_VARIABLE_ALBEDO_GUIDANCE   = 3,
  OUTPUT_VARIABLE_NORMAL_GUIDANCE   = 4,
  OUTPUT_VARIABLE_COUNT             = 5
} typedef OutputVariable;

enum ToyShape { TOY_SPHERE = 0, TOY_PLANE = 1 } typedef ToyShape;

enum SnapResolution { SNAP_RESOLUTION_WINDOW = 0, SNAP_RESOLUTION_RENDER = 1 } typedef SnapResolution;

enum DenoisingMode { DENOISING_OFF = 0, DENOISING_ON = 1, DENOISING_UPSCALING = 2 } typedef DenoisingMode;

enum VolumeType { VOLUME_TYPE_FOG = 0, VOLUME_TYPE_OCEAN = 1, VOLUME_TYPE_PARTICLE = 2, VOLUME_TYPE_NONE = 0xFFFFFFFF } typedef VolumeType;

struct Star {
  float altitude;
  float azimuth;
  float radius;
  float intensity;
} typedef Star;

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

struct GlobalMaterial {
  RGBF default_material;
  int lights_active;
  int light_tree_active;
  float alpha_cutoff;
  int colored_transparency;
  int invert_roughness;
  int override_materials;
  int enable_ior_shadowing;
  float caustic_roughness_clamp;
} typedef GlobalMaterial;

struct Scene {
  Camera camera;
  Triangle* triangles;
  TriangleLight* triangle_lights;
  TriangleGeomData triangle_data;
  TriangleGeomData triangle_lights_data;
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
  uint32_t width;
  uint32_t height;
  uint32_t internal_width;
  uint32_t internal_height;
  uint16_t user_selected_x;
  uint16_t user_selected_y;
  int realtime;
  DeviceBuffer* ior_stack;
  DeviceBuffer* trace_tasks;
  DeviceBuffer* trace_counts;
  DeviceBuffer* trace_results;
  DeviceBuffer* task_counts;
  DeviceBuffer* task_offsets;
  DeviceBuffer* frame_variance;
  DeviceBuffer* frame_accumulate;
  DeviceBuffer* frame_post;
  DeviceBuffer* frame_final;
  DeviceBuffer* frame_direct_buffer;
  DeviceBuffer* frame_direct_accumulate;
  DeviceBuffer* frame_indirect_buffer;
  DeviceBuffer* frame_indirect_accumulate;
  DeviceBuffer* albedo_buffer;
  DeviceBuffer* normal_buffer;
  DeviceBuffer* records;
  DeviceBuffer* buffer_8bit;
  DeviceBuffer* cloud_noise;
  DeviceBuffer* sky_ms_luts;
  DeviceBuffer* sky_tm_luts;
  DeviceBuffer* sky_hdri_luts;
  DeviceBuffer* sky_moon_albedo_tex;
  DeviceBuffer* sky_moon_normal_tex;
  DeviceBuffer* bsdf_energy_lut;
  DeviceBuffer* bluenoise_1D;
  DeviceBuffer* bluenoise_2D;
  int max_ray_depth;
  int reservoir_size;
  int offline_samples;
  Scene scene;
  DenoisingMode denoiser;
  float temporal_frames;
  int shading_mode;
  RGBF** bloom_mips_gpu;
  int bloom_mips_count;
  RGBF** lens_flare_buffers_gpu;
  int snap_resolution;
  OutputImageFormat image_format;
  int post_process_menu;
  OutputVariable output_variable;
  General settings;
  AtmoSettings atmo_settings;
  void* denoise_setup;
  int accumulate;
  RayEmitter emitter;
  RISSettings ris_settings;
  BridgeSettings bridge_settings;
  DeviceBuffer* hit_id_history;
  DeviceBuffer* state_buffer;
  TextureAtlas tex_atlas;
  OptixDeviceContext optix_ctx;
  OptixKernel optix_kernel;
  OptixKernel optix_kernel_geometry;
  OptixKernel optix_kernel_volume;
  OptixKernel optix_kernel_particle;
  OptixKernel optix_kernel_volume_bridges;
  OptixBVH optix_bvh;
  OptixBVH optix_bvh_shadow;
  OptixBVH optix_bvh_light;
  BVHType bvh_type;
  int luminary_bvh_initialized;
  ParticlesInstance particles_instance;
  DeviceInfo device_info;
  int undersampling_setting;
  int undersampling;
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
