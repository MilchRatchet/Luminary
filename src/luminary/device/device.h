#ifndef LUMINARY_DEVICE_H
#define LUMINARY_DEVICE_H

#include "device_bsdf.h"
#include "device_cloud.h"
#include "device_light.h"
#include "device_memory.h"
#include "device_mesh.h"
#include "device_omm.h"
#include "device_output.h"
#include "device_particle.h"
#include "device_post.h"
#include "device_renderer.h"
#include "device_result_interface.h"
#include "device_sky.h"
#include "device_staging_manager.h"
#include "device_utils.h"
#include "kernel.h"
#include "optix_bvh.h"
#include "optix_kernel.h"
#include "texture.h"

// Architectures that were already legacy when Luminary was created are omitted.
enum DeviceArch {
  DEVICE_ARCH_UNKNOWN,
  DEVICE_ARCH_PASCAL,
  DEVICE_ARCH_VOLTA,
  DEVICE_ARCH_TURING,
  DEVICE_ARCH_AMPERE_HPC,
  DEVICE_ARCH_AMPERE_CONSUMER,
  DEVICE_ARCH_ADA,
  DEVICE_ARCH_HOPPER,
  DEVICE_ARCH_BLACKWELL_HPC,
  DEVICE_ARCH_BLACKWELL_CONSUMER,
  DEVICE_ARCH_COUNT
} typedef DeviceArch;

struct DeviceProperties {
  char name[256];
  DeviceArch arch;
  uint32_t rt_core_version;
  size_t memory_size;
  uint32_t major;
  uint32_t minor;
  uint32_t sm_count;
  size_t l2_cache_size;
  uint32_t max_block_count;
} typedef DeviceProperties;

struct DeviceConstantMemoryDirtyProperties {
  bool is_dirty;
  bool update_everything;
  DeviceConstantMemoryMember member;
} typedef DeviceConstantMemoryDirtyProperties;

struct DeviceOptixProperties {
  uint32_t max_trace_depth;
  uint32_t max_traversable_graph_depth;
  uint32_t max_primitives_per_gas;
  uint32_t max_instances_per_ias;
  uint32_t rtcore_version;
  uint32_t max_instance_id;
  uint32_t num_bits_instance_visibility_mask;
  uint32_t max_sbt_records_per_gas;
  uint32_t max_sbt_offset;
  uint32_t shader_execution_reordering;
} typedef DeviceOptixProperties;

enum GBufferMetaState { GBUFFER_META_STATE_NOT_READY, GBUFFER_META_STATE_QUEUED, GBUFFER_META_STATE_READY } typedef GBufferMetaState;

enum DeviceState { DEVICE_STATE_DISABLED, DEVICE_STATE_ENABLED, DEVICE_STATE_UNAVAILABLE } typedef DeviceState;

struct Device {
  DeviceState state;
  uint32_t index;
  DeviceProperties properties;
  DeviceOptixProperties optix_properties;
  SampleCountSlice sample_count;
  uint32_t undersampling_state;
  uint32_t aggregate_sample_count;
  bool exit_requested;
  bool optix_callback_error;
  bool is_main_device;
  bool state_abort;
  CUdevice cuda_device;
  CUcontext cuda_ctx;
  CUDAKernel* cuda_kernels[CUDA_KERNEL_TYPE_COUNT];
  CUdeviceptr cuda_device_const_memory;
  OptixDeviceContext optix_ctx;
  OptixKernel* optix_kernels[OPTIX_KERNEL_TYPE_COUNT];
  CUstream stream_main;
  CUstream stream_output;
  CUstream stream_abort;
  CUstream stream_callbacks;
  CUevent event_queue_render;
  CUevent event_queue_output;
  CUevent event_queue_gbuffer_meta;
  DevicePointers buffers;
  STAGING DeviceConstantMemory* constant_memory;
  DeviceConstantMemoryDirtyProperties constant_memory_dirty;
  STAGING uint32_t* abort_flags;
  STAGING GBufferMetaData* gbuffer_meta_dst;
  DeviceTexture* moon_albedo_tex;
  DeviceTexture* moon_normal_tex;
  DeviceStagingManager* staging_manager;
  ARRAY DeviceTexture** textures;
  uint32_t num_materials;
  uint32_t num_instances;
  ARRAY DeviceMesh** meshes;
  ARRAY OpacityMicromap** omms;
  bool meshes_need_building;
  OptixBVHInstanceCache* optix_instance_cache;
  OptixBVH* optix_bvh_ias;
  OptixBVH* optix_bvh_light;
  DeviceSkyLUT* sky_lut;
  DeviceSkyHDRI* sky_hdri;
  DeviceSkyStars* sky_stars;
  DeviceBSDFLUT* bsdf_lut;
  DevicePost* post;
  DeviceRenderer* renderer;
  DeviceOutput* output;
  GBufferMetaState gbuffer_meta_state;
  DeviceCloudNoise* cloud_noise;
  DeviceParticlesHandle* particles_handle;
} typedef Device;

struct DeviceRegisterCallbackFuncs {
  CUhostFn render_continue_callback_func;
  CUhostFn render_finished_callback_func;
  CUhostFn output_callback_func;
} typedef DeviceRegisterCallbackFuncs;

void _device_init(void);
void _device_shutdown(void);

LuminaryResult device_create(Device** device, uint32_t index);
LuminaryResult device_register_as_main(Device* device);
LuminaryResult device_unregister_as_main(Device* device);
LuminaryResult device_set_enable(Device* device, bool enable);
LuminaryResult device_compile_kernels(Device* device, CUlibrary library);
LuminaryResult device_load_embedded_data(Device* device);
LuminaryResult device_get_internal_resolution(Device* device, uint32_t* width, uint32_t* height);
LuminaryResult device_update_scene_entity(Device* device, const void* object, SceneEntity entity);
LuminaryResult device_update_dynamic_const_mem(Device* device, uint32_t sample_id, uint16_t x, uint16_t y);
LuminaryResult device_update_depth_const_mem(Device* device, uint8_t depth);
LuminaryResult device_sync_constant_memory(Device* device);
LuminaryResult device_allocate_work_buffers(Device* device);
LuminaryResult device_update_mesh(Device* device, const Mesh* mesh);
LuminaryResult device_apply_instance_updates(Device* device, const ARRAY MeshInstanceUpdate* instance_updates);
LuminaryResult device_add_textures(Device* device, const Texture** textures, uint32_t num_textures);
LuminaryResult device_apply_material_updates(
  Device* device, const ARRAY MaterialUpdate* updates, const ARRAY DeviceMaterialCompressed* materials);
LuminaryResult device_build_light_tree(Device* device, LightTree* tree);
LuminaryResult device_update_light_tree_data(Device* device, LightTree* tree);
LuminaryResult device_build_sky_lut(Device* device, SkyLUT* sky_lut);
LuminaryResult device_update_sky_lut(Device* device, const SkyLUT* sky_lut);
LuminaryResult device_build_sky_hdri(Device* device, SkyHDRI* sky_hdri);
LuminaryResult device_update_sky_hdri(Device* device, const SkyHDRI* sky_hdri);
LuminaryResult device_update_sky_stars(Device* device, const SkyStars* sky_stars);
LuminaryResult device_build_bsdf_lut(Device* device, BSDFLUT* bsdf_lut);
LuminaryResult device_update_bsdf_lut(Device* device, const BSDFLUT* bsdf_lut);
LuminaryResult device_update_cloud_noise(Device* device, const Cloud* cloud);
LuminaryResult device_update_particles(Device* device, const Particles* particles);
LuminaryResult device_update_post(Device* device, const Camera* camera);
LuminaryResult device_clear_lighting_buffers(Device* device);
LuminaryResult device_setup_undersampling(Device* device, uint32_t undersampling);
LuminaryResult device_update_sample_count(Device* device, SampleCountSlice* sample_count);
LuminaryResult device_register_callbacks(Device* device, DeviceRegisterCallbackFuncs funcs, DeviceCommonCallbackData callback_data);
LuminaryResult device_set_output_dirty(Device* device);
LuminaryResult device_update_output_properties(Device* device, uint32_t width, uint32_t height);
LuminaryResult device_update_output_camera_params(Device* device, const Camera* camera);
LuminaryResult device_add_output_request(Device* device, OutputRequestProperties properties);
LuminaryResult device_start_render(Device* device, DeviceRendererQueueArgs* args);
LuminaryResult device_validate_render_callback(Device* device, DeviceRenderCallbackData* callback_data, bool* is_valid);
LuminaryResult device_finish_render_iteration(Device* device, SampleCountSlice* sample_count, DeviceRenderCallbackData* callback_data);
LuminaryResult device_continue_render(Device* device);
LuminaryResult device_update_render_time(Device* device, DeviceRenderCallbackData* callback_data);
LuminaryResult device_handle_result_sharing(Device* device, DeviceResultInterface* interface);
LuminaryResult device_set_abort(Device* device);
LuminaryResult device_unset_abort(Device* device);
LuminaryResult device_query_gbuffer_meta(Device* device);
LuminaryResult device_get_gbuffer_meta(Device* device, uint16_t x, uint16_t y, GBufferMetaData* data);
LuminaryResult device_destroy(Device** device);

#endif /* LUMINARY_DEVICE_H */
