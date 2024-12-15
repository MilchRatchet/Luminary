#ifndef LUMINARY_DEVICE_H
#define LUMINARY_DEVICE_H

#include "device_bsdf.h"
#include "device_light.h"
#include "device_memory.h"
#include "device_mesh.h"
#include "device_output.h"
#include "device_post.h"
#include "device_renderer.h"
#include "device_sky.h"
#include "device_staging_manager.h"
#include "device_utils.h"
#include "kernel.h"
#include "optix_bvh.h"
#include "optix_kernel.h"
#include "texture.h"

// Set of architectures supported by Luminary
enum DeviceArch {
  DEVICE_ARCH_UNKNOWN   = 0,
  DEVICE_ARCH_PASCAL    = 1,
  DEVICE_ARCH_VOLTA     = 2,
  DEVICE_ARCH_TURING    = 3,
  DEVICE_ARCH_AMPERE    = 4,
  DEVICE_ARCH_ADA       = 5,
  DEVICE_ARCH_HOPPER    = 6,
  DEVICE_ARCH_BLACKWELL = 7
} typedef DeviceArch;

struct DeviceProperties {
  char name[256];
  DeviceArch arch;
  uint32_t rt_core_version;
  size_t memory_size;
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

struct Device {
  uint32_t index;
  DeviceProperties properties;
  DeviceOptixProperties optix_properties;
  SampleCountSlice sample_count;
  uint32_t undersampling_state;
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
  CUstream stream_secondary;
  CUstream stream_callbacks;
  CUevent event_queue_render;
  CUevent event_queue_output;
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
  OptixBVHInstanceCache* optix_instance_cache;
  OptixBVH* optix_bvh_ias;
  OptixBVH* optix_bvh_light;
  DeviceSkyLUT* sky_lut;
  DeviceSkyHDRI* sky_hdri;
  DeviceBSDFLUT* bsdf_lut;
  DevicePost* post;
  DeviceRenderer* renderer;
  DeviceOutput* output;
} typedef Device;

void _device_init(void);
void _device_shutdown(void);

LuminaryResult device_create(Device** device, uint32_t index);
LuminaryResult device_register_as_main(Device* device);
LuminaryResult device_unregister_as_main(Device* device);
LuminaryResult device_compile_kernels(Device* device, CUlibrary library);
LuminaryResult device_load_embedded_data(Device* device);
LuminaryResult device_update_scene_entity(Device* device, const void* object, SceneEntity entity);
LuminaryResult device_update_dynamic_const_mem(Device* device, uint32_t sample_id);
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
LuminaryResult device_build_bsdf_lut(Device* device, BSDFLUT* bsdf_lut);
LuminaryResult device_update_bsdf_lut(Device* device, const BSDFLUT* bsdf_lut);
LuminaryResult device_clear_lighting_buffers(Device* device);
LuminaryResult device_setup_undersampling(Device* device, uint32_t undersampling);
LuminaryResult device_update_sample_count(Device* device, SampleCountSlice* sample_count);
LuminaryResult device_register_callbacks(
  Device* device, CUhostFn render_callback_func, CUhostFn output_callback_func, DeviceCommonCallbackData callback_data);
LuminaryResult device_start_render(Device* device, DeviceRendererQueueArgs* args);
LuminaryResult device_continue_render(Device* device, SampleCountSlice* sample_count, DeviceRenderCallbackData* callback_data);
LuminaryResult device_set_abort(Device* device);
LuminaryResult device_unset_abort(Device* device);
LuminaryResult device_get_gbuffer_meta_data(Device* device, uint32_t x, uint32_t y, GBufferMetaData* data);
LuminaryResult device_destroy(Device** device);

#endif /* LUMINARY_DEVICE_H */
