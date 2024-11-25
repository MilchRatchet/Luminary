#ifndef LUMINARY_DEVICE_H
#define LUMINARY_DEVICE_H

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

struct Device {
  uint32_t index;
  DeviceProperties properties;
  SampleCountSlice sample_count;
  bool exit_requested;
  bool optix_callback_error;
  bool is_main_device;
  CUdevice cuda_device;
  CUcontext cuda_ctx;
  CUDAKernel* cuda_kernels[CUDA_KERNEL_TYPE_COUNT];
  CUdeviceptr cuda_device_const_memory;
  OptixDeviceContext optix_ctx;
  OptixKernel* optix_kernels[OPTIX_KERNEL_TYPE_COUNT];
  CUstream stream_main;
  CUstream stream_secondary;
  DevicePointers buffers;
  STAGING DeviceConstantMemory* constant_memory;
  DeviceConstantMemoryDirtyProperties constant_memory_dirty;
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
LuminaryResult device_update_dynamic_const_mem(Device* device, uint32_t sample_id, uint32_t depth);
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
LuminaryResult device_destroy(Device** device);

#endif /* LUMINARY_DEVICE_H */
