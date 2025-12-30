#include "lum_function_tables.h"

#include "internal_error.h"

////////////////////////////////////////////////////////////////////
// Utils
////////////////////////////////////////////////////////////////////

LuminaryResult lum_function_resolve_stack_address(LumVirtualMachine* vm, const LumMemoryAllocation* mem, void** ptr) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(mem);
  __CHECK_NULL_ARGUMENT(ptr);

  __DEBUG_ASSERT((mem->offset & LUM_MEMORY_CONSTANT_MEMORY_SPACE_BIT) == 0);

  uint8_t* base_ptr = (uint8_t*) vm->stack_memory;
  *ptr              = (void*) (base_ptr + (mem->offset & LUM_MEMORY_OFFSET_MASK));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_function_resolve_generic_address(LumVirtualMachine* vm, const LumMemoryAllocation* mem, const void** ptr) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(mem);
  __CHECK_NULL_ARGUMENT(ptr);

  const uint8_t* base_ptr;
  if (mem->offset & LUM_MEMORY_CONSTANT_MEMORY_SPACE_BIT)
    base_ptr = (const uint8_t*) vm->constant_memory;
  else
    base_ptr = (const uint8_t*) vm->stack_memory;

  *ptr = (const void*) (base_ptr + (mem->offset & LUM_MEMORY_OFFSET_MASK));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Settings
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_settings(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LuminaryRendererSettings* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  __FAILURE_HANDLE(luminary_host_get_settings(host, dst));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_settings(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LuminaryRendererSettings* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  __FAILURE_HANDLE(luminary_host_set_settings(host, src));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Camera
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_camera(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LuminaryCamera* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  __FAILURE_HANDLE(luminary_host_get_camera(host, dst));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_camera(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LuminaryCamera* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  __FAILURE_HANDLE(luminary_host_set_camera(host, src));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Ocean
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_ocean(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LuminaryOcean* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  __FAILURE_HANDLE(luminary_host_get_ocean(host, dst));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_ocean(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LuminaryOcean* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  __FAILURE_HANDLE(luminary_host_set_ocean(host, src));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Sky
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_sky(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LuminarySky* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  __FAILURE_HANDLE(luminary_host_get_sky(host, dst));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_sky(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LuminarySky* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  __FAILURE_HANDLE(luminary_host_set_sky(host, src));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Cloud
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_cloud(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LuminaryCloud* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  __FAILURE_HANDLE(luminary_host_get_cloud(host, dst));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_cloud(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LuminaryCloud* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  __FAILURE_HANDLE(luminary_host_set_cloud(host, src));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Fog
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_fog(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LuminaryFog* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  __FAILURE_HANDLE(luminary_host_get_fog(host, dst));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_fog(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LuminaryFog* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  __FAILURE_HANDLE(luminary_host_set_fog(host, src));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Particles
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_particles(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LuminaryParticles* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  __FAILURE_HANDLE(luminary_host_get_particles(host, dst));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_particles(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LuminaryParticles* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  __FAILURE_HANDLE(luminary_host_set_particles(host, src));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Tables
////////////////////////////////////////////////////////////////////

const LumFunctionLoad lum_function_tables_ldg[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_RGBF]             = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_VEC3]             = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_UINT]             = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_BOOL]             = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_FLOAT]            = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_ENUM]             = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_SETTINGS]         = (const LumFunctionLoad) _lum_function_load_settings,
  [LUM_BUILTIN_TYPE_CAMERA]           = (const LumFunctionLoad) _lum_function_load_camera,
  [LUM_BUILTIN_TYPE_OCEAN]            = (const LumFunctionLoad) _lum_function_load_ocean,
  [LUM_BUILTIN_TYPE_SKY]              = (const LumFunctionLoad) _lum_function_load_sky,
  [LUM_BUILTIN_TYPE_CLOUD]            = (const LumFunctionLoad) _lum_function_load_cloud,
  [LUM_BUILTIN_TYPE_FOG]              = (const LumFunctionLoad) _lum_function_load_fog,
  [LUM_BUILTIN_TYPE_PARTICLES]        = (const LumFunctionLoad) _lum_function_load_particles,
  [LUM_BUILTIN_TYPE_MATERIAL]         = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_INSTANCE]         = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_LUMINARY]         = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_FILE]             = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_VERSIONCONTROL]   = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = (const LumFunctionLoad) 0};

const LumFunctionStore lum_function_tables_stg[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_RGBF]             = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_VEC3]             = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_UINT]             = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_BOOL]             = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_FLOAT]            = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_ENUM]             = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_SETTINGS]         = (const LumFunctionStore) _lum_function_store_settings,
  [LUM_BUILTIN_TYPE_CAMERA]           = (const LumFunctionStore) _lum_function_store_camera,
  [LUM_BUILTIN_TYPE_OCEAN]            = (const LumFunctionStore) _lum_function_store_ocean,
  [LUM_BUILTIN_TYPE_SKY]              = (const LumFunctionStore) _lum_function_store_sky,
  [LUM_BUILTIN_TYPE_CLOUD]            = (const LumFunctionStore) _lum_function_store_cloud,
  [LUM_BUILTIN_TYPE_FOG]              = (const LumFunctionStore) _lum_function_store_fog,
  [LUM_BUILTIN_TYPE_PARTICLES]        = (const LumFunctionStore) _lum_function_store_particles,
  [LUM_BUILTIN_TYPE_MATERIAL]         = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_INSTANCE]         = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_LUMINARY]         = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_FILE]             = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_VERSIONCONTROL]   = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = (const LumFunctionStore) 0};
