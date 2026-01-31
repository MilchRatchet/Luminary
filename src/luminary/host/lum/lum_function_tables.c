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

static LuminaryResult _lum_function_load_settings(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinSettings* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  *dst = vm->host->settings;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_settings(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinSettings* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  vm->host->settings = *src;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Camera
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_camera(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinCamera* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  *dst = vm->host->camera;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_camera(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinCamera* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  vm->host->camera = *src;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Ocean
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_ocean(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinOcean* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  *dst = vm->host->ocean;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_ocean(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinOcean* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  vm->host->ocean = *src;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Sky
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_sky(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinSky* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  *dst = vm->host->sky;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_sky(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinSky* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  vm->host->sky = *src;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Cloud
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_cloud(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinCloud* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  *dst = vm->host->cloud;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_cloud(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinCloud* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  vm->host->cloud = *src;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Fog
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_fog(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinFog* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  *dst = vm->host->fog;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_fog(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinFog* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  vm->host->fog = *src;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Particles
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_particles(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinParticles* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  *dst = vm->host->particles;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_particles(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinParticles* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  vm->host->particles = *src;

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
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = (const LumFunctionLoad) 0};

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
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = (const LumFunctionStore) 0};
