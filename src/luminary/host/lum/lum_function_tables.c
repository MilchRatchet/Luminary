#include "lum_function_tables.h"

#include <string.h>

#include "host/wavefront.h"
#include "internal_error.h"
#include "internal_path.h"
#include "lum_wavefront.h"

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

static LuminaryResult lum_function_resolve_string_address(LumVirtualMachine* vm, const LumBuiltinString* string, const char** ptr) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(string);
  __CHECK_NULL_ARGUMENT(ptr);

  const uint8_t* base_ptr = (const uint8_t*) vm->constant_memory;

  *ptr = (string->const_mem_size > 0) ? (const char*) (base_ptr + string->const_mem_address) : (const char*) 0;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// RGBF
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_rgbf(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LuminaryRGBF* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  memset(dst, 0, sizeof(LuminaryRGBF));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_rgbf(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Vec3
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_vec3(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LuminaryVec3* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  memset(dst, 0, sizeof(LuminaryVec3));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_vec3(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

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
// Mesh
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_mesh(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinMesh* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  const LumBuiltinString* string;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->name, (const void**) &string));

  const char* name;
  __FAILURE_HANDLE(lum_function_resolve_string_address(vm, string, &name));

  bool found;
  uint32_t id;
  __FAILURE_HANDLE(dictionary_find_by_name(vm->host->mesh_name_dict, name, &id, &found));

  dst->id = (found) ? id : MESH_ID_INVALID;

  if (found == false)
    warn_message("Failed to find mesh '%s'.", name);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_mesh(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Texture
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_texture(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinTexture* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  const LumBuiltinString* string;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->name, (const void**) &string));

  const char* name;
  __FAILURE_HANDLE(lum_function_resolve_string_address(vm, string, &name));

  bool found;
  uint32_t id;
  __FAILURE_HANDLE(dictionary_find_by_name(vm->host->texture_name_dict, name, &id, &found));

  dst->id = (found) ? id : TEXTURE_ID_INVALID;

  if (found == false)
    warn_message("Failed to find texture '%s'.", name);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_texture(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Material
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_material(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinMaterial* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  const LumBuiltinString* string;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->name, (const void**) &string));

  const char* name;
  __FAILURE_HANDLE(lum_function_resolve_string_address(vm, string, &name));

  bool found;
  uint32_t id;
  __FAILURE_HANDLE(dictionary_find_by_name(vm->host->material_name_dict, name, &id, &found));

  if (found) {
    *dst = vm->host->materials[id];
  }
  else {
    __FAILURE_HANDLE(lum_builtin_material_init(dst, vm->host->version));

    __FAILURE_HANDLE(array_get_num_elements(vm->host->materials, &id));

    __FAILURE_HANDLE(array_push(&vm->host->materials, dst));
    __FAILURE_HANDLE(dictionary_add_entry(vm->host->material_name_dict, id, name));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_material(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinMaterial* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  const LumBuiltinString* string;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->name, (const void**) &string));

  const char* name;
  __FAILURE_HANDLE(lum_function_resolve_string_address(vm, string, &name));

  bool found;
  uint32_t id;
  __FAILURE_HANDLE(dictionary_find_by_name(vm->host->material_name_dict, name, &id, &found));

  if (found == false)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Material '%s' is missing.", name);

  vm->host->materials[id] = *src;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Instance
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_instance(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinInstance* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  const LumBuiltinString* string;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->name, (const void**) &string));

  const char* name;
  __FAILURE_HANDLE(lum_function_resolve_string_address(vm, string, &name));

  bool found;
  uint32_t id;
  __FAILURE_HANDLE(dictionary_find_by_name(vm->host->material_name_dict, name, &id, &found));

  if (found) {
    *dst = vm->host->mesh_instances[id];
  }
  else {
    __FAILURE_HANDLE(lum_builtin_instance_init(dst, vm->host->version));

    __FAILURE_HANDLE(array_get_num_elements(vm->host->mesh_instances, &id));

    __FAILURE_HANDLE(array_push(&vm->host->mesh_instances, dst));
    __FAILURE_HANDLE(dictionary_add_entry(vm->host->mesh_instance_name_dict, id, name));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_instance(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinInstance* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  const LumBuiltinString* string;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->name, (const void**) &string));

  const char* name;
  __FAILURE_HANDLE(lum_function_resolve_string_address(vm, string, &name));

  bool found;
  uint32_t id;
  __FAILURE_HANDLE(dictionary_find_by_name(vm->host->mesh_instance_name_dict, name, &id, &found));

  if (found == false)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Instance '%s' is missing.", name);

  vm->host->mesh_instances[id] = *src;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// WavefrontObj
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_wavefrontobj(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinWavefrontObjFile* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  __FAILURE_HANDLE(lum_builtin_wavefrontobjfile_init(dst, vm->host->version));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_wavefrontobj(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinWavefrontObjFile* src;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->src, (void**) &src));

  const LumBuiltinString* string;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->name, (const void**) &string));

  const char* name;
  __FAILURE_HANDLE(lum_function_resolve_string_address(vm, string, &name));

  Path* obj_path;
  __FAILURE_HANDLE(path_extend(&obj_path, vm->working_directory, name));

  WavefrontArguments args;
  __FAILURE_HANDLE(wavefront_arguments_get_default(&args));

  if (src->name_prefix.const_mem_address != LUM_BUILTIN_STRING_INVALID_ADDRESS) {
    __FAILURE_HANDLE(lum_function_resolve_string_address(vm, &src->name_prefix, &args.name_prefix));
  }

  WavefrontContent* content;
  __FAILURE_HANDLE(wavefront_create(&content, args));

  __FAILURE_HANDLE(wavefront_read_file(content, obj_path, vm->work_queue));

  __FAILURE_HANDLE(luminary_path_destroy(&obj_path));

  uint32_t texture_count_before;
  __FAILURE_HANDLE(array_get_num_elements(vm->host->textures, &texture_count_before));

  __FAILURE_HANDLE(wavefront_content_get_textures(content, &vm->host->textures, vm->host->texture_name_dict));

  uint32_t material_count_before;
  __FAILURE_HANDLE(array_get_num_elements(vm->host->materials, &material_count_before));

  __FAILURE_HANDLE(wavefront_content_get_meshes(content, &vm->host->meshes, vm->host->mesh_name_dict, material_count_before));

  __FAILURE_HANDLE(wavefront_content_get_versioned_materials(
    content, &vm->host->materials, vm->host->material_name_dict, texture_count_before, vm->host->version));

  __FAILURE_HANDLE(wavefront_destroy(&content));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// AdaptiveSampling
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_adaptive_sampling(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinAdaptiveSampling* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  *dst = vm->host->settings.adaptive_sampling_settings;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_adaptive_sampling(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinAdaptiveSampling* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  vm->host->settings.adaptive_sampling_settings = *src;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Camera Thin Lens
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_camera_thin_lens(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinCameraThinLens* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  *dst = vm->host->camera.thin_lens;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_camera_thin_lens(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinCameraThinLens* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  vm->host->camera.thin_lens = *src;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Physical
////////////////////////////////////////////////////////////////////

static LuminaryResult _lum_function_load_camera_physical(LumVirtualMachine* vm, const LumFunctionLoadInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  LumBuiltinCameraPhysical* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &info->dst, (void**) &dst));

  *dst = vm->host->camera.physical;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_function_store_camera_physical(LumVirtualMachine* vm, const LumFunctionStoreInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(info);

  const LumBuiltinCameraPhysical* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &info->src, (const void**) &src));

  vm->host->camera.physical = *src;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Tables
////////////////////////////////////////////////////////////////////

const LumFunctionLoad lum_function_tables_ldg[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID]             = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_RGBF]             = (const LumFunctionLoad) _lum_function_load_rgbf,
  [LUM_BUILTIN_TYPE_VEC3]             = (const LumFunctionLoad) _lum_function_load_vec3,
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
  [LUM_BUILTIN_TYPE_MESH]             = (const LumFunctionLoad) _lum_function_load_mesh,
  [LUM_BUILTIN_TYPE_TEXTURE]          = (const LumFunctionLoad) _lum_function_load_texture,
  [LUM_BUILTIN_TYPE_MATERIAL]         = (const LumFunctionLoad) _lum_function_load_material,
  [LUM_BUILTIN_TYPE_INSTANCE]         = (const LumFunctionLoad) _lum_function_load_instance,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = (const LumFunctionLoad) _lum_function_load_wavefrontobj,
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = (const LumFunctionLoad) _lum_function_load_adaptive_sampling,
  [LUM_BUILTIN_TYPE_CLOUDLAYER]       = (const LumFunctionLoad) 0,
  [LUM_BUILTIN_TYPE_CAMERATHINLENS]   = (const LumFunctionLoad) _lum_function_load_camera_thin_lens,
  [LUM_BUILTIN_TYPE_CAMERAPHYSICAL]   = (const LumFunctionLoad) _lum_function_load_camera_physical,
};

const LumFunctionStore lum_function_tables_stg[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_VOID]             = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_RGBF]             = (const LumFunctionStore) _lum_function_store_rgbf,
  [LUM_BUILTIN_TYPE_VEC3]             = (const LumFunctionStore) _lum_function_store_vec3,
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
  [LUM_BUILTIN_TYPE_MESH]             = (const LumFunctionStore) _lum_function_store_mesh,
  [LUM_BUILTIN_TYPE_TEXTURE]          = (const LumFunctionStore) _lum_function_store_texture,
  [LUM_BUILTIN_TYPE_MATERIAL]         = (const LumFunctionStore) _lum_function_store_material,
  [LUM_BUILTIN_TYPE_INSTANCE]         = (const LumFunctionStore) _lum_function_store_instance,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = (const LumFunctionStore) _lum_function_store_wavefrontobj,
  [LUM_BUILTIN_TYPE_ADAPTIVESAMPLING] = (const LumFunctionStore) _lum_function_store_adaptive_sampling,
  [LUM_BUILTIN_TYPE_CLOUDLAYER]       = (const LumFunctionStore) 0,
  [LUM_BUILTIN_TYPE_CAMERATHINLENS]   = (const LumFunctionStore) _lum_function_store_camera_thin_lens,
  [LUM_BUILTIN_TYPE_CAMERAPHYSICAL]   = (const LumFunctionStore) _lum_function_store_camera_physical,
};
