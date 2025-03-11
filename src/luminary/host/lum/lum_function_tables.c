#include "lum_function_tables.h"

#include "lum_function_implementations.h"

static const LumFunctionEntry lum_function_tables_settings[] = {
  {.name      = "init",
   .func      = &lum_function_settings_init,
   .signature = {.dst = LUM_DATA_TYPE_VOID, .src_a = LUM_DATA_TYPE_NULL, .src_b = LUM_DATA_TYPE_NULL, .src_c = LUM_DATA_TYPE_NULL},
   .is_static = true}};

static const LumFunctionEntry lum_function_tables_camera[] = {
  {.name      = "init",
   .func      = &lum_function_camera_init,
   .signature = {.dst = LUM_DATA_TYPE_VOID, .src_a = LUM_DATA_TYPE_NULL, .src_b = LUM_DATA_TYPE_NULL, .src_c = LUM_DATA_TYPE_NULL},
   .is_static = true}};

static const LumFunctionEntry lum_function_tables_ocean[] = {
  {.name      = "init",
   .func      = &lum_function_ocean_init,
   .signature = {.dst = LUM_DATA_TYPE_VOID, .src_a = LUM_DATA_TYPE_NULL, .src_b = LUM_DATA_TYPE_NULL, .src_c = LUM_DATA_TYPE_NULL},
   .is_static = true}};

static const LumFunctionEntry lum_function_tables_sky[] = {
  {.name      = "init",
   .func      = &lum_function_sky_init,
   .signature = {.dst = LUM_DATA_TYPE_VOID, .src_a = LUM_DATA_TYPE_NULL, .src_b = LUM_DATA_TYPE_NULL, .src_c = LUM_DATA_TYPE_NULL},
   .is_static = true}};

static const LumFunctionEntry lum_function_tables_cloud[] = {
  {.name      = "init",
   .func      = &lum_function_cloud_init,
   .signature = {.dst = LUM_DATA_TYPE_VOID, .src_a = LUM_DATA_TYPE_NULL, .src_b = LUM_DATA_TYPE_NULL, .src_c = LUM_DATA_TYPE_NULL},
   .is_static = true}};

static const LumFunctionEntry lum_function_tables_fog[] = {
  {.name      = "init",
   .func      = &lum_function_fog_init,
   .signature = {.dst = LUM_DATA_TYPE_VOID, .src_a = LUM_DATA_TYPE_NULL, .src_b = LUM_DATA_TYPE_NULL, .src_c = LUM_DATA_TYPE_NULL},
   .is_static = true}};

static const LumFunctionEntry lum_function_tables_particles[] = {
  {.name      = "init",
   .func      = &lum_function_particles_init,
   .signature = {.dst = LUM_DATA_TYPE_VOID, .src_a = LUM_DATA_TYPE_NULL, .src_b = LUM_DATA_TYPE_NULL, .src_c = LUM_DATA_TYPE_NULL},
   .is_static = true}};

static const LumFunctionEntry lum_function_tables_material[] = {
  {.name      = "init",
   .func      = &lum_function_material_init,
   .signature = {.dst = LUM_DATA_TYPE_VOID, .src_a = LUM_DATA_TYPE_NULL, .src_b = LUM_DATA_TYPE_NULL, .src_c = LUM_DATA_TYPE_NULL},
   .is_static = true}};

static const LumFunctionEntry lum_function_tables_instance[] = {
  {.name      = "init",
   .func      = &lum_function_instance_init,
   .signature = {.dst = LUM_DATA_TYPE_VOID, .src_a = LUM_DATA_TYPE_NULL, .src_b = LUM_DATA_TYPE_NULL, .src_c = LUM_DATA_TYPE_NULL},
   .is_static = true}};

static const LumFunctionEntry lum_function_tables_metadata[] = {
  {.name      = "setMinorVersion",
   .func      = &lum_function_metadata_setminorversion,
   .signature = {.dst = LUM_DATA_TYPE_NULL, .src_a = LUM_DATA_TYPE_UINT32, .src_b = LUM_DATA_TYPE_NULL, .src_c = LUM_DATA_TYPE_NULL},
   .is_static = true}};

const LumFunctionEntry* lum_function_tables[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_RGBF] = (const LumFunctionEntry*) 0,      [LUM_BUILTIN_TYPE_VEC3] = (const LumFunctionEntry*) 0,
  [LUM_BUILTIN_TYPE_UINT16] = (const LumFunctionEntry*) 0,    [LUM_BUILTIN_TYPE_UINT32] = (const LumFunctionEntry*) 0,
  [LUM_BUILTIN_TYPE_BOOL] = (const LumFunctionEntry*) 0,      [LUM_BUILTIN_TYPE_FLOAT] = (const LumFunctionEntry*) 0,
  [LUM_BUILTIN_TYPE_ENUM] = (const LumFunctionEntry*) 0,      [LUM_BUILTIN_TYPE_SETTINGS] = lum_function_tables_settings,
  [LUM_BUILTIN_TYPE_CAMERA] = lum_function_tables_camera,     [LUM_BUILTIN_TYPE_OCEAN] = lum_function_tables_ocean,
  [LUM_BUILTIN_TYPE_SKY] = lum_function_tables_sky,           [LUM_BUILTIN_TYPE_CLOUD] = lum_function_tables_cloud,
  [LUM_BUILTIN_TYPE_FOG] = lum_function_tables_fog,           [LUM_BUILTIN_TYPE_PARTICLES] = lum_function_tables_particles,
  [LUM_BUILTIN_TYPE_MATERIAL] = lum_function_tables_material, [LUM_BUILTIN_TYPE_INSTANCE] = lum_function_tables_instance,
  [LUM_BUILTIN_TYPE_METADATA] = lum_function_tables_metadata};

const uint32_t lum_function_tables_count[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_RGBF]      = (uint32_t) 0,
  [LUM_BUILTIN_TYPE_VEC3]      = (uint32_t) 0,
  [LUM_BUILTIN_TYPE_UINT16]    = (uint32_t) 0,
  [LUM_BUILTIN_TYPE_UINT32]    = (uint32_t) 0,
  [LUM_BUILTIN_TYPE_BOOL]      = (uint32_t) 0,
  [LUM_BUILTIN_TYPE_FLOAT]     = (uint32_t) 0,
  [LUM_BUILTIN_TYPE_ENUM]      = (uint32_t) 0,
  [LUM_BUILTIN_TYPE_SETTINGS]  = (uint32_t) sizeof(lum_function_tables_settings) / sizeof(LumFunctionEntry),
  [LUM_BUILTIN_TYPE_CAMERA]    = (uint32_t) sizeof(lum_function_tables_camera) / sizeof(LumFunctionEntry),
  [LUM_BUILTIN_TYPE_OCEAN]     = (uint32_t) sizeof(lum_function_tables_ocean) / sizeof(LumFunctionEntry),
  [LUM_BUILTIN_TYPE_SKY]       = (uint32_t) sizeof(lum_function_tables_sky) / sizeof(LumFunctionEntry),
  [LUM_BUILTIN_TYPE_CLOUD]     = (uint32_t) sizeof(lum_function_tables_cloud) / sizeof(LumFunctionEntry),
  [LUM_BUILTIN_TYPE_FOG]       = (uint32_t) sizeof(lum_function_tables_fog) / sizeof(LumFunctionEntry),
  [LUM_BUILTIN_TYPE_PARTICLES] = (uint32_t) sizeof(lum_function_tables_particles) / sizeof(LumFunctionEntry),
  [LUM_BUILTIN_TYPE_MATERIAL]  = (uint32_t) sizeof(lum_function_tables_material) / sizeof(LumFunctionEntry),
  [LUM_BUILTIN_TYPE_INSTANCE]  = (uint32_t) sizeof(lum_function_tables_instance) / sizeof(LumFunctionEntry),
  [LUM_BUILTIN_TYPE_METADATA]  = (uint32_t) sizeof(lum_function_tables_metadata) / sizeof(LumFunctionEntry)};
