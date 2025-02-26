#include "lum_function_tables.h"

static LumFunctionEntry lum_function_tables_settings[] = {{.name = "init"}};

static LumFunctionEntry lum_function_tables_camera[] = {{.name = "init"}};

static LumFunctionEntry lum_function_tables_ocean[] = {{.name = "init"}};

static LumFunctionEntry lum_function_tables_sky[] = {{.name = "init"}};

static LumFunctionEntry lum_function_tables_cloud[] = {{.name = "init"}};

static LumFunctionEntry lum_function_tables_fog[] = {{.name = "init"}};

static LumFunctionEntry lum_function_tables_particles[] = {{.name = "init"}};

static LumFunctionEntry lum_function_tables_material[] = {{.name = "init"}};

static LumFunctionEntry lum_function_tables_instance[] = {{.name = "init"}};

LumFunctionEntry* lum_function_tables[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_RGBF]      = (LumFunctionEntry*) 0,
  [LUM_BUILTIN_TYPE_VEC3]      = (LumFunctionEntry*) 0,
  [LUM_BUILTIN_TYPE_UINT16]    = (LumFunctionEntry*) 0,
  [LUM_BUILTIN_TYPE_UINT32]    = (LumFunctionEntry*) 0,
  [LUM_BUILTIN_TYPE_BOOL]      = (LumFunctionEntry*) 0,
  [LUM_BUILTIN_TYPE_FLOAT]     = (LumFunctionEntry*) 0,
  [LUM_BUILTIN_TYPE_ENUM]      = (LumFunctionEntry*) 0,
  [LUM_BUILTIN_TYPE_SETTINGS]  = lum_function_tables_settings,
  [LUM_BUILTIN_TYPE_CAMERA]    = lum_function_tables_camera,
  [LUM_BUILTIN_TYPE_OCEAN]     = lum_function_tables_ocean,
  [LUM_BUILTIN_TYPE_SKY]       = lum_function_tables_sky,
  [LUM_BUILTIN_TYPE_CLOUD]     = lum_function_tables_cloud,
  [LUM_BUILTIN_TYPE_FOG]       = lum_function_tables_fog,
  [LUM_BUILTIN_TYPE_PARTICLES] = lum_function_tables_particles,
  [LUM_BUILTIN_TYPE_MATERIAL]  = lum_function_tables_material,
  [LUM_BUILTIN_TYPE_INSTANCE]  = lum_function_tables_instance};

uint32_t lum_function_tables_count[LUM_BUILTIN_TYPE_COUNT] = {
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
  [LUM_BUILTIN_TYPE_INSTANCE]  = (uint32_t) sizeof(lum_function_tables_instance) / sizeof(LumFunctionEntry)};
