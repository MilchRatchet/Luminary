#include "lum_function_tables.h"

#include "lum_function_implementations.h"

const LumFunction* lum_function_tables_ldg[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_RGBF]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_VEC3]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_UINT]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_BOOL]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_FLOAT]            = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_ENUM]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_SETTINGS]         = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_CAMERA]           = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_OCEAN]            = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_SKY]              = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_CLOUD]            = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_FOG]              = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_PARTICLES]        = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_MATERIAL]         = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_INSTANCE]         = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_LUMINARY]         = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_FILE]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_VERSIONCONTROL]   = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = (const LumFunction*) 0};

const LumFunction* lum_function_tables_stg[LUM_BUILTIN_TYPE_COUNT] = {
  [LUM_BUILTIN_TYPE_RGBF]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_VEC3]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_UINT]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_BOOL]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_FLOAT]            = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_ENUM]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_SETTINGS]         = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_CAMERA]           = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_OCEAN]            = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_SKY]              = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_CLOUD]            = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_FOG]              = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_PARTICLES]        = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_MATERIAL]         = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_INSTANCE]         = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_LUMINARY]         = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_FILE]             = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_VERSIONCONTROL]   = (const LumFunction*) 0,
  [LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE] = (const LumFunction*) 0};
