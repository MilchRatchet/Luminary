#ifndef LUMINARY_LUM_BUILTINS_H
#define LUMINARY_LUM_BUILTINS_H

#include "utils.h"

enum LumBuiltinType {
  // Version 1
  LUM_BUILTIN_TYPE_VOID,
  LUM_BUILTIN_TYPE_RGBF,
  LUM_BUILTIN_TYPE_VEC3,
  LUM_BUILTIN_TYPE_UINT,
  LUM_BUILTIN_TYPE_BOOL,
  LUM_BUILTIN_TYPE_FLOAT,
  LUM_BUILTIN_TYPE_ENUM,
  LUM_BUILTIN_TYPE_SETTINGS,
  LUM_BUILTIN_TYPE_CAMERA,
  LUM_BUILTIN_TYPE_OCEAN,
  LUM_BUILTIN_TYPE_SKY,
  LUM_BUILTIN_TYPE_CLOUD,
  LUM_BUILTIN_TYPE_FOG,
  LUM_BUILTIN_TYPE_PARTICLES,
  LUM_BUILTIN_TYPE_MATERIAL,
  LUM_BUILTIN_TYPE_INSTANCE,
  LUM_BUILTIN_TYPE_STRING,
  LUM_BUILTIN_TYPE_LUMINARY,
  LUM_BUILTIN_TYPE_WAVEFRONTOBJFILE,
  LUM_BUILTIN_TYPE_ADAPTIVESAMPLING,
  LUM_BUILTIN_TYPE_COUNT_VERSION_1,

  LUM_BUILTIN_TYPE_COUNT = LUM_BUILTIN_TYPE_COUNT_VERSION_1
} typedef LumBuiltinType;

extern const char* lum_builtin_types_strings[LUM_BUILTIN_TYPE_COUNT];
extern const size_t lum_builtin_types_sizes[LUM_BUILTIN_TYPE_COUNT];
extern const char* lum_builtin_types_mnemonic[LUM_BUILTIN_TYPE_COUNT];
extern const bool lum_builtin_types_addressable[LUM_BUILTIN_TYPE_COUNT];

#define LUM_BUILTIN_ENUM_COUNT                                                                            \
  (LUMINARY_SHADING_MODE_COUNT + LUMINARY_FILTER_COUNT + LUMINARY_TONEMAP_COUNT + LUMINARY_APERTURE_COUNT \
   + LUMINARY_JERLOV_WATER_TYPE_COUNT + LUMINARY_SKY_MODE_COUNT + LUMINARY_MATERIAL_BASE_SUBSTRATE_COUNT)

struct LumBuiltinEnumValuePair {
  const char* string;
  uint32_t value;
} typedef LumBuiltinEnumValuePair;

extern const LumBuiltinEnumValuePair lum_builtin_enums[LUM_BUILTIN_ENUM_COUNT];

struct LumBuiltinTypeMember {
  LumBuiltinType type;
  size_t offset;
  const char* name;
} typedef LumBuiltinTypeMember;

extern const uint32_t lum_builtin_types_member_counts[LUM_BUILTIN_TYPE_COUNT];
extern const LumBuiltinTypeMember* lum_builtin_types_member[LUM_BUILTIN_TYPE_COUNT];

#endif /* LUMINARY_LUM_BUILTINS_H */
