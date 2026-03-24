#ifndef LUMINARY_LUM_WAVEFRONT_H
#define LUMINARY_LUM_WAVEFRONT_H

#include "host/wavefront.h"
#include "lum_builtins.h"
#include "utils.h"

LuminaryResult wavefront_content_get_versioned_materials(
  WavefrontContent* content, ARRAYPTR LumBuiltinMaterial** materials, Dictionary* material_name_dict, uint32_t texture_offset,
  uint32_t version);

#endif /* LUMINARY_LUM_WAVEFRONT_H */
