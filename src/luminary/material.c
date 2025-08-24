#include "material.h"

#include "internal_error.h"

LuminaryResult material_get_default(Material* material) {
  __CHECK_NULL_ARGUMENT(material);

  material->id                       = 0;
  material->base_substrate           = LUMINARY_MATERIAL_BASE_SUBSTRATE_OPAQUE;
  material->albedo                   = (RGBAF) {.r = 0.9f, .g = 0.9f, .b = 0.9f, .a = 0.9f};
  material->emission                 = (RGBF) {.r = 0.0f, .g = 0.0f, .b = 0.0f};
  material->emission_scale           = 1.0f;
  material->roughness                = 0.7f;
  material->roughness_clamp          = 0.25f;
  material->refraction_index         = 1.0f;
  material->emission_active          = false;
  material->thin_walled              = false;
  material->metallic                 = false;
  material->colored_transparency     = false;
  material->normal_map_is_compressed = true;
  material->albedo_tex               = TEXTURE_NONE;
  material->luminance_tex            = TEXTURE_NONE;
  material->roughness_tex            = TEXTURE_NONE;
  material->metallic_tex             = TEXTURE_NONE;
  material->normal_tex               = TEXTURE_NONE;

  return LUMINARY_SUCCESS;
}

#define __MATERIAL_DIRTY(var)     \
  {                               \
    if (input->var != old->var) { \
      *dirty = true;              \
      return LUMINARY_SUCCESS;    \
    }                             \
  }

LuminaryResult material_check_for_dirty(const Material* input, const Material* old, bool* dirty) {
  __CHECK_NULL_ARGUMENT(input);
  __CHECK_NULL_ARGUMENT(old);
  __CHECK_NULL_ARGUMENT(dirty);

  *dirty = false;

  __MATERIAL_DIRTY(base_substrate);

  if (input->albedo_tex == TEXTURE_NONE) {
    __MATERIAL_DIRTY(albedo.r);
    __MATERIAL_DIRTY(albedo.g);
    __MATERIAL_DIRTY(albedo.b);
    __MATERIAL_DIRTY(albedo.a);
  }
  else {
    __MATERIAL_DIRTY(albedo_tex);
  }

  if (input->luminance_tex == TEXTURE_NONE) {
    __MATERIAL_DIRTY(emission.r);
    __MATERIAL_DIRTY(emission.g);
    __MATERIAL_DIRTY(emission.b);
  }
  else {
    __MATERIAL_DIRTY(luminance_tex);
    __MATERIAL_DIRTY(emission_scale);
  }

  if (input->roughness_tex == TEXTURE_NONE) {
    __MATERIAL_DIRTY(roughness);
  }
  else {
    __MATERIAL_DIRTY(roughness_tex);
  }

  switch (input->base_substrate) {
    case LUMINARY_MATERIAL_BASE_SUBSTRATE_OPAQUE:
      if (input->metallic_tex == TEXTURE_NONE) {
        __MATERIAL_DIRTY(metallic);
      }
      else {
        __MATERIAL_DIRTY(metallic_tex);
      }
      break;
    case LUMINARY_MATERIAL_BASE_SUBSTRATE_TRANSLUCENT:
      __MATERIAL_DIRTY(refraction_index);
      __MATERIAL_DIRTY(thin_walled);
      break;
    default:
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Invalid base substrate.");
  }

  __MATERIAL_DIRTY(roughness_clamp);
  __MATERIAL_DIRTY(emission_active);
  __MATERIAL_DIRTY(colored_transparency);
  __MATERIAL_DIRTY(roughness_as_smoothness);
  __MATERIAL_DIRTY(normal_map_is_compressed);

  return LUMINARY_SUCCESS;
}
