#include "material.h"

#include "internal_error.h"

LuminaryResult material_get_default(Material* material) {
  __CHECK_NULL_ARGUMENT(material);

  material->id                    = 0;
  material->albedo                = (RGBAF){.r = 0.9f, .g = 0.9f, .b = 0.9f, .a = 0.9f};
  material->emission              = (RGBF){.r = 0.0f, .g = 0.0f, .b = 0.0f};
  material->emission_scale        = 1.0f;
  material->metallic              = 0.0f;
  material->roughness             = 0.7f;
  material->roughness_clamp       = 0.25f;
  material->refraction_index      = 1.0f;
  material->albedo_tex            = TEXTURE_NONE;
  material->luminance_tex         = TEXTURE_NONE;
  material->material_tex          = TEXTURE_NONE;
  material->normal_tex            = TEXTURE_NONE;
  material->flags.emission_active = 1;

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

  if (input->material_tex == TEXTURE_NONE) {
    __MATERIAL_DIRTY(metallic);
    __MATERIAL_DIRTY(roughness);
  }
  else {
    __MATERIAL_DIRTY(material_tex);
  }

  __MATERIAL_DIRTY(roughness_clamp);
  __MATERIAL_DIRTY(refraction_index);
  __MATERIAL_DIRTY(flags.emission_active);

  return LUMINARY_SUCCESS;
}
