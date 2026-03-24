#include "lum_wavefront.h"

#include "internal_error.h"

static LuminaryResult _wavefront_content_get_versioned_materials_v1(
  WavefrontContent* content, ARRAYPTR LumBuiltinMaterial** materials, Dictionary* material_name_dict, uint32_t texture_offset) {
  __CHECK_NULL_ARGUMENT(content);
  __CHECK_NULL_ARGUMENT(materials);
  __CHECK_NULL_ARGUMENT(material_name_dict);

  uint32_t material_count;
  __FAILURE_HANDLE(array_get_num_elements(content->materials, &material_count));

  uint32_t material_id_offset;
  __FAILURE_HANDLE(array_get_num_elements(*materials, &material_id_offset));

  for (uint32_t mat_id = 0; mat_id < material_count; mat_id++) {
    const WavefrontMaterial wavefront_mat = content->materials[mat_id];

    const bool has_albedo_tex    = (wavefront_mat.texture[WF_ALBEDO] != TEXTURE_NONE);
    const bool has_luminance_tex = (wavefront_mat.texture[WF_LUMINANCE] != TEXTURE_NONE);
    const bool has_roughness_tex = (wavefront_mat.texture[WF_ROUGHNESS] != TEXTURE_NONE);
    const bool has_metallic_tex  = (wavefront_mat.texture[WF_METALLIC] != TEXTURE_NONE);
    const bool has_normal_tex    = (wavefront_mat.texture[WF_NORMAL] != TEXTURE_NONE);
    const bool has_emission = (wavefront_mat.emission.r > 0.0f) || (wavefront_mat.emission.g > 0.0f) || (wavefront_mat.emission.b > 0.0f);

    LumBuiltinMaterial mat;
    __FAILURE_HANDLE(lum_builtin_material_init(&mat, 1));

    mat.base_substrate           = LUMINARY_MATERIAL_BASE_SUBSTRATE_OPAQUE;
    mat.albedo.r                 = wavefront_mat.diffuse_reflectivity.r;
    mat.albedo.g                 = wavefront_mat.diffuse_reflectivity.g;
    mat.albedo.b                 = wavefront_mat.diffuse_reflectivity.b;
    mat.opacity                  = wavefront_mat.dissolve;
    mat.emission                 = wavefront_mat.emission;
    mat.emission_scale           = content->args->emission_scale;
    mat.refraction_index         = wavefront_mat.refraction_index;
    mat.roughness                = 1.0f - wavefront_mat.specular_exponent / 1000.0f;
    mat.roughness_clamp          = 0.25f;
    mat.roughness_as_smoothness  = content->args->legacy_smoothness;
    mat.emission_active          = has_luminance_tex || has_emission;
    mat.thin_walled              = false;
    mat.normal_map_is_compressed = true;
    mat.bidirectional_emission   = content->args->force_bidirectional_emission;
    mat.metallic                 = wavefront_mat.specular_reflectivity.r > 0.5f;
    mat.albedo_tex.id            = has_albedo_tex ? texture_offset + wavefront_mat.texture[WF_ALBEDO] : TEXTURE_NONE;
    mat.luminance_tex.id         = has_luminance_tex ? texture_offset + wavefront_mat.texture[WF_LUMINANCE] : TEXTURE_NONE;
    mat.roughness_tex.id         = has_roughness_tex ? texture_offset + wavefront_mat.texture[WF_ROUGHNESS] : TEXTURE_NONE;
    mat.metallic_tex.id          = has_metallic_tex ? texture_offset + wavefront_mat.texture[WF_METALLIC] : TEXTURE_NONE;
    mat.normal_tex.id            = has_normal_tex ? texture_offset + wavefront_mat.texture[WF_NORMAL] : TEXTURE_NONE;

    __FAILURE_HANDLE(array_push(materials, &mat));

    __FAILURE_HANDLE(dictionary_add_entry(material_name_dict, material_id_offset + mat_id, content->material_names[mat_id]));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult wavefront_content_get_versioned_materials(
  WavefrontContent* content, ARRAYPTR LumBuiltinMaterial** materials, Dictionary* material_name_dict, uint32_t texture_offset,
  uint32_t version) {
  __CHECK_NULL_ARGUMENT(content);
  __CHECK_NULL_ARGUMENT(materials);
  __CHECK_NULL_ARGUMENT(material_name_dict);

  if (content->state != WAVEFRONT_CONTENT_STATE_READY_TO_CONVERT) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Wavefront content was in an illegal state.");
  }

  if (version < 1)
    return LUMINARY_SUCCESS;

  if (version == 1) {
    __FAILURE_HANDLE(_wavefront_content_get_versioned_materials_v1(content, materials, material_name_dict, texture_offset));

    return LUMINARY_SUCCESS;
  }

  return LUMINARY_SUCCESS;
}
