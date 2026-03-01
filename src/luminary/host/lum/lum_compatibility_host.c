#include "lum_compatibility_host.h"

#include <string.h>

#include "host/internal_host.h"
#include "internal_error.h"

LuminaryResult lum_compatibility_host_create(LumCompatibilityHost** host) {
  __CHECK_NULL_ARGUMENT(host);

  __FAILURE_HANDLE(host_malloc(host, sizeof(LumCompatibilityHost)));
  memset(*host, 0, sizeof(LumCompatibilityHost));

  __FAILURE_HANDLE(array_create(&(*host)->meshes, sizeof(Mesh*), 16));
  __FAILURE_HANDLE(array_create(&(*host)->materials, sizeof(LumBuiltinMaterial), 16));
  __FAILURE_HANDLE(array_create(&(*host)->mesh_instances, sizeof(LumBuiltinInstance), 16));
  __FAILURE_HANDLE(array_create(&(*host)->textures, sizeof(Texture*), 16));
  __FAILURE_HANDLE(dictionary_create(&(*host)->mesh_instance_name_dict));
  __FAILURE_HANDLE(dictionary_create(&(*host)->material_name_dict));
  __FAILURE_HANDLE(dictionary_create(&(*host)->mesh_name_dict));
  __FAILURE_HANDLE(dictionary_create(&(*host)->texture_name_dict));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_compatibility_host_init(LumCompatibilityHost* host, uint32_t version) {
  __CHECK_NULL_ARGUMENT(host);

  if (version < 1)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Version 0 is not valid.");

  if (version > LUM_VERSION_CURRENT)
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Version %u is not supported by this version of Luminary.", version);

  __FAILURE_HANDLE(lum_builtin_settings_init(&host->settings, version));
  __FAILURE_HANDLE(lum_builtin_camera_init(&host->camera, version));
  __FAILURE_HANDLE(lum_builtin_ocean_init(&host->ocean, version));
  __FAILURE_HANDLE(lum_builtin_sky_init(&host->sky, version));
  __FAILURE_HANDLE(lum_builtin_cloud_init(&host->cloud, version));
  __FAILURE_HANDLE(lum_builtin_fog_init(&host->fog, version));
  __FAILURE_HANDLE(lum_builtin_particles_init(&host->particles, version));

  host->version = version;

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_compatibility_host_apply(LumCompatibilityHost* host, LuminaryHost* dst_host) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(dst_host);

  LuminaryRendererSettings settings;
  __FAILURE_HANDLE(lum_builtin_settings_convert(&host->settings, &settings, host->version));
  __FAILURE_HANDLE(luminary_host_set_settings(dst_host, &settings));

  LuminaryCamera camera;
  __FAILURE_HANDLE(lum_builtin_camera_convert(&host->camera, &camera, host->version));
  __FAILURE_HANDLE(luminary_host_set_camera(dst_host, &camera));

  LuminaryOcean ocean;
  __FAILURE_HANDLE(lum_builtin_ocean_convert(&host->ocean, &ocean, host->version));
  __FAILURE_HANDLE(luminary_host_set_ocean(dst_host, &ocean));

  LuminarySky sky;
  __FAILURE_HANDLE(lum_builtin_sky_convert(&host->sky, &sky, host->version));
  __FAILURE_HANDLE(luminary_host_set_sky(dst_host, &sky));

  LuminaryCloud cloud;
  __FAILURE_HANDLE(lum_builtin_cloud_convert(&host->cloud, &cloud, host->version));
  __FAILURE_HANDLE(luminary_host_set_cloud(dst_host, &cloud));

  LuminaryFog fog;
  __FAILURE_HANDLE(lum_builtin_fog_convert(&host->fog, &fog, host->version));
  __FAILURE_HANDLE(luminary_host_set_fog(dst_host, &fog));

  LuminaryParticles particles;
  __FAILURE_HANDLE(lum_builtin_particles_convert(&host->particles, &particles, host->version));
  __FAILURE_HANDLE(luminary_host_set_particles(dst_host, &particles));

  // TODO: Use API
  __FAILURE_HANDLE(array_append(&dst_host->meshes, host->meshes));
  __FAILURE_HANDLE(array_append(&dst_host->textures, host->textures));

  uint32_t material_count;
  __FAILURE_HANDLE(array_get_num_elements(host->materials, &material_count));

  for (uint32_t material_id = 0; material_id < material_count; material_id++) {
    LuminaryMaterial material;
    __FAILURE_HANDLE(lum_builtin_material_convert(host->materials + material_id, &material, host->version));

    bool material_name_found;
    const char* material_name;
    __FAILURE_HANDLE(dictionary_find_by_id(host->material_name_dict, material_id, &material_name, &material_name_found));

    if (material_name_found == false)
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Material name is missing.");

    uint16_t host_material_id;
    __FAILURE_HANDLE(luminary_host_get_material_from_name(dst_host, material_name, &host_material_id));

    if (host_material_id == MATERIAL_ID_INVALID)
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Luminary host did not return a valid material ID.");

    __FAILURE_HANDLE(luminary_host_set_material(dst_host, host_material_id, &material));
  }

  uint32_t instance_count;
  __FAILURE_HANDLE(array_get_num_elements(host->mesh_instances, &instance_count));

  for (uint32_t instance_id = 0; instance_id < instance_count; instance_id++) {
    LuminaryInstance instance;
    __FAILURE_HANDLE(lum_builtin_instance_convert(host->mesh_instances + instance_id, &instance, host->version));

    bool instance_name_found;
    const char* instance_name;
    __FAILURE_HANDLE(dictionary_find_by_id(host->mesh_instance_name_dict, instance_id, &instance_name, &instance_name_found));

    if (instance_name_found == false)
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Instance name is missing.");

    uint32_t host_instance_id;
    __FAILURE_HANDLE(luminary_host_get_instance_from_name(dst_host, instance_name, &host_instance_id));

    if (host_instance_id == INSTANCE_ID_INVALID)
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Luminary host did not return a valid instance ID.");

    __FAILURE_HANDLE(luminary_host_set_instance(dst_host, host_instance_id, &instance));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_compatibility_host_destroy(LumCompatibilityHost** host) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(*host);

  __FAILURE_HANDLE(array_destroy(&(*host)->meshes));
  __FAILURE_HANDLE(array_destroy(&(*host)->materials));
  __FAILURE_HANDLE(array_destroy(&(*host)->mesh_instances));
  __FAILURE_HANDLE(array_destroy(&(*host)->textures));
  __FAILURE_HANDLE(dictionary_destroy(&(*host)->mesh_instance_name_dict));
  __FAILURE_HANDLE(dictionary_destroy(&(*host)->material_name_dict));
  __FAILURE_HANDLE(dictionary_destroy(&(*host)->mesh_name_dict));
  __FAILURE_HANDLE(dictionary_destroy(&(*host)->texture_name_dict));

  __FAILURE_HANDLE(host_free(host));

  return LUMINARY_SUCCESS;
}
