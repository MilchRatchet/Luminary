#include "lum_file_content.h"

#include "camera.h"
#include "cloud.h"
#include "fog.h"
#include "host/internal_host.h"
#include "internal_error.h"
#include "internal_path.h"
#include "mesh.h"
#include "ocean.h"
#include "particles.h"
#include "settings.h"
#include "sky.h"

LuminaryResult lum_file_content_create(LumFileContent** _content) {
  __CHECK_NULL_ARGUMENT(_content);

  LumFileContent* content;
  __FAILURE_HANDLE(host_malloc(&content, sizeof(LumFileContent)));

  __FAILURE_HANDLE(array_create(&content->obj_file_path_strings, sizeof(char*), 16));

  __FAILURE_HANDLE(settings_get_default(&content->settings));
  __FAILURE_HANDLE(camera_get_default(&content->camera));
  __FAILURE_HANDLE(ocean_get_default(&content->ocean));
  __FAILURE_HANDLE(sky_get_default(&content->sky));
  __FAILURE_HANDLE(cloud_get_default(&content->cloud));
  __FAILURE_HANDLE(fog_get_default(&content->fog));
  __FAILURE_HANDLE(particles_get_default(&content->particles));

  __FAILURE_HANDLE(wavefront_arguments_get_default(&content->wavefront_args));

  __FAILURE_HANDLE(array_create(&content->instances, sizeof(MeshInstance), 16));

  *_content = content;

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_file_content_apply(LumFileContent* content, LuminaryHost* host, const Path* base_path) {
  __CHECK_NULL_ARGUMENT(content);
  __CHECK_NULL_ARGUMENT(host);

  ////////////////////////////////////////////////////////////////////
  // Load meshes
  ////////////////////////////////////////////////////////////////////

  uint32_t mesh_id_offset;
  __FAILURE_HANDLE(array_get_num_elements(host->meshes, &mesh_id_offset));

  uint32_t num_obj_files_to_load;
  __FAILURE_HANDLE(array_get_num_elements(content->obj_file_path_strings, &num_obj_files_to_load));

  for (uint32_t obj_file_id = 0; obj_file_id < num_obj_files_to_load; obj_file_id++) {
    Path* obj_path;
    __FAILURE_HANDLE(path_extend(&obj_path, base_path, content->obj_file_path_strings[obj_file_id]));

    __FAILURE_HANDLE(host_queue_load_obj_file(host, obj_path, &content->wavefront_args));

    __FAILURE_HANDLE(luminary_path_destroy(&obj_path));
  }

  ////////////////////////////////////////////////////////////////////
  // Add instances
  ////////////////////////////////////////////////////////////////////

  uint32_t num_instances_added;
  __FAILURE_HANDLE(array_get_num_elements(content->instances, &num_instances_added));

  for (uint32_t instance_id = 0; instance_id < num_instances_added; instance_id++) {
    MeshInstance instance = content->instances[instance_id];

    // Account for any meshes that were loaded prior to loading this lum file.
    instance.mesh_id += mesh_id_offset;

    __FAILURE_HANDLE(scene_add_entry(host->scene_caller, &instance, SCENE_ENTITY_INSTANCES));

    // We have added an instance, so the scene is dirty and we need to queue the propagation
    __FAILURE_HANDLE(host_update_scene(host));
  }

  ////////////////////////////////////////////////////////////////////
  // Update global scene entities
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(luminary_host_set_settings(host, &content->settings));
  __FAILURE_HANDLE(luminary_host_set_camera(host, &content->camera));
  __FAILURE_HANDLE(luminary_host_set_ocean(host, &content->ocean));
  __FAILURE_HANDLE(luminary_host_set_sky(host, &content->sky));
  __FAILURE_HANDLE(luminary_host_set_cloud(host, &content->cloud));
  __FAILURE_HANDLE(luminary_host_set_fog(host, &content->fog));
  __FAILURE_HANDLE(luminary_host_set_particles(host, &content->particles));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_file_content_destroy(LumFileContent** content) {
  __CHECK_NULL_ARGUMENT(content);
  __CHECK_NULL_ARGUMENT(*content);

  uint32_t num_obj_file_path_strings;
  __FAILURE_HANDLE(array_get_num_elements((*content)->obj_file_path_strings, &num_obj_file_path_strings));

  for (uint32_t i = 0; i < num_obj_file_path_strings; i++) {
    __FAILURE_HANDLE(host_free(&(*content)->obj_file_path_strings[i]));
  }

  __FAILURE_HANDLE(array_destroy(&(*content)->obj_file_path_strings));
  __FAILURE_HANDLE(array_destroy(&(*content)->instances));

  __FAILURE_HANDLE(host_free(content));

  return LUMINARY_SUCCESS;
}
