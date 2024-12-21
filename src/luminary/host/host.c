#include <luminary/host.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

#include "camera.h"
#include "internal_error.h"
#include "internal_host.h"
#include "internal_path.h"
#include "lum.h"
#include "wavefront.h"

#define HOST_RINGBUFFER_SIZE (0x100000ull)
#define HOST_QUEUE_SIZE (0x400ull)

////////////////////////////////////////////////////////////////////
// Queue worker functions
////////////////////////////////////////////////////////////////////

LuminaryResult _host_queue_worker(Host* host) {
  __CHECK_NULL_ARGUMENT(host);

  bool success = true;

  while (success) {
    QueueEntry entry;
    __FAILURE_HANDLE(queue_pop_blocking(host->work_queue, &entry, &success));

    // Verify that the device_manager didn't crash.
    if (host->device_manager) {
      __FAILURE_HANDLE(thread_get_last_result(host->device_manager->work_thread));
    }

    if (!success)
      break;

    __FAILURE_HANDLE(wall_time_set_string(host->queue_wall_time, entry.name));
    __FAILURE_HANDLE(wall_time_start(host->queue_wall_time));

    __FAILURE_HANDLE(entry.function(host, entry.args));

    if (entry.clear_func) {
      __FAILURE_HANDLE(entry.clear_func(host, entry.args));
    }

    __FAILURE_HANDLE(wall_time_stop(host->queue_wall_time));
    __FAILURE_HANDLE(wall_time_set_string(host->queue_wall_time, (const char*) 0));

#ifdef LUMINARY_WORK_QUEUE_STATS_PRINT
    double time;
    __FAILURE_HANDLE(wall_time_get_time(host->queue_wall_time, &time));

    if (time > LUMINARY_WORK_QUEUE_STATS_PRINT_THRESHOLD) {
      warn_message("host queue: %s (%fs)", entry.name, time);
    }
#endif
  }

  return LUMINARY_SUCCESS;
}

static bool _host_queue_entry_equal_operator(QueueEntry* left, QueueEntry* right) {
  return (left->function == right->function);
}

////////////////////////////////////////////////////////////////////
// Queue work functions
////////////////////////////////////////////////////////////////////

struct HostLoadObjArgs {
  Path* path;
} typedef HostLoadObjArgs;

static LuminaryResult _host_load_obj_file(Host* host, HostLoadObjArgs* args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(args);

  WavefrontContent* wavefront_content;

  __FAILURE_HANDLE(wavefront_create(&wavefront_content));
  __FAILURE_HANDLE(wavefront_read_file(wavefront_content, args->path));

  // TODO: Lum v5 files contain materials already, figure out how to handle materials coming from mtl files then.

  uint32_t num_meshes_before;
  __FAILURE_HANDLE(array_get_num_elements(host->meshes, &num_meshes_before));

  uint32_t num_textures_before;
  __FAILURE_HANDLE(array_get_num_elements(host->textures, &num_textures_before));

  // Lock the scene lists because we need to freeze the current material count and then add all the new materials.
  // This can cause the caller to stall if he performs other list updates.
  __FAILURE_HANDLE_LOCK_CRITICAL();
  __FAILURE_HANDLE_CRITICAL(scene_lock(host->scene_caller, SCENE_ENTITY_TYPE_LIST));

  ARRAY Material* added_materials;
  __FAILURE_HANDLE_CRITICAL(array_create(&added_materials, sizeof(Material), 16));

  uint32_t material_offset;
  __FAILURE_HANDLE_CRITICAL(scene_get_entry_count(host->scene_caller, SCENE_ENTITY_MATERIALS, &material_offset));

  __FAILURE_HANDLE_CRITICAL(
    wavefront_convert_content(wavefront_content, &host->meshes, &host->textures, &added_materials, material_offset));

  uint32_t num_added_materials;
  __FAILURE_HANDLE_CRITICAL(array_get_num_elements(added_materials, &num_added_materials));

  for (uint32_t new_material_id = 0; new_material_id < num_added_materials; new_material_id++) {
    __FAILURE_HANDLE_CRITICAL(scene_add_entry(host->scene_caller, added_materials + new_material_id, SCENE_ENTITY_MATERIALS));
  }

  __FAILURE_HANDLE_UNLOCK_CRITICAL();
  __FAILURE_HANDLE(scene_unlock(host->scene_caller, SCENE_ENTITY_TYPE_LIST));

  __FAILURE_HANDLE_CHECK_CRITICAL();

  __FAILURE_HANDLE(array_destroy(&added_materials));
  __FAILURE_HANDLE(wavefront_destroy(&wavefront_content));

  __FAILURE_HANDLE(scene_propagate_changes(host->scene_host, host->scene_caller));

  uint32_t num_meshes_after;
  __FAILURE_HANDLE(array_get_num_elements(host->meshes, &num_meshes_after));

  __FAILURE_HANDLE(
    device_manager_add_meshes(host->device_manager, (const Mesh**) host->meshes + num_meshes_before, num_meshes_after - num_meshes_before));

  uint32_t num_textures_after;
  __FAILURE_HANDLE(array_get_num_elements(host->textures, &num_textures_after));

  __FAILURE_HANDLE(device_manager_add_textures(
    host->device_manager, (const Texture**) host->textures + num_textures_before, num_textures_after - num_textures_before));

  // Clean up
  __FAILURE_HANDLE(luminary_path_destroy(&args->path));
  __FAILURE_HANDLE(ringbuffer_release_entry(host->ringbuffer, sizeof(HostLoadObjArgs)));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _host_propagate_scene_changes_queue_work(Host* host, void* args) {
  LUM_UNUSED(args);

  __CHECK_NULL_ARGUMENT(host);

  __FAILURE_HANDLE(scene_propagate_changes(host->scene_host, host->scene_caller));

  __FAILURE_HANDLE(device_manager_update_scene(host->device_manager));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _host_copy_output_queue_work(Host* host, OutputCopyHandle* args) {
  __CHECK_NULL_ARGUMENT(host);

  uint32_t handle;
  __FAILURE_HANDLE(output_handler_acquire_new(host->output_handler, args->width, args->height, &handle));

  void* dst;
  __FAILURE_HANDLE(output_handler_get_buffer(host->output_handler, handle, &dst));

  memcpy(dst, args->src, args->width * args->height * sizeof(ARGB8));

  __FAILURE_HANDLE(output_handler_release_new(host->output_handler, handle));

  // Clean up
  __FAILURE_HANDLE(ringbuffer_release_entry(host->ringbuffer, sizeof(OutputCopyHandle)));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _host_start_render_queue_work(Host* host, void* args) {
  __CHECK_NULL_ARGUMENT(host);

  LUM_UNUSED(args);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Internal implementation
////////////////////////////////////////////////////////////////////

static LuminaryResult _host_update_scene(Host* host) {
  __CHECK_NULL_ARGUMENT(host);

  QueueEntry entry;

  entry.name              = "Updating scene";
  entry.function          = (QueueEntryFunction) _host_propagate_scene_changes_queue_work;
  entry.clear_func        = (QueueEntryFunction) 0;
  entry.args              = (void*) 0;
  entry.remove_duplicates = true;

  // TODO: Abstract this like in the device_manager.
  bool already_queued;
  __FAILURE_HANDLE(queue_push_unique(host->work_queue, &entry, (LuminaryEqOp) _host_queue_entry_equal_operator, &already_queued));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _host_set_scene_entity(Host* host, void* object, SceneEntity entity) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(object);

  bool scene_changed = false;
  __FAILURE_HANDLE(scene_update(host->scene_caller, object, entity, &scene_changed));

  // If there are no changes, skip the propagation to avoid hammering the queue.
  if (scene_changed) {
    __FAILURE_HANDLE(_host_update_scene(host));
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// External API implementation
////////////////////////////////////////////////////////////////////

LuminaryResult luminary_host_create(Host** _host) {
  __CHECK_NULL_ARGUMENT(_host);

  Host* host;
  __FAILURE_HANDLE(host_malloc(&host, sizeof(Host)));
  memset(host, 0, sizeof(Host));

  __FAILURE_HANDLE(output_handler_create(&host->output_handler));

  __FAILURE_HANDLE(array_create(&host->meshes, sizeof(Mesh*), 16));
  __FAILURE_HANDLE(array_create(&host->textures, sizeof(Texture*), 16));

  __FAILURE_HANDLE(scene_create(&host->scene_caller));
  __FAILURE_HANDLE(scene_create(&host->scene_host));

  __FAILURE_HANDLE(thread_create(&host->work_thread));
  __FAILURE_HANDLE(queue_create(&host->work_queue, sizeof(QueueEntry), HOST_QUEUE_SIZE));
  __FAILURE_HANDLE(ringbuffer_create(&host->ringbuffer, HOST_RINGBUFFER_SIZE));
  __FAILURE_HANDLE(wall_time_create(&host->queue_wall_time));

  __FAILURE_HANDLE(device_manager_create(&host->device_manager, host));

  host->enable_output = false;

  __FAILURE_HANDLE(thread_start(host->work_thread, (ThreadMainFunc) _host_queue_worker, host));

  *_host = host;

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_destroy(Host** host) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(*host);

  ////////////////////////////////////////////////////////////////////
  // Shutdown device thread queue
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(device_manager_shutdown_queue((*host)->device_manager));

  ////////////////////////////////////////////////////////////////////
  // Shutdown host thread queue
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(queue_set_is_blocking((*host)->work_queue, false));
  __FAILURE_HANDLE(thread_join((*host)->work_thread));
  __FAILURE_HANDLE(thread_get_last_result((*host)->work_thread));

  ////////////////////////////////////////////////////////////////////
  // Destroy member
  ////////////////////////////////////////////////////////////////////

  __FAILURE_HANDLE(device_manager_destroy(&(*host)->device_manager));

  __FAILURE_HANDLE(wall_time_destroy(&(*host)->queue_wall_time));
  __FAILURE_HANDLE(ringbuffer_destroy(&(*host)->ringbuffer));
  __FAILURE_HANDLE(queue_destroy(&(*host)->work_queue));

  __FAILURE_HANDLE(thread_destroy(&(*host)->work_thread));

  uint32_t mesh_count;
  __FAILURE_HANDLE(array_get_num_elements((*host)->meshes, &mesh_count));

  for (uint32_t mesh_id = 0; mesh_id < mesh_count; mesh_id++) {
    __FAILURE_HANDLE(mesh_destroy(&(*host)->meshes[mesh_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*host)->meshes));

  uint32_t texture_count;
  __FAILURE_HANDLE(array_get_num_elements((*host)->textures, &texture_count));

  for (uint32_t tex_id = 0; tex_id < texture_count; tex_id++) {
    __FAILURE_HANDLE(texture_destroy(&(*host)->textures[tex_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*host)->textures));

  __FAILURE_HANDLE(scene_destroy(&(*host)->scene_caller));
  __FAILURE_HANDLE(scene_destroy(&(*host)->scene_host));

  __FAILURE_HANDLE(output_handler_destroy(&(*host)->output_handler));

  __FAILURE_HANDLE(host_free(host));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_load_obj_file(Host* host, Path* path) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(path);

  HostLoadObjArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(host->ringbuffer, sizeof(HostLoadObjArgs), (void**) &args));

  __FAILURE_HANDLE(path_copy(&args->path, path));

  QueueEntry entry;

  entry.name              = "Loading Obj";
  entry.function          = (QueueEntryFunction) _host_load_obj_file;
  entry.clear_func        = (QueueEntryFunction) 0;
  entry.args              = args;
  entry.remove_duplicates = false;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_load_lum_file(Host* host, Path* path) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(path);

  LumFileContent* content;
  __FAILURE_HANDLE(lum_content_create(&content));

  ////////////////////////////////////////////////////////////////////
  // Read lum file
  ////////////////////////////////////////////////////////////////////

  Path* lum_path;
  __FAILURE_HANDLE(path_copy(&lum_path, path));

  __FAILURE_HANDLE(lum_read_file(lum_path, content));

  __FAILURE_HANDLE(luminary_path_destroy(&lum_path));

  ////////////////////////////////////////////////////////////////////
  // Load meshes
  ////////////////////////////////////////////////////////////////////

  uint32_t mesh_id_offset;
  __FAILURE_HANDLE(array_get_num_elements(host->meshes, &mesh_id_offset));

  uint32_t num_obj_files_to_load;
  __FAILURE_HANDLE(array_get_num_elements(content->obj_file_path_strings, &num_obj_files_to_load));

  for (uint32_t obj_file_id = 0; obj_file_id < num_obj_files_to_load; obj_file_id++) {
    Path* obj_path;
    __FAILURE_HANDLE(path_extend(&obj_path, path, content->obj_file_path_strings[obj_file_id]));

    __FAILURE_HANDLE(luminary_host_load_obj_file(host, obj_path));

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
    __FAILURE_HANDLE(_host_update_scene(host));
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
  __FAILURE_HANDLE(luminary_host_set_toy(host, &content->toy));

  __FAILURE_HANDLE(lum_content_destroy(&content));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_queue_string(const Host* host, const char** string) {
  __CHECK_NULL_ARGUMENT(host);

  if (!string) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "String is NULL.");
  }

  __FAILURE_HANDLE(wall_time_get_string(host->queue_wall_time, string));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_settings(Host* host, RendererSettings* settings) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(settings);

  __FAILURE_HANDLE(scene_get(host->scene_caller, settings, SCENE_ENTITY_SETTINGS));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_settings(Host* host, RendererSettings* settings) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(settings);

  _host_set_scene_entity(host, (void*) settings, SCENE_ENTITY_SETTINGS);

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_camera(Host* host, Camera* camera) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(camera);

  __FAILURE_HANDLE(scene_get(host->scene_caller, camera, SCENE_ENTITY_CAMERA));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_camera(Host* host, Camera* camera) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(camera);

  _host_set_scene_entity(host, (void*) camera, SCENE_ENTITY_CAMERA);

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_ocean(Host* host, Ocean* ocean) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(ocean);

  __FAILURE_HANDLE(scene_get(host->scene_caller, ocean, SCENE_ENTITY_OCEAN));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_ocean(Host* host, Ocean* ocean) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(ocean);

  _host_set_scene_entity(host, (void*) ocean, SCENE_ENTITY_OCEAN);

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_sky(Host* host, Sky* sky) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(sky);

  __FAILURE_HANDLE(scene_get(host->scene_caller, sky, SCENE_ENTITY_SKY));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_sky(Host* host, Sky* sky) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(sky);

  _host_set_scene_entity(host, (void*) sky, SCENE_ENTITY_SKY);

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_cloud(Host* host, Cloud* cloud) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(cloud);

  __FAILURE_HANDLE(scene_get(host->scene_caller, cloud, SCENE_ENTITY_CLOUD));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_cloud(Host* host, Cloud* cloud) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(cloud);

  _host_set_scene_entity(host, (void*) cloud, SCENE_ENTITY_CLOUD);

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_fog(Host* host, Fog* fog) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(fog);

  __FAILURE_HANDLE(scene_get(host->scene_caller, fog, SCENE_ENTITY_FOG));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_fog(Host* host, Fog* fog) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(fog);

  _host_set_scene_entity(host, (void*) fog, SCENE_ENTITY_FOG);

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_particles(Host* host, Particles* particles) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(particles);

  __FAILURE_HANDLE(scene_get(host->scene_caller, particles, SCENE_ENTITY_PARTICLES));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_particles(Host* host, Particles* particles) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(particles);

  _host_set_scene_entity(host, (void*) particles, SCENE_ENTITY_PARTICLES);

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_toy(Host* host, Toy* toy) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(toy);

  __FAILURE_HANDLE(scene_get(host->scene_caller, toy, SCENE_ENTITY_TOY));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_toy(Host* host, Toy* toy) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(toy);

  _host_set_scene_entity(host, (void*) toy, SCENE_ENTITY_TOY);

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_output_properties(LuminaryHost* host, LuminaryOutputProperties properties) {
  __CHECK_NULL_ARGUMENT(host);

  __FAILURE_HANDLE(output_handler_set_properties(host->output_handler, properties));
  __FAILURE_HANDLE(device_manager_set_output_properties(host->device_manager, properties.width, properties.height));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_acquire_output(LuminaryHost* host, LuminaryOutputHandle* output_handle) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(output_handle);

  __FAILURE_HANDLE(output_handler_acquire(host->output_handler, output_handle));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_output_buffer(LuminaryHost* host, LuminaryOutputHandle output_handle, void** output_buffer) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(output_buffer);

  __FAILURE_HANDLE(output_handler_get_buffer(host->output_handler, output_handle, output_buffer));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_release_output(LuminaryHost* host, LuminaryOutputHandle output_handle) {
  __CHECK_NULL_ARGUMENT(host);

  __FAILURE_HANDLE(output_handler_release(host->output_handler, output_handle));

  return LUMINARY_SUCCESS;
}

LuminaryResult host_queue_output_copy_from_device(Host* host, OutputCopyHandle copy_handle) {
  __CHECK_NULL_ARGUMENT(host);

  OutputCopyHandle* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(host->ringbuffer, sizeof(OutputCopyHandle), (void**) &args));

  memcpy(args, &copy_handle, sizeof(OutputCopyHandle));

  QueueEntry entry;

  entry.name              = "Copy Output";
  entry.function          = (QueueEntryFunction) _host_copy_output_queue_work;
  entry.clear_func        = (QueueEntryFunction) 0;
  entry.args              = (void*) args;
  entry.remove_duplicates = false;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}
