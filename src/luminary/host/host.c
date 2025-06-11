#include <assert.h>
#include <luminary/host.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

#include "camera.h"
#include "internal_error.h"
#include "internal_host.h"
#include "internal_path.h"
#include "lum.h"
#include "mesh.h"
#include "png.h"
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
  WavefrontArguments wavefront_args;
} typedef HostLoadObjArgs;

static LuminaryResult _host_load_obj_file(Host* host, HostLoadObjArgs* args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(args);

  WavefrontContent* wavefront_content;

  __FAILURE_HANDLE(wavefront_create(&wavefront_content, args->wavefront_args));
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

static LuminaryResult _host_copy_output_queue_work(Host* host, OutputDescriptor* args) {
  __CHECK_NULL_ARGUMENT(host);

  uint32_t handle;
  if (args->is_recurring_output) {
    __FAILURE_HANDLE(output_handler_acquire_new(host->output_handler, *args, &handle));
  }
  else {
    __FAILURE_HANDLE(output_handler_acquire_from_request_new(host->output_handler, *args, &handle));
  }

  Image dst;
  __FAILURE_HANDLE(output_handler_get_image(host->output_handler, handle, &dst));

  memcpy(dst.buffer, args->data, args->meta_data.width * args->meta_data.height * sizeof(ARGB8));

  __FAILURE_HANDLE(output_handler_release_new(host->output_handler, handle));

  // Clean up
  __FAILURE_HANDLE(ringbuffer_release_entry(host->ringbuffer, sizeof(OutputDescriptor)));

  return LUMINARY_SUCCESS;
}

struct HostAddOutputRequestArgs {
  OutputRequestProperties props;
} typedef HostAddOutputRequestArgs;

static LuminaryResult _host_add_output_request_clear_work(Host* host, HostAddOutputRequestArgs* args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(ringbuffer_release_entry(host->ringbuffer, sizeof(HostAddOutputRequestArgs)));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _host_add_output_request_queue_work(Host* host, HostAddOutputRequestArgs* args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(device_manager_add_output_request(host->device_manager, args->props));

  return LUMINARY_SUCCESS;
}

struct HostSavePNGArgs {
  char* path;
  LuminaryOutputHandle handle;
} typedef HostSavePNGArgs;

static LuminaryResult _host_save_png_clear_work(Host* host, HostSavePNGArgs* args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(output_handler_release(host->output_handler, args->handle));

  __FAILURE_HANDLE(host_free(&args->path));

  __FAILURE_HANDLE(ringbuffer_release_entry(host->ringbuffer, sizeof(HostSavePNGArgs)));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _host_save_png_queue_work(Host* host, HostSavePNGArgs* args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(args);

  LuminaryImage output_image;
  __FAILURE_HANDLE(luminary_host_get_image(host, args->handle, &output_image));

  __FAILURE_HANDLE(png_store_image(output_image, args->path));

  return LUMINARY_SUCCESS;
}

struct HostEnableDeviceArgs {
  bool enable;
  uint32_t device_id;
} typedef HostEnableDeviceArgs;

static LuminaryResult _host_enable_device_clear_work(Host* host, HostEnableDeviceArgs* args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(ringbuffer_release_entry(host->ringbuffer, sizeof(HostEnableDeviceArgs)));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _host_enable_device_queue_work(Host* host, HostEnableDeviceArgs* args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(device_manager_enable_device(host->device_manager, args->device_id, args->enable));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Internal implementation
////////////////////////////////////////////////////////////////////

static LuminaryResult _host_update_scene(Host* host) {
  __CHECK_NULL_ARGUMENT(host);

  QueueEntry entry;
  memset(&entry, 0, sizeof(QueueEntry));

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

static LuminaryResult _host_set_scene_entity_entry(Host* host, void* object, SceneEntity entity, uint32_t id) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(object);

  bool scene_changed = false;
  __FAILURE_HANDLE(scene_update_entry(host->scene_caller, object, entity, id, &scene_changed));

  // If there are no changes, skip the propagation to avoid hammering the queue.
  if (scene_changed) {
    __FAILURE_HANDLE(_host_update_scene(host));
  }

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// External API implementation
////////////////////////////////////////////////////////////////////

LuminaryResult luminary_host_create(Host** host, LuminaryHostCreateInfo info) {
  __CHECK_NULL_ARGUMENT(host);

  __FAILURE_HANDLE(host_malloc(host, sizeof(Host)));
  memset(*host, 0, sizeof(Host));

  __FAILURE_HANDLE(output_handler_create(&(*host)->output_handler));

  __FAILURE_HANDLE(array_create(&(*host)->meshes, sizeof(Mesh*), 16));
  __FAILURE_HANDLE(array_create(&(*host)->textures, sizeof(Texture*), 16));

  __FAILURE_HANDLE(scene_create(&(*host)->scene_caller));
  __FAILURE_HANDLE(scene_create(&(*host)->scene_host));

  __FAILURE_HANDLE(thread_create(&(*host)->work_thread));
  __FAILURE_HANDLE(queue_create(&(*host)->work_queue, sizeof(QueueEntry), HOST_QUEUE_SIZE));
  __FAILURE_HANDLE(ringbuffer_create(&(*host)->ringbuffer, HOST_RINGBUFFER_SIZE));
  __FAILURE_HANDLE(wall_time_create(&(*host)->queue_wall_time));

  DeviceManagerCreateInfo device_manager_create_info;
  device_manager_create_info.device_mask = info.device_mask;

  __FAILURE_HANDLE(device_manager_create(&(*host)->device_manager, *host, device_manager_create_info));

  (*host)->enable_output = false;

  __FAILURE_HANDLE(thread_start((*host)->work_thread, (ThreadMainFunc) _host_queue_worker, *host));

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

LuminaryResult luminary_host_get_device_count(LuminaryHost* host, uint32_t* device_count) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(device_count);

  __FAILURE_HANDLE(array_get_num_elements(host->device_manager->devices, device_count));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_device_info(LuminaryHost* host, uint32_t device_id, LuminaryDeviceInfo* info) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(info);

  uint32_t num_devices;
  __FAILURE_HANDLE(array_get_num_elements(host->device_manager->devices, &num_devices));

  if (device_id >= num_devices) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_DEVICE, "Device ID exceeds number of devices.");
  }

  Device* device = host->device_manager->devices[device_id];

  info->is_unavailable = device->state == DEVICE_STATE_UNAVAILABLE;
  info->is_enabled     = device->state == DEVICE_STATE_ENABLED;

  static_assert(sizeof(info->name) >= 256 && sizeof(device->properties.name) >= 256, "Name buffers are too small.");
  memcpy(info->name, device->properties.name, 256);

  __FAILURE_HANDLE(device_memory_get_total_allocation_size(device->cuda_device, &info->allocated_memory_size));

  info->memory_size = device->properties.memory_size;

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_device_enable(LuminaryHost* host, uint32_t device_id, bool enable) {
  __CHECK_NULL_ARGUMENT(host);

  uint32_t num_devices;
  __FAILURE_HANDLE(array_get_num_elements(host->device_manager->devices, &num_devices));

  if (device_id >= num_devices) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_DEVICE, "Device ID exceeds number of devices.");
  }

  Device* device = host->device_manager->devices[device_id];

  // Device is already in the requested state
  if ((device->state == DEVICE_STATE_ENABLED) == enable)
    return LUMINARY_SUCCESS;

  if (device->state == DEVICE_STATE_UNAVAILABLE) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_API_ARGUMENT, "Device is not available.");
  }

  // TODO: This needs to do the following: Queue a job that sets the state and then sets the Integration dirty flag.
  device->state = (enable) ? DEVICE_STATE_ENABLED : DEVICE_STATE_DISABLED;

  HostEnableDeviceArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(host->ringbuffer, sizeof(HostLoadObjArgs), (void**) &args));

  args->enable    = enable;
  args->device_id = device_id;

  QueueEntry entry;
  memset(&entry, 0, sizeof(QueueEntry));

  entry.name       = "Enabling device";
  entry.function   = (QueueEntryFunction) _host_enable_device_queue_work;
  entry.clear_func = (QueueEntryFunction) _host_enable_device_clear_work;
  entry.args       = args;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _host_queue_load_obj_file(Host* host, Path* path, WavefrontArguments wavefront_args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(path);

  HostLoadObjArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(host->ringbuffer, sizeof(HostLoadObjArgs), (void**) &args));

  __FAILURE_HANDLE(path_copy(&args->path, path));

  args->wavefront_args = wavefront_args;

  QueueEntry entry;
  memset(&entry, 0, sizeof(QueueEntry));

  entry.name       = "Loading Obj";
  entry.function   = (QueueEntryFunction) _host_load_obj_file;
  entry.clear_func = (QueueEntryFunction) 0;
  entry.args       = args;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_load_obj_file(Host* host, Path* path) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(path);

  WavefrontArguments args;
  __FAILURE_HANDLE(wavefront_arguments_get_default(&args));

  __FAILURE_HANDLE(_host_queue_load_obj_file(host, path, args));

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

    __FAILURE_HANDLE(_host_queue_load_obj_file(host, obj_path, content->wavefront_args));

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

  __FAILURE_HANDLE(lum_content_destroy(&content));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_current_sample_time(Host* host, double* time) {
  __CHECK_NULL_ARGUMENT(host);

  __FAILURE_HANDLE(sample_time_get_time(host->device_manager->sample_time, time));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_queue_string(const Host* host, const char** string) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(string);

  __FAILURE_HANDLE(wall_time_get_string(host->queue_wall_time, string));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_device_queue_string(const Host* host, const char** string) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(string);

  __FAILURE_HANDLE(wall_time_get_string(host->device_manager->queue_wall_time, string));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_queue_time(const Host* host, double* time) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(time);

  __FAILURE_HANDLE(wall_time_get_time(host->queue_wall_time, time));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_device_queue_time(const Host* host, double* time) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(time);

  __FAILURE_HANDLE(wall_time_get_time(host->device_manager->queue_wall_time, time));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_settings(Host* host, RendererSettings* settings) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(settings);

  __FAILURE_HANDLE(scene_get(host->scene_caller, settings, SCENE_ENTITY_SETTINGS));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_settings(Host* host, const RendererSettings* settings) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(settings);

  __FAILURE_HANDLE(_host_set_scene_entity(host, (void*) settings, SCENE_ENTITY_SETTINGS));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_camera(Host* host, Camera* camera) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(camera);

  __FAILURE_HANDLE(scene_get(host->scene_caller, camera, SCENE_ENTITY_CAMERA));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_camera(Host* host, const Camera* camera) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(camera);

  __FAILURE_HANDLE(_host_set_scene_entity(host, (void*) camera, SCENE_ENTITY_CAMERA));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_ocean(Host* host, Ocean* ocean) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(ocean);

  __FAILURE_HANDLE(scene_get(host->scene_caller, ocean, SCENE_ENTITY_OCEAN));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_ocean(Host* host, const Ocean* ocean) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(ocean);

  __FAILURE_HANDLE(_host_set_scene_entity(host, (void*) ocean, SCENE_ENTITY_OCEAN));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_sky(Host* host, Sky* sky) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(sky);

  __FAILURE_HANDLE(scene_get(host->scene_caller, sky, SCENE_ENTITY_SKY));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_sky(Host* host, const Sky* sky) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(sky);

  __FAILURE_HANDLE(_host_set_scene_entity(host, (void*) sky, SCENE_ENTITY_SKY));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_cloud(Host* host, Cloud* cloud) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(cloud);

  __FAILURE_HANDLE(scene_get(host->scene_caller, cloud, SCENE_ENTITY_CLOUD));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_cloud(Host* host, const Cloud* cloud) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(cloud);

  __FAILURE_HANDLE(_host_set_scene_entity(host, (void*) cloud, SCENE_ENTITY_CLOUD));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_fog(Host* host, Fog* fog) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(fog);

  __FAILURE_HANDLE(scene_get(host->scene_caller, fog, SCENE_ENTITY_FOG));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_fog(Host* host, const Fog* fog) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(fog);

  __FAILURE_HANDLE(_host_set_scene_entity(host, (void*) fog, SCENE_ENTITY_FOG));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_particles(Host* host, Particles* particles) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(particles);

  __FAILURE_HANDLE(scene_get(host->scene_caller, particles, SCENE_ENTITY_PARTICLES));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_particles(Host* host, const Particles* particles) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(particles);

  __FAILURE_HANDLE(_host_set_scene_entity(host, (void*) particles, SCENE_ENTITY_PARTICLES));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_material(Host* host, uint16_t id, LuminaryMaterial* material) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(material);

  __FAILURE_HANDLE(scene_get_entry(host->scene_caller, material, SCENE_ENTITY_MATERIALS, id));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_material(Host* host, uint16_t id, const LuminaryMaterial* material) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(material);

  __FAILURE_HANDLE(_host_set_scene_entity_entry(host, (void*) material, SCENE_ENTITY_MATERIALS, id));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_instance(Host* host, uint32_t id, LuminaryInstance* instance) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(instance);

  MeshInstance mesh_instance;
  __FAILURE_HANDLE(scene_get_entry(host->scene_caller, &mesh_instance, SCENE_ENTITY_INSTANCES, id));

  __FAILURE_HANDLE(mesh_instance_to_public_api_instance(instance, &mesh_instance));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_instance(Host* host, const LuminaryInstance* instance) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(instance);

  MeshInstance mesh_instance;
  __FAILURE_HANDLE(mesh_instance_from_public_api_instance(&mesh_instance, instance));

  __FAILURE_HANDLE(_host_set_scene_entity_entry(host, (void*) &mesh_instance, SCENE_ENTITY_INSTANCES, mesh_instance.id));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_new_instance(Host* host, LuminaryInstance* instance) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(instance);

  MeshInstance mesh_instance;
  __FAILURE_HANDLE(mesh_instance_get_default(&mesh_instance));

  __FAILURE_HANDLE(scene_add_entry(host->scene_caller, &mesh_instance, SCENE_ENTITY_INSTANCES));
  __FAILURE_HANDLE(_host_update_scene(host));

  __FAILURE_HANDLE(mesh_instance_to_public_api_instance(instance, &mesh_instance));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_num_meshes(Host* host, uint32_t* num_meshes) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(num_meshes);

  __FAILURE_HANDLE(array_get_num_elements(host->meshes, num_meshes));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_num_materials(Host* host, uint32_t* num_materials) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(num_materials);

  __FAILURE_HANDLE(scene_get_entry_count(host->scene_caller, SCENE_ENTITY_MATERIALS, num_materials));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_num_instances(Host* host, uint32_t* num_instances) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(num_instances);

  __FAILURE_HANDLE(scene_get_entry_count(host->scene_caller, SCENE_ENTITY_INSTANCES, num_instances));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_output_properties(Host* host, LuminaryOutputProperties properties) {
  __CHECK_NULL_ARGUMENT(host);

  __FAILURE_HANDLE(output_handler_set_properties(host->output_handler, properties));
  __FAILURE_HANDLE(device_manager_set_output_properties(host->device_manager, properties.width, properties.height));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_request_output(Host* host, LuminaryOutputRequestProperties properties, LuminaryOutputPromiseHandle* handle) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(handle);

  __FAILURE_HANDLE(output_handler_add_request(host->output_handler, properties, handle));

  HostAddOutputRequestArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(host->ringbuffer, sizeof(HostAddOutputRequestArgs), (void**) &args));

  args->props = properties;

  QueueEntry entry;
  memset(&entry, 0, sizeof(QueueEntry));

  entry.name       = "Add Output Request";
  entry.function   = (QueueEntryFunction) _host_add_output_request_queue_work;
  entry.clear_func = (QueueEntryFunction) _host_add_output_request_clear_work;
  entry.args       = (void*) args;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_try_await_output(Host* host, LuminaryOutputPromiseHandle handle, LuminaryOutputHandle* output_handle) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(output_handle);

  __FAILURE_HANDLE(output_handler_acquire_from_promise(host->output_handler, handle, output_handle));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_acquire_output(Host* host, LuminaryOutputHandle* output_handle) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(output_handle);

  __FAILURE_HANDLE(output_handler_acquire_recurring(host->output_handler, output_handle));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_image(LuminaryHost* host, LuminaryOutputHandle output_handle, Image* image) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(image);

  __FAILURE_HANDLE(output_handler_get_image(host->output_handler, output_handle, image));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_release_output(Host* host, LuminaryOutputHandle output_handle) {
  __CHECK_NULL_ARGUMENT(host);

  __FAILURE_HANDLE(output_handler_release(host->output_handler, output_handle));

  return LUMINARY_SUCCESS;
}

static float _host_bfloat16_to_float32(uint16_t val) {
  float float32;

  memset(&float32, 0, sizeof(float));

#if ENDIAN_ORDER == BIG_ENDIAN
  memcpy(&float32, &val, sizeof(uint16_t));
#else
  memcpy(((uint8_t*) (&float32)) + sizeof(float) - sizeof(uint16_t), &val, sizeof(uint16_t));
#endif

  return float32;
}

LuminaryResult luminary_host_get_pixel_info(Host* host, uint16_t x, uint16_t y, LuminaryPixelQueryResult* result) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(result);

  Device* device = host->device_manager->devices[host->device_manager->main_device_index];

  GBufferMetaData meta_data;
  __FAILURE_HANDLE(device_get_gbuffer_meta(device, x, y, &meta_data));

  result->pixel_query_is_valid =
    (meta_data.depth != DEPTH_INVALID) || (meta_data.instance_id != 0xFFFFFFFF) || (meta_data.material_id != MATERIAL_ID_INVALID);
  result->depth         = meta_data.depth;
  result->instance_id   = meta_data.instance_id;
  result->material_id   = meta_data.material_id;
  result->rel_hit_pos.x = _host_bfloat16_to_float32(meta_data.rel_hit_x_bfloat16);
  result->rel_hit_pos.y = _host_bfloat16_to_float32(meta_data.rel_hit_y_bfloat16);
  result->rel_hit_pos.z = _host_bfloat16_to_float32(meta_data.rel_hit_z_bfloat16);

  return LUMINARY_SUCCESS;
}

LuminaryResult host_queue_output_copy_from_device(Host* host, OutputDescriptor descriptor) {
  __CHECK_NULL_ARGUMENT(host);

  OutputDescriptor* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(host->ringbuffer, sizeof(OutputDescriptor), (void**) &args));

  memcpy(args, &descriptor, sizeof(OutputDescriptor));

  QueueEntry entry;
  memset(&entry, 0, sizeof(QueueEntry));

  entry.name       = "Copy Output";
  entry.function   = (QueueEntryFunction) _host_copy_output_queue_work;
  entry.clear_func = (QueueEntryFunction) 0;
  entry.args       = (void*) args;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_save_png(LuminaryHost* host, LuminaryOutputHandle handle, LuminaryPath* path) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(path);

  if (handle == LUMINARY_OUTPUT_HANDLE_INVALID) {
    __RETURN_ERROR(LUMINARY_ERROR_INVALID_API_ARGUMENT, "Invalid output handle.");
  }

  __FAILURE_HANDLE(output_handler_acquire(host->output_handler, handle));

  const char* file_path_string;
  __FAILURE_HANDLE(path_apply(path, (const char*) 0, &file_path_string));

  const size_t path_length = strlen(file_path_string);

  HostSavePNGArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(host->ringbuffer, sizeof(HostSavePNGArgs), (void**) &args));

  args->handle = handle;

  __FAILURE_HANDLE(host_malloc(&(args->path), path_length + 1));

  memcpy(args->path, file_path_string, path_length);
  args->path[path_length] = '\0';

  QueueEntry entry;
  memset(&entry, 0, sizeof(QueueEntry));

  entry.name       = "Save PNG";
  entry.function   = (QueueEntryFunction) _host_save_png_queue_work;
  entry.clear_func = (QueueEntryFunction) _host_save_png_clear_work;
  entry.args       = (void*) args;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_request_sky_hdri_build(Host* host) {
  __CHECK_NULL_ARGUMENT(host);

  __FAILURE_HANDLE(scene_set_hdri_dirty(host->scene_caller));
  __FAILURE_HANDLE(_host_update_scene(host));

  return LUMINARY_SUCCESS;
}
