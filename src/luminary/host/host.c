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

#define HOST_RINGBUFFER_SIZE (0x10000ull)
#define HOST_QUEUE_SIZE (0x100ull)

////////////////////////////////////////////////////////////////////
// Queue worker functions
////////////////////////////////////////////////////////////////////

LuminaryResult _host_queue_worker(Host* host) {
  __CHECK_NULL_ARGUMENT(host);

  bool success = true;

  while (success) {
    QueueEntry entry;
    __FAILURE_HANDLE(queue_pop_blocking(host->work_queue, &entry, &success));

    if (!success)
      break;

    __FAILURE_HANDLE(wall_time_set_string(host->queue_wall_time, entry.name));
    __FAILURE_HANDLE(wall_time_start(host->queue_wall_time));

    __FAILURE_HANDLE(entry.function(host, entry.args));

    __FAILURE_HANDLE(wall_time_stop(host->queue_wall_time));
    __FAILURE_HANDLE(wall_time_set_string(host->queue_wall_time, (const char*) 0));
  }

  return LUMINARY_SUCCESS;
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
  __FAILURE_HANDLE(wavefront_convert_content(wavefront_content, &host->meshes, &host->materials, &host->textures));
  __FAILURE_HANDLE(wavefront_destroy(&wavefront_content));

  // Clean up
  __FAILURE_HANDLE(luminary_path_destroy(&args->path));
  __FAILURE_HANDLE(ringbuffer_release_entry(host->ringbuffer, sizeof(HostLoadObjArgs)));

  return LUMINARY_SUCCESS;
}

struct HostSetSceneEntityArgs {
  void* object;
  size_t size_of_object;
  SceneEntity entity;
} typedef HostSetSceneEntityArgs;

static LuminaryResult _host_set_scene_entity_queue_work(Host* host, HostSetSceneEntityArgs* args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(scene_update(host->scene_internal, args->object, args->entity))

  __FAILURE_HANDLE(ringbuffer_release_entry(host->ringbuffer, args->size_of_object));
  __FAILURE_HANDLE(ringbuffer_release_entry(host->ringbuffer, sizeof(HostSetSceneEntityArgs)));

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// Internal implementation
////////////////////////////////////////////////////////////////////

static LuminaryResult _host_set_scene_entity(Host* host, void* object, size_t size_of_object, SceneEntity entity, const char* string) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(object);

  __FAILURE_HANDLE(scene_update(host->scene_external, object, entity));

  HostSetSceneEntityArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(host->ringbuffer, sizeof(HostSetSceneEntityArgs), (void**) &args));

  args->size_of_object = size_of_object;
  args->entity         = entity;

  __FAILURE_HANDLE(ringbuffer_allocate_entry(host->ringbuffer, args->size_of_object, (void**) &args->object));
  memcpy(args->object, object, args->size_of_object);

  QueueEntry entry;

  entry.name     = string;
  entry.function = (QueueEntryFunction) _host_set_scene_entity_queue_work;
  entry.args     = args;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

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

  __FAILURE_HANDLE(thread_create(&host->work_thread));
  __FAILURE_HANDLE(queue_create(&host->work_queue, sizeof(QueueEntry), HOST_QUEUE_SIZE));
  __FAILURE_HANDLE(ringbuffer_create(&host->ringbuffer, HOST_RINGBUFFER_SIZE));
  __FAILURE_HANDLE(wall_time_create(&host->queue_wall_time));

  __FAILURE_HANDLE(array_create(&host->meshes, sizeof(Mesh*), 16));
  __FAILURE_HANDLE(array_create(&host->materials, sizeof(Material*), 16));
  __FAILURE_HANDLE(array_create(&host->textures, sizeof(Texture*), 16));

  __FAILURE_HANDLE(scene_create(&host->scene_external));
  __FAILURE_HANDLE(scene_create(&host->scene_internal));

  host->enable_output = false;

  __FAILURE_HANDLE(thread_start(host->work_thread, (ThreadMainFunc) _host_queue_worker, host));

  *_host = host;

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_destroy(Host** host) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(*host);

  __FAILURE_HANDLE(queue_flush_blocking((*host)->work_queue));

  __FAILURE_HANDLE(thread_join((*host)->work_thread));

  __FAILURE_HANDLE(thread_get_last_result((*host)->work_thread));

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

  uint32_t material_count;
  __FAILURE_HANDLE(array_get_num_elements((*host)->materials, &material_count));

  for (uint32_t mat_id = 0; mat_id < material_count; mat_id++) {
    __FAILURE_HANDLE(host_free(&(*host)->materials[mat_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*host)->materials));

  uint32_t texture_count;
  __FAILURE_HANDLE(array_get_num_elements((*host)->textures, &texture_count));

  for (uint32_t tex_id = 0; tex_id < texture_count; tex_id++) {
    __FAILURE_HANDLE(texture_destroy(&(*host)->textures[tex_id]));
  }

  __FAILURE_HANDLE(array_destroy(&(*host)->textures));

  __FAILURE_HANDLE(scene_destroy(&(*host)->scene_external));
  __FAILURE_HANDLE(scene_destroy(&(*host)->scene_internal));

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

  entry.name     = "Loading Obj";
  entry.function = (QueueEntryFunction) _host_load_obj_file;
  entry.args     = args;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_load_lum_file(Host* host, Path* path) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(path);

  LumFileContent* content;
  __FAILURE_HANDLE(lum_content_create(&content));

  Path* lum_path;
  __FAILURE_HANDLE(path_copy(&lum_path, path));

  __FAILURE_HANDLE(lum_read_file(lum_path, content));

  __FAILURE_HANDLE(luminary_path_destroy(&lum_path));

  __FAILURE_HANDLE(luminary_host_set_settings(host, &content->settings));
  __FAILURE_HANDLE(luminary_host_set_camera(host, &content->camera));
  __FAILURE_HANDLE(luminary_host_set_ocean(host, &content->ocean));
  __FAILURE_HANDLE(luminary_host_set_sky(host, &content->sky));
  __FAILURE_HANDLE(luminary_host_set_cloud(host, &content->cloud));
  __FAILURE_HANDLE(luminary_host_set_fog(host, &content->fog));
  __FAILURE_HANDLE(luminary_host_set_particles(host, &content->particles));
  __FAILURE_HANDLE(luminary_host_set_toy(host, &content->toy));

  uint32_t num_obj_files_to_load;
  __FAILURE_HANDLE(array_get_num_elements(content->obj_file_path_strings, &num_obj_files_to_load));

  for (uint32_t obj_file_id = 0; obj_file_id < num_obj_files_to_load; obj_file_id++) {
    Path* obj_path;
    __FAILURE_HANDLE(path_extend(&obj_path, path, content->obj_file_path_strings[obj_file_id]));

    __FAILURE_HANDLE(luminary_host_load_obj_file(host, obj_path));

    __FAILURE_HANDLE(luminary_path_destroy(&obj_path));
  }

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

LuminaryResult luminary_host_get_max_sample_count(Host* host, uint32_t* max_sample_count);
LuminaryResult luminary_host_set_max_sample_count(Host* host, uint32_t* max_sample_count);

LuminaryResult luminary_host_get_settings(Host* host, RendererSettings* settings) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(settings);

  __FAILURE_HANDLE(scene_get_locking(host->scene_external, settings, SCENE_ENTITY_SETTINGS));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_settings(Host* host, RendererSettings* settings) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(settings);

  _host_set_scene_entity(host, (void*) settings, sizeof(RendererSettings), SCENE_ENTITY_SETTINGS, "Updating settings");

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_camera(Host* host, Camera* camera) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(camera);

  __FAILURE_HANDLE(scene_get_locking(host->scene_external, camera, SCENE_ENTITY_CAMERA));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_camera(Host* host, Camera* camera) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(camera);

  _host_set_scene_entity(host, (void*) camera, sizeof(Camera), SCENE_ENTITY_CAMERA, "Updating camera");

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_ocean(Host* host, Ocean* ocean) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(ocean);

  __FAILURE_HANDLE(scene_get_locking(host->scene_external, ocean, SCENE_ENTITY_OCEAN));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_ocean(Host* host, Ocean* ocean) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(ocean);

  _host_set_scene_entity(host, (void*) ocean, sizeof(Ocean), SCENE_ENTITY_OCEAN, "Updating ocean");

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_sky(Host* host, Sky* sky) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(sky);

  __FAILURE_HANDLE(scene_get_locking(host->scene_external, sky, SCENE_ENTITY_SKY));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_sky(Host* host, Sky* sky) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(sky);

  _host_set_scene_entity(host, (void*) sky, sizeof(Sky), SCENE_ENTITY_SKY, "Updating sky");

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_cloud(Host* host, Cloud* cloud) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(cloud);

  __FAILURE_HANDLE(scene_get_locking(host->scene_external, cloud, SCENE_ENTITY_CLOUD));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_cloud(Host* host, Cloud* cloud) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(cloud);

  _host_set_scene_entity(host, (void*) cloud, sizeof(Cloud), SCENE_ENTITY_CLOUD, "Updating cloud");

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_fog(Host* host, Fog* fog) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(fog);

  __FAILURE_HANDLE(scene_get_locking(host->scene_external, fog, SCENE_ENTITY_FOG));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_fog(Host* host, Fog* fog) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(fog);

  _host_set_scene_entity(host, (void*) fog, sizeof(Fog), SCENE_ENTITY_FOG, "Updating fog");

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_particles(Host* host, Particles* particles) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(particles);

  __FAILURE_HANDLE(scene_get_locking(host->scene_external, particles, SCENE_ENTITY_PARTICLES));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_particles(Host* host, Particles* particles) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(particles);

  _host_set_scene_entity(host, (void*) particles, sizeof(Particles), SCENE_ENTITY_PARTICLES, "Updating particles");

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_toy(Host* host, Toy* toy) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(toy);

  __FAILURE_HANDLE(scene_get_locking(host->scene_external, toy, SCENE_ENTITY_TOY));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_toy(Host* host, Toy* toy) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(toy);

  _host_set_scene_entity(host, (void*) toy, sizeof(Toy), SCENE_ENTITY_TOY, "Updating toy");

  return LUMINARY_SUCCESS;
}
