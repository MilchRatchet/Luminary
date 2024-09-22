#include <luminary/host.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

#include "camera.h"
#include "internal_error.h"
#include "internal_host.h"
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
  const char* path;
} typedef HostLoadObjArgs;

static LuminaryResult _host_load_obj_file(Host* host, HostLoadObjArgs* args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(args);

  WavefrontContent* wavefront_content;

  __FAILURE_HANDLE(wavefront_create(&wavefront_content));
  __FAILURE_HANDLE(wavefront_read_file(wavefront_content, args->path));
  __FAILURE_HANDLE(wavefront_convert_content(wavefront_content, host->meshes, host->materials));
  __FAILURE_HANDLE(wavefront_destroy(&wavefront_content));

  __FAILURE_HANDLE(ringbuffer_release_entry(host->ring_buffer, sizeof(HostLoadObjArgs)));

  return LUMINARY_SUCCESS;
}

struct HostSetCameraArgs {
  Camera new_camera;
} typedef HostSetCameraArgs;

static LuminaryResult _host_set_camera(Host* host, HostSetCameraArgs* args) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(ringbuffer_release_entry(host->ring_buffer, sizeof(HostSetCameraArgs)));

  return LUMINARY_ERROR_NOT_IMPLEMENTED;
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
  __FAILURE_HANDLE(ringbuffer_create(&host->ring_buffer, HOST_RINGBUFFER_SIZE));
  __FAILURE_HANDLE(wall_time_create(&host->queue_wall_time));

  __FAILURE_HANDLE(array_create(&host->meshes, sizeof(Mesh*), 16));
  __FAILURE_HANDLE(array_create(&host->materials, sizeof(Material*), 16));

  __FAILURE_HANDLE(camera_get_default(&host->camera));

  memcpy(&host->camera_external, &host->camera, sizeof(Camera));

  host->enable_output = false;

  __FAILURE_HANDLE(thread_start(host->work_thread, (ThreadMainFunc) _host_queue_worker, host));

  *_host = host;

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_destroy(LuminaryHost** host) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(*host);

  __FAILURE_HANDLE(queue_flush_blocking((*host)->work_queue));

  __FAILURE_HANDLE(thread_join((*host)->work_thread));

  __FAILURE_HANDLE(thread_get_last_result((*host)->work_thread));

  __FAILURE_HANDLE(wall_time_destroy(&(*host)->queue_wall_time));
  __FAILURE_HANDLE(ringbuffer_destroy(&(*host)->ring_buffer));
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

  __FAILURE_HANDLE(host_free(host));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_load_obj_file(LuminaryHost* host, const char* path) {
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(path);

  HostLoadObjArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(host->ring_buffer, sizeof(HostLoadObjArgs), (void**) &args));

  args->path = path;

  QueueEntry entry;

  entry.name     = "Loading Obj";
  entry.function = (QueueEntryFunction) _host_load_obj_file;
  entry.args     = args;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_queue_string(const LuminaryHost* host, const char** string) {
  __CHECK_NULL_ARGUMENT(host);

  if (!string) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "String is NULL.");
  }

  __FAILURE_HANDLE(wall_time_get_string(host->queue_wall_time, string));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_camera(Host* host, Camera* camera) {
  __CHECK_NULL_ARGUMENT(host);

  if (!camera) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Camera is NULL.");
  }

  memcpy(camera, &host->camera_external, sizeof(Camera));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_camera(Host* host, Camera* camera) {
  __CHECK_NULL_ARGUMENT(host);

  if (!camera) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Camera is NULL.");
  }

  memcpy(&host->camera_external, camera, sizeof(Camera));

  HostSetCameraArgs* args;
  __FAILURE_HANDLE(ringbuffer_allocate_entry(host->ring_buffer, sizeof(HostSetCameraArgs), (void**) &args));

  memcpy(&args->new_camera, camera, sizeof(Camera));

  QueueEntry entry;

  entry.name     = "Updating camera";
  entry.function = (QueueEntryFunction) _host_set_camera;
  entry.args     = args;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}
