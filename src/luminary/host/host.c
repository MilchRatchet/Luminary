#include <luminary/host.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>

#include "camera.h"
#include "internal_error.h"
#include "internal_host.h"

#define HOST_RINGBUFFER_SIZE (0x10000ull)
#define HOST_QUEUE_SIZE (0x100ull)

////////////////////////////////////////////////////////////////////
// Queue worker functions
////////////////////////////////////////////////////////////////////

LuminaryResult _host_queue_worker(Host* host) {
  while (!host->exit_requested) {
    QueueEntry entry;
    bool success;
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

struct HostSetCameraArgs {
  Camera new_camera;
} typedef HostSetCameraArgs;

static LuminaryResult _host_set_camera(Host* host, HostSetCameraArgs* args) {
  LUM_UNUSED(host);
  LUM_UNUSED(args);
  return LUMINARY_ERROR_NOT_IMPLEMENTED;
}

////////////////////////////////////////////////////////////////////
// External API implementation
////////////////////////////////////////////////////////////////////

LuminaryResult luminary_host_create(Host** _host) {
  if (!_host) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Host is NULL.");
  }

  Host* host;
  __FAILURE_HANDLE(host_malloc(&host, sizeof(Host)));

  memset(host, 0, sizeof(Host));

  __FAILURE_HANDLE(queue_create(&host->work_queue, sizeof(QueueEntry), HOST_QUEUE_SIZE));
  __FAILURE_HANDLE(ringbuffer_create(&host->ring_buffer, HOST_RINGBUFFER_SIZE));
  __FAILURE_HANDLE(wall_time_create(&host->queue_wall_time));

  __FAILURE_HANDLE(camera_get_default(&host->camera));

  memcpy(&host->camera_external, &host->camera, sizeof(Camera));

  host->current_work_string = (const char*) 0;
  host->enable_output       = false;
  host->exit_requested      = false;

  *_host = host;

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_queue_string(const LuminaryHost* host, const char** string) {
  if (!host) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Host is NULL.");
  }

  if (!string) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "String is NULL.");
  }

  __FAILURE_HANDLE(wall_time_get_string(host->queue_wall_time, string));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_camera(Host* host, Camera* camera) {
  if (!host) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Host is NULL.");
  }

  if (!camera) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Camera is NULL.");
  }

  memcpy(camera, &host->camera_external, sizeof(Camera));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_camera(Host* host, Camera* camera) {
  if (!host) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Host is NULL.");
  }

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
