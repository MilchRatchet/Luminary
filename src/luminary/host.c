#include <luminary/host.h>
#include <stdlib.h>
#include <string.h>

#include "camera.h"
#include "internal_error.h"
#include "internal_host.h"

#define HOST_RINGBUFFER_SIZE (0x10000ull)
#define HOST_QUEUE_SIZE (0x100ull)

////////////////////////////////////////////////////////////////////
// Queue worker functions
////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////
// Queue work functions
////////////////////////////////////////////////////////////////////

struct HostSetCameraArgs {
  Camera new_camera;
} typedef HostSetCameraArgs;

static LuminaryResult _host_set_camera(Host* host, HostSetCameraArgs* args) {
  bool camera_is_dirty = false;

  // TODO: Implement camera update logic.

  return LUMINARY_ERROR_NOT_IMPLEMENTED;
}

////////////////////////////////////////////////////////////////////
// External API implementation
////////////////////////////////////////////////////////////////////

LuminaryResult luminary_host_create(Host** _host) {
  if (!_host) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Host was NULL.");
  }

  Host* host;
  __FAILURE_HANDLE(host_malloc(&host, sizeof(Host)));

  memset(host, 0, sizeof(Host));

  __FAILURE_HANDLE(queue_create(&host->work_queue, sizeof(QueueEntry), HOST_QUEUE_SIZE));
  __FAILURE_HANDLE(ringbuffer_create(&host->ring_buffer, HOST_RINGBUFFER_SIZE));

  __FAILURE_HANDLE(camera_get_default(&host->camera));

  memcpy(&host->camera_external, &host->camera, sizeof(Camera));

  host->enable_output = false;

  *_host = host;

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_get_camera(Host* host, Camera* camera) {
  if (!host) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Host was NULL.");
  }

  if (!camera) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Camera was NULL.");
  }

  memcpy(camera, &host->camera_external, sizeof(Camera));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_camera(Host* host, Camera* camera) {
  if (!host) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Host was NULL.");
  }

  if (!camera) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Camera was NULL.");
  }

  memcpy(&host->camera_external, camera, sizeof(Camera));

  HostSetCameraArgs* args;
  __FAILURE_HANDLE(ringbuffer_get_entry(host->ring_buffer, sizeof(HostSetCameraArgs), &args));

  memcpy(&args->new_camera, camera, sizeof(Camera));

  QueueEntry entry;

  entry.name     = "Updating camera";
  entry.function = _host_set_camera;
  entry.args     = args;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}
