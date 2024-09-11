#include <luminary/host.h>
#include <stdlib.h>
#include <string.h>

#include "internal_error.h"
#include "internal_host.h"

////////////////////////////////////////////////////////////////////
// Queue work functions
////////////////////////////////////////////////////////////////////

struct HostSetCameraArgs {
  Camera new_camera;
} typedef HostSetCameraArgs;

static LuminaryResult _host_set_camera(Host* host, HostSetCameraArgs* args) {
  bool camera_is_dirty = false;

  // TODO: Implement camera update logic.

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// External API implementation
////////////////////////////////////////////////////////////////////

LuminaryResult luminary_host_get_camera(Host* host, Camera* camera) {
  if (!camera) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Camera was NULL.");
  }

  memcpy(camera, host->camera_external, sizeof(Camera));

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_host_set_camera(Host* host, Camera* camera) {
  if (!camera) {
    __RETURN_ERROR(LUMINARY_ERROR_ARGUMENT_NULL, "Camera was NULL.");
  }

  host->camera_external = camera;

  // TODO: Get memory for the args.
  HostSetCameraArgs* args;
  __FAILURE_HANDLE(host_malloc(&args, sizeof(HostSetCameraArgs)));

  memcpy(&args->new_camera, camera, sizeof(Camera));

  QueueEntry entry;

  entry.name     = "Updating camera";
  entry.function = _host_set_camera;
  entry.args     = args;

  __FAILURE_HANDLE(queue_push(host->work_queue, &entry));

  return LUMINARY_SUCCESS;
}
