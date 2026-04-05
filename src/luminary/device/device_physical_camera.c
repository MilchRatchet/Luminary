#include "device_physical_camera.h"

#include <float.h>

#include "device.h"
#include "internal_error.h"
#include "lens_library.h"

LuminaryResult physical_camera_create(PhysicalCamera** physical_camera) {
  __CHECK_NULL_ARGUMENT(physical_camera);

  __FAILURE_HANDLE(host_malloc(physical_camera, sizeof(PhysicalCamera)));
  memset(*physical_camera, 0, sizeof(PhysicalCamera));

  return LUMINARY_SUCCESS;
}

LuminaryResult physical_camera_generate(PhysicalCamera* physical_camera, const Camera* camera) {
  __CHECK_NULL_ARGUMENT(physical_camera);

  LensTemplateData template_data;
  __FAILURE_HANDLE(lens_library_get_template_data(camera->lens_template, camera->lens[camera->lens_template].focal_length, &template_data));

  if (physical_camera->num_allocated_interfaces < template_data.num_interfaces) {
    if (physical_camera->camera_interfaces)
      __FAILURE_HANDLE(host_free(&physical_camera->camera_interfaces));

    if (physical_camera->camera_media)
      __FAILURE_HANDLE(host_free(&physical_camera->camera_media));

    __FAILURE_HANDLE(host_malloc(&physical_camera->camera_interfaces, template_data.num_interfaces * sizeof(DeviceCameraInterface)));
    __FAILURE_HANDLE(host_malloc(&physical_camera->camera_media, (template_data.num_interfaces + 1) * sizeof(DeviceCameraMedium)));

    physical_camera->num_allocated_interfaces = template_data.num_interfaces;
  }

  physical_camera->num_interfaces      = template_data.num_interfaces;
  physical_camera->design_focal_length = template_data.design_focal_length;
  physical_camera->aperture_point      = template_data.aperture_point;
  physical_camera->exit_pupil_point    = template_data.exit_pupil_point;
  physical_camera->exit_pupil_diameter = template_data.exit_pupil_diameter;

  if (camera->lens_template == LUMINARY_LENS_TEMPLATE_THIN_LENS)
    return LUMINARY_SUCCESS;

  for (uint32_t interface_id = 0; interface_id < physical_camera->num_interfaces; interface_id++) {
    physical_camera->camera_interfaces[interface_id] = (DeviceCameraInterface) {
      .radius             = template_data.interfaces[interface_id].radius,
      .vertex             = template_data.interfaces[interface_id].vertex,
      .cylindrical_radius = template_data.interfaces[interface_id].cylindrical_radius,
    };
  }

  for (uint32_t medium_id = 0; medium_id < physical_camera->num_interfaces + 1; medium_id++) {
    physical_camera->camera_media[medium_id] = (DeviceCameraMedium) {
      .design_ior         = template_data.media[medium_id].design_ior,
      .abbe               = template_data.media[medium_id].abbe,
      .cylindrical_radius = template_data.media[medium_id].cylindrical_radius,
    };
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult physical_camera_destroy(PhysicalCamera** physical_camera) {
  __CHECK_NULL_ARGUMENT(physical_camera);

  if ((*physical_camera)->camera_interfaces != (DeviceCameraInterface*) 0) {
    __FAILURE_HANDLE(host_free(&(*physical_camera)->camera_interfaces));
  }

  if ((*physical_camera)->camera_media != (DeviceCameraMedium*) 0) {
    __FAILURE_HANDLE(host_free(&(*physical_camera)->camera_media));
  }

  __FAILURE_HANDLE(host_free(physical_camera));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_physical_camera_create(DevicePhysicalCamera** physical_camera) {
  __CHECK_NULL_ARGUMENT(physical_camera);

  __FAILURE_HANDLE(host_malloc(physical_camera, sizeof(DevicePhysicalCamera)));
  memset(*physical_camera, 0, sizeof(DevicePhysicalCamera));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_physical_camera_update(
  DevicePhysicalCamera* physical_camera, Device* device, const PhysicalCamera* shared_camera, bool* buffers_have_changed) {
  __CHECK_NULL_ARGUMENT(physical_camera);
  __CHECK_NULL_ARGUMENT(shared_camera);

  *buffers_have_changed = false;

  if (shared_camera->num_interfaces != physical_camera->allocated_num_interfaces) {
    if (physical_camera->camera_interfaces)
      __FAILURE_HANDLE(device_free(&physical_camera->camera_interfaces));

    if (physical_camera->camera_media)
      __FAILURE_HANDLE(device_free(&physical_camera->camera_media));

    __FAILURE_HANDLE(device_malloc(&physical_camera->camera_interfaces, shared_camera->num_interfaces * sizeof(DeviceCameraInterface)));
    __FAILURE_HANDLE(device_malloc(&physical_camera->camera_media, (shared_camera->num_interfaces + 1) * sizeof(DeviceCameraMedium)));

    physical_camera->allocated_num_interfaces = shared_camera->num_interfaces;
    *buffers_have_changed                     = true;
  }

  physical_camera->aux_data.num_interfaces      = shared_camera->num_interfaces;
  physical_camera->aux_data.design_focal_length = shared_camera->design_focal_length;
  physical_camera->aux_data.aperture_point      = shared_camera->aperture_point;
  physical_camera->aux_data.exit_pupil_point    = shared_camera->exit_pupil_point;
  physical_camera->aux_data.exit_pupil_radius   = shared_camera->exit_pupil_diameter * 0.5f;

  if (physical_camera->allocated_num_interfaces > 0) {
    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, shared_camera->camera_interfaces, (DEVICE void*) physical_camera->camera_interfaces, 0,
      physical_camera->allocated_num_interfaces * sizeof(DeviceCameraInterface)));
    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, shared_camera->camera_media, (DEVICE void*) physical_camera->camera_media, 0,
      (physical_camera->allocated_num_interfaces + 1) * sizeof(DeviceCameraMedium)));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_physical_camera_get_ptrs(DevicePhysicalCamera* physical_camera, DevicePhysicalCameraPtrs* ptrs) {
  __CHECK_NULL_ARGUMENT(physical_camera);
  __CHECK_NULL_ARGUMENT(ptrs);

  ptrs->camera_interfaces = DEVICE_CUPTR(physical_camera->camera_interfaces);
  ptrs->camera_media      = DEVICE_CUPTR(physical_camera->camera_media);

  return LUMINARY_SUCCESS;
}

LuminaryResult device_physical_camera_destroy(DevicePhysicalCamera** physical_camera) {
  __CHECK_NULL_ARGUMENT(physical_camera);
  __CHECK_NULL_ARGUMENT(*physical_camera);

  if ((*physical_camera)->camera_interfaces)
    __FAILURE_HANDLE(device_free(&(*physical_camera)->camera_interfaces));

  if ((*physical_camera)->camera_media)
    __FAILURE_HANDLE(device_free(&(*physical_camera)->camera_media));

  __FAILURE_HANDLE(host_free(physical_camera));

  return LUMINARY_SUCCESS;
}
