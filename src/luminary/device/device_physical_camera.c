#include "device_physical_camera.h"

#include "internal_error.h"

LuminaryResult physical_camera_create(PhysicalCamera** physical_camera) {
  __CHECK_NULL_ARGUMENT(physical_camera);

  __FAILURE_HANDLE(host_malloc(physical_camera, sizeof(PhysicalCamera)));
  memset(*physical_camera, 0, sizeof(PhysicalCamera));

  // TODO TODO TODO
  // This is just a placeholder initialization until the camera is given through the API

  (*physical_camera)->num_interfaces = 12;

  __FAILURE_HANDLE(host_malloc(&(*physical_camera)->camera_interfaces, 12 * sizeof(DeviceCameraInterface)));
  __FAILURE_HANDLE(host_malloc(&(*physical_camera)->camera_media, 13 * sizeof(DeviceCameraMedium)));

  // Diameter are rough estimates
  (*physical_camera)->camera_interfaces[0]  = (DeviceCameraInterface) {.radius = -94.29f, .vertex = 0.0f, .diameter = 14.0f};
  (*physical_camera)->camera_interfaces[1]  = (DeviceCameraInterface) {.radius = 181.58f, .vertex = 7.17f, .diameter = 14.0f};
  (*physical_camera)->camera_interfaces[2]  = (DeviceCameraInterface) {.radius = -72.86f, .vertex = 9.3f, .diameter = 12.0f};
  (*physical_camera)->camera_interfaces[3]  = (DeviceCameraInterface) {.radius = 76.74f, .vertex = 21.7f, .diameter = 12.0f};
  (*physical_camera)->camera_interfaces[4]  = (DeviceCameraInterface) {.radius = -43.02f, .vertex = 23.83f, .diameter = 12.0f};
  (*physical_camera)->camera_interfaces[5]  = (DeviceCameraInterface) {.radius = 27.44f, .vertex = 45.14f, .diameter = 17.0f};
  (*physical_camera)->camera_interfaces[6]  = (DeviceCameraInterface) {.radius = -321.70f, .vertex = 49.53f, .diameter = 17.0f};
  (*physical_camera)->camera_interfaces[7]  = (DeviceCameraInterface) {.radius = 50.96f, .vertex = 70.01f, .diameter = 17.0f};
  (*physical_camera)->camera_interfaces[8]  = (DeviceCameraInterface) {.radius = 120.34f, .vertex = 70.97f, .diameter = 20.0f};
  (*physical_camera)->camera_interfaces[9]  = (DeviceCameraInterface) {.radius = 68.99f, .vertex = 78.97f, .diameter = 20.0f};
  (*physical_camera)->camera_interfaces[10] = (DeviceCameraInterface) {.radius = 251.93f, .vertex = 79.18f, .diameter = 23.2f};
  (*physical_camera)->camera_interfaces[11] = (DeviceCameraInterface) {.radius = 94.00f, .vertex = 88.18f, .diameter = 23.2f};

  (*physical_camera)->camera_media[0]  = (DeviceCameraMedium) {.design_ior = 1.0f, .abbe = 0.0f};
  (*physical_camera)->camera_media[1]  = (DeviceCameraMedium) {.design_ior = 1.6435f, .abbe = 53.5f};
  (*physical_camera)->camera_media[2]  = (DeviceCameraMedium) {.design_ior = 1.0f, .abbe = 0.0f};
  (*physical_camera)->camera_media[3]  = (DeviceCameraMedium) {.design_ior = 1.6935f, .abbe = 53.5f};
  (*physical_camera)->camera_media[4]  = (DeviceCameraMedium) {.design_ior = 1.5174f, .abbe = 52.5f};
  (*physical_camera)->camera_media[5]  = (DeviceCameraMedium) {.design_ior = 1.0f, .abbe = 0.0f};
  (*physical_camera)->camera_media[6]  = (DeviceCameraMedium) {.design_ior = 1.7174f, .abbe = 29.5f};
  (*physical_camera)->camera_media[7]  = (DeviceCameraMedium) {.design_ior = 1.6385f, .abbe = 55.5f};
  (*physical_camera)->camera_media[8]  = (DeviceCameraMedium) {.design_ior = 1.0f, .abbe = 0.0f};
  (*physical_camera)->camera_media[9]  = (DeviceCameraMedium) {.design_ior = 1.7173f, .abbe = 47.9f};
  (*physical_camera)->camera_media[10] = (DeviceCameraMedium) {.design_ior = 1.0f, .abbe = 0.0f};
  (*physical_camera)->camera_media[11] = (DeviceCameraMedium) {.design_ior = 1.6935f, .abbe = 53.5f};
  (*physical_camera)->camera_media[12] = (DeviceCameraMedium) {.design_ior = 1.0f, .abbe = 0.0f};

  // TODO TODO TODO

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
