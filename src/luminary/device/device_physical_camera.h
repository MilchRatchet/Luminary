#ifndef LUMINARY_DEVICE_CAMERA_H
#define LUMINARY_DEVICE_CAMERA_H

#include "device_utils.h"

struct PhysicalCamera {
  uint32_t num_interfaces;
  DeviceCameraInterface* camera_interfaces;
  DeviceCameraMedium* camera_media;
} typedef PhysicalCamera;

LuminaryResult physical_camera_create(PhysicalCamera** physical_camera);
LuminaryResult physical_camera_destroy(PhysicalCamera** physical_camera);

#endif /* LUMINARY_DEVICE_CAMERA_H */
