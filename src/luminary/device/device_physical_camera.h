#ifndef LUMINARY_DEVICE_CAMERA_H
#define LUMINARY_DEVICE_CAMERA_H

#include "device_utils.h"

struct Device typedef Device;

struct PhysicalCamera {
  uint32_t num_interfaces;
  DeviceCameraInterface* camera_interfaces;
  DeviceCameraMedium* camera_media;
} typedef PhysicalCamera;

LuminaryResult physical_camera_create(PhysicalCamera** physical_camera);
LuminaryResult physical_camera_destroy(PhysicalCamera** physical_camera);

struct DevicePhysicalCameraPtrs {
  CUdeviceptr camera_interfaces;
  CUdeviceptr camera_media;
} typedef DevicePhysicalCameraPtrs;

struct DevicePhysicalCamera {
  uint32_t allocated_num_interfaces;
  DEVICE DeviceCameraInterface* camera_interfaces;
  DEVICE DeviceCameraMedium* camera_media;
} typedef DevicePhysicalCamera;

LuminaryResult device_physical_camera_create(DevicePhysicalCamera** physical_camera);
DEVICE_CTX_FUNC LuminaryResult device_physical_camera_update(
  DevicePhysicalCamera* physical_camera, Device* device, const PhysicalCamera* shared_camera, bool* buffers_have_changed);
DEVICE_CTX_FUNC LuminaryResult device_physical_camera_get_ptrs(DevicePhysicalCamera* physical_camera, DevicePhysicalCameraPtrs* ptrs);
LuminaryResult device_physical_camera_destroy(DevicePhysicalCamera** physical_camera);

#endif /* LUMINARY_DEVICE_CAMERA_H */
