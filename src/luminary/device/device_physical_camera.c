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

struct RayTransferMatrix {
  float A;
  float B;
  float C;
  float D;
} typedef RayTransferMatrix;

static RayTransferMatrix _physical_camera_run_ray_transfer_matrix_analysis(const LensTemplateData* template_data) {
  float A = 1.0f;
  float B = 0.0f;
  float C = 0.0f;
  float D = 1.0f;

  float z_curr = -template_data->last_vertex;

  for (int32_t i = template_data->num_interfaces - 1; i >= 0; i--) {
    float z_v = -template_data->interfaces[i].vertex;
    float d   = z_v - z_curr;

    float new_A = A + d * C;
    float new_B = B + d * D;
    A           = new_A;
    B           = new_B;

    z_curr = z_v;

    // media[i+1] is before the interface when travelling backwards from the scene
    float n1 = template_data->media[i + 1].design_ior;
    float n2 = template_data->media[i].design_ior;
    float r  = template_data->interfaces[i].radius;

    if (r != FLT_MAX && r != 0.0f) {
      float M21 = (n1 - n2) / (r * n2);
      float M22 = n1 / n2;

      float new_C = M21 * A + M22 * C;
      float new_D = M21 * B + M22 * D;
      C           = new_C;
      D           = new_D;
    }
    else {
      float M22 = n1 / n2;
      C         = M22 * C;
      D         = M22 * D;
    }
  }

  return (RayTransferMatrix) {.A = A, .B = B, .C = C, .D = D};
}

static float _physical_camera_compute_auto_focus(
  const LensTemplateData* template_data, const RayTransferMatrix matrix, float object_distance) {
  const float d_o = object_distance * CAMERA_COMMON_INV_SCALE;

  // Standard focus condition from an object at d_o hitting a transfer matrix:
  const float d_i = -(matrix.A * d_o + matrix.B) / (matrix.C * d_o + matrix.D);

  // d_i is the distance past the final traced interface (interfaces[0])
  const float sensor_distance = d_i - template_data->interfaces[0].vertex;

  return sensor_distance;
}

LuminaryResult physical_camera_generate(PhysicalCamera* physical_camera, const Camera* camera) {
  __CHECK_NULL_ARGUMENT(physical_camera);

  LensTemplateData template_data;
  __FAILURE_HANDLE(lens_library_get_template_data(camera->lens_template, &template_data));

  if (physical_camera->num_allocated_interfaces < template_data.num_interfaces) {
    if (physical_camera->camera_interfaces)
      __FAILURE_HANDLE(host_free(&physical_camera->camera_interfaces));

    if (physical_camera->camera_media)
      __FAILURE_HANDLE(host_free(&physical_camera->camera_media));

    __FAILURE_HANDLE(host_malloc(&physical_camera->camera_interfaces, template_data.num_interfaces * sizeof(DeviceCameraInterface)));
    __FAILURE_HANDLE(host_malloc(&physical_camera->camera_media, (template_data.num_interfaces + 1) * sizeof(DeviceCameraMedium)));

    physical_camera->num_allocated_interfaces = template_data.num_interfaces;
  }

  float sensor_distance = 1.0f;

  float aperture_radius = 0.0f;
  if (camera->lens.aperture_stop < 32.0f * 1024.0f)
    aperture_radius = (template_data.design_focal_length / camera->lens.aperture_stop) * 0.5f;

  float exit_pupil_radius = template_data.exit_pupil_diameter * 0.5f;
  float exit_pupil_point  = template_data.exit_pupil_point;

  if (camera->lens_template != LUMINARY_LENS_TEMPLATE_THIN_LENS) {
    const RayTransferMatrix matrix = _physical_camera_run_ray_transfer_matrix_analysis(&template_data);

    if (camera->lens.use_auto_focus) {
      sensor_distance = _physical_camera_compute_auto_focus(&template_data, matrix, camera->lens.object_distance / camera->scale);
    }
    else {
      sensor_distance = camera->lens.sensor_distance;
    }

    if (matrix.D != 0.0f) {
      const float matrix_determinant = matrix.A * matrix.D - matrix.B * matrix.C;

      exit_pupil_point  = matrix.B / matrix.D;
      exit_pupil_radius = fabsf(matrix_determinant / matrix.D) * aperture_radius;
    }
  }

  physical_camera->num_interfaces    = template_data.num_interfaces;
  physical_camera->aperture_radius   = aperture_radius;
  physical_camera->aperture_point    = template_data.aperture_point;
  physical_camera->exit_pupil_point  = exit_pupil_point;
  physical_camera->exit_pupil_radius = exit_pupil_radius;
  physical_camera->last_vertex       = template_data.last_vertex;
  physical_camera->sensor_distance   = sensor_distance;

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

  physical_camera->aux_data.num_interfaces    = shared_camera->num_interfaces;
  physical_camera->aux_data.aperture_radius   = shared_camera->aperture_radius;
  physical_camera->aux_data.aperture_point    = shared_camera->aperture_point;
  physical_camera->aux_data.exit_pupil_point  = shared_camera->exit_pupil_point;
  physical_camera->aux_data.exit_pupil_radius = shared_camera->exit_pupil_radius;
  physical_camera->aux_data.last_vertex       = shared_camera->last_vertex;
  physical_camera->aux_data.sensor_distance   = shared_camera->sensor_distance;

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
