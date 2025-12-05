#include "device_embedded_data.h"

#include "device.h"
#include "device_embedded.h"
#include "host/png.h"
#include "internal_error.h"

LuminaryResult device_embedded_data_create(DeviceEmbeddedData** data) {
  __CHECK_NULL_ARGUMENT(data);

  __FAILURE_HANDLE(host_malloc(data, sizeof(DeviceEmbeddedData)));
  memset(*data, 0, sizeof(DeviceEmbeddedData));

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_embedded_data_update(DeviceEmbeddedData* data, Device* device, bool* buffers_have_changed) {
  __CHECK_NULL_ARGUMENT(data);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(buffers_have_changed);

  *buffers_have_changed = false;

  if (data->bluenoise_1D == (DEVICE void*) 0) {
    void* bluenoise_1D_data;
    int64_t bluenoise_1D_data_length;
    __FAILURE_HANDLE(device_embedded_load(DEVICE_EMBEDDED_FILE_BLUENOISE1D, &bluenoise_1D_data, &bluenoise_1D_data_length));

    __FAILURE_HANDLE(device_malloc(&data->bluenoise_1D, bluenoise_1D_data_length));

    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, bluenoise_1D_data, (DEVICE void*) data->bluenoise_1D, 0, bluenoise_1D_data_length));

    *buffers_have_changed = true;
  }

  if (data->bluenoise_2D == (DEVICE void*) 0) {
    void* bluenoise_2D_data;
    int64_t bluenoise_2D_data_length;
    __FAILURE_HANDLE(device_embedded_load(DEVICE_EMBEDDED_FILE_BLUENOISE2D, &bluenoise_2D_data, &bluenoise_2D_data_length));

    __FAILURE_HANDLE(device_malloc(&data->bluenoise_2D, bluenoise_2D_data_length));

    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, bluenoise_2D_data, (DEVICE void*) data->bluenoise_2D, 0, bluenoise_2D_data_length));

    *buffers_have_changed = true;
  }

  if (data->bridge_lut == (DEVICE void*) 0) {
    void* lut_data;
    int64_t lut_length;
    __FAILURE_HANDLE(device_embedded_load(DEVICE_EMBEDDED_FILE_BRIDGE_LUT, &lut_data, &lut_length));

    __FAILURE_HANDLE(device_malloc(&data->bridge_lut, lut_length));

    __FAILURE_HANDLE(device_staging_manager_register(device->staging_manager, lut_data, (DEVICE void*) data->bridge_lut, 0, lut_length));

    *buffers_have_changed = true;
  }

  if (data->moon_albedo_tex == (DeviceTexture*) 0) {
    void* moon_albedo_data;
    int64_t moon_albedo_data_length;
    __FAILURE_HANDLE(device_embedded_load(DEVICE_EMBEDDED_FILE_MOON_ALBEDO, &moon_albedo_data, &moon_albedo_data_length));

    Texture* moon_albedo_tex;
    __FAILURE_HANDLE(texture_create(&moon_albedo_tex));
    __FAILURE_HANDLE(png_load(moon_albedo_tex, moon_albedo_data, moon_albedo_data_length, "moon_albedo.png"));

    __FAILURE_HANDLE(device_texture_create(&data->moon_albedo_tex, moon_albedo_tex, device, device->stream_main));

    __FAILURE_HANDLE(texture_destroy(&moon_albedo_tex));

    *buffers_have_changed = true;
  }

  if (data->moon_normal_tex == (DeviceTexture*) 0) {
    void* moon_normal_data;
    int64_t moon_normal_data_length;
    __FAILURE_HANDLE(device_embedded_load(DEVICE_EMBEDDED_FILE_MOON_NORMAL, &moon_normal_data, &moon_normal_data_length));

    Texture* moon_normal_tex;
    __FAILURE_HANDLE(texture_create(&moon_normal_tex));
    __FAILURE_HANDLE(png_load(moon_normal_tex, moon_normal_data, moon_normal_data_length, "moon_normal.png"));

    __FAILURE_HANDLE(device_texture_create(&data->moon_normal_tex, moon_normal_tex, device, device->stream_main));

    __FAILURE_HANDLE(texture_destroy(&moon_normal_tex));

    *buffers_have_changed = true;
  }

  if (data->spectral_cdf == (DEVICE void*) 0) {
    void* spectral_cdf_data;
    int64_t spectral_cdf_length;
    __FAILURE_HANDLE(device_embedded_load(DEVICE_EMBEDDED_FILE_CIE1931_CDF, &spectral_cdf_data, &spectral_cdf_length));

    __FAILURE_HANDLE(device_malloc(&data->spectral_cdf, spectral_cdf_length));

    __FAILURE_HANDLE(device_staging_manager_register(
      device->staging_manager, spectral_cdf_data, (DEVICE void*) data->spectral_cdf, 0, spectral_cdf_length));

    *buffers_have_changed = true;
  }

  if (data->spectral_xy_tex == (DeviceTexture*) 0) {
    void* spectral_xy_lut_data;
    int64_t spectral_xy_lut_length;
    __FAILURE_HANDLE(device_embedded_load(DEVICE_EMBEDDED_FILE_CIE1931_XY_LUT, &spectral_xy_lut_data, &spectral_xy_lut_length));

    const uint32_t num_elements = spectral_xy_lut_length / (sizeof(float) * 2);

    Texture* spectral_xy_lut_tex;
    __FAILURE_HANDLE(texture_create(&spectral_xy_lut_tex));
    __FAILURE_HANDLE(texture_fill(spectral_xy_lut_tex, num_elements, 1, 1, spectral_xy_lut_data, TEXTURE_DATA_TYPE_FP32, 2));
    __FAILURE_HANDLE(texture_set_memory_owner(spectral_xy_lut_tex, false));

    spectral_xy_lut_tex->wrap_mode_U = TEXTURE_WRAPPING_MODE_CLAMP;
    spectral_xy_lut_tex->wrap_mode_V = TEXTURE_WRAPPING_MODE_CLAMP;
    spectral_xy_lut_tex->wrap_mode_W = TEXTURE_WRAPPING_MODE_CLAMP;

    __FAILURE_HANDLE(device_texture_create(&data->spectral_xy_tex, spectral_xy_lut_tex, device, device->stream_main));

    __FAILURE_HANDLE(texture_destroy(&spectral_xy_lut_tex));

    *buffers_have_changed = true;
  }

  if (data->spectral_z_tex == (DeviceTexture*) 0) {
    void* spectral_z_lut_data;
    int64_t spectral_z_lut_length;
    __FAILURE_HANDLE(device_embedded_load(DEVICE_EMBEDDED_FILE_CIE1931_Z_LUT, &spectral_z_lut_data, &spectral_z_lut_length));

    const uint32_t num_elements = spectral_z_lut_length / sizeof(float);

    Texture* spectral_z_lut_tex;
    __FAILURE_HANDLE(texture_create(&spectral_z_lut_tex));
    __FAILURE_HANDLE(texture_fill(spectral_z_lut_tex, num_elements, 1, 1, spectral_z_lut_data, TEXTURE_DATA_TYPE_FP32, 1));
    __FAILURE_HANDLE(texture_set_memory_owner(spectral_z_lut_tex, false));

    spectral_z_lut_tex->wrap_mode_U = TEXTURE_WRAPPING_MODE_CLAMP;
    spectral_z_lut_tex->wrap_mode_V = TEXTURE_WRAPPING_MODE_CLAMP;
    spectral_z_lut_tex->wrap_mode_W = TEXTURE_WRAPPING_MODE_CLAMP;

    __FAILURE_HANDLE(device_texture_create(&data->spectral_z_tex, spectral_z_lut_tex, device, device->stream_main));

    __FAILURE_HANDLE(texture_destroy(&spectral_z_lut_tex));

    *buffers_have_changed = true;
  }

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_embedded_data_get_ptrs(DeviceEmbeddedData* data, DeviceEmbeddedDataPtrs* ptrs) {
  __CHECK_NULL_ARGUMENT(data);
  __CHECK_NULL_ARGUMENT(ptrs);

  ptrs->bluenoise_1D = DEVICE_CUPTR(data->bluenoise_1D);
  ptrs->bluenoise_2D = DEVICE_CUPTR(data->bluenoise_2D);
  ptrs->bridge_lut   = DEVICE_CUPTR(data->bridge_lut);
  ptrs->spectral_cdf = DEVICE_CUPTR(data->spectral_cdf);

  __FAILURE_HANDLE(device_struct_texture_object_convert(data->moon_albedo_tex, &ptrs->moon_albedo_tex));
  __FAILURE_HANDLE(device_struct_texture_object_convert(data->moon_normal_tex, &ptrs->moon_normal_tex));
  __FAILURE_HANDLE(device_struct_texture_object_convert(data->spectral_xy_tex, &ptrs->spectral_xy_tex));
  __FAILURE_HANDLE(device_struct_texture_object_convert(data->spectral_z_tex, &ptrs->spectral_z_tex));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_embedded_data_destroy(DeviceEmbeddedData** data) {
  __CHECK_NULL_ARGUMENT(data);

  if ((*data)->bluenoise_1D)
    __FAILURE_HANDLE(device_free(&(*data)->bluenoise_1D));

  if ((*data)->bluenoise_2D)
    __FAILURE_HANDLE(device_free(&(*data)->bluenoise_2D));

  if ((*data)->bridge_lut)
    __FAILURE_HANDLE(device_free(&(*data)->bridge_lut));

  if ((*data)->spectral_cdf)
    __FAILURE_HANDLE(device_free(&(*data)->spectral_cdf));

  if ((*data)->moon_albedo_tex)
    __FAILURE_HANDLE(device_texture_destroy(&(*data)->moon_albedo_tex));

  if ((*data)->moon_normal_tex)
    __FAILURE_HANDLE(device_texture_destroy(&(*data)->moon_normal_tex));

  if ((*data)->spectral_xy_tex)
    __FAILURE_HANDLE(device_texture_destroy(&(*data)->spectral_xy_tex));

  if ((*data)->spectral_z_tex)
    __FAILURE_HANDLE(device_texture_destroy(&(*data)->spectral_z_tex));

  __FAILURE_HANDLE(host_free(data));

  return LUMINARY_SUCCESS;
}
