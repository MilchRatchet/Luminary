#include "device_bsdf.h"

#include "device.h"
#include "internal_error.h"

LuminaryResult bsdf_lut_create(BSDFLUT** lut) {
  __CHECK_NULL_ARGUMENT(lut);

  __FAILURE_HANDLE(host_malloc(lut, sizeof(BSDFLUT)));
  memset(*lut, 0, sizeof(BSDFLUT));

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult bsdf_lut_generate(BSDFLUT* lut, Device* device) {
  __CHECK_NULL_ARGUMENT(lut);
  __CHECK_NULL_ARGUMENT(device);

  // TODO: Actually compute the LUTs.
  uint16_t* conductor_data;
  __FAILURE_HANDLE(host_malloc(&conductor_data, sizeof(uint16_t) * BSDF_LUT_SIZE * BSDF_LUT_SIZE));
  memset(conductor_data, 0xFF, sizeof(uint16_t) * BSDF_LUT_SIZE * BSDF_LUT_SIZE);

  uint16_t* specular_data;
  __FAILURE_HANDLE(host_malloc(&specular_data, sizeof(uint16_t) * BSDF_LUT_SIZE * BSDF_LUT_SIZE));
  memset(specular_data, 0xFF, sizeof(uint16_t) * BSDF_LUT_SIZE * BSDF_LUT_SIZE);

  uint16_t* dielectric_data;
  __FAILURE_HANDLE(host_malloc(&dielectric_data, sizeof(uint16_t) * BSDF_LUT_SIZE * BSDF_LUT_SIZE * BSDF_LUT_SIZE));
  memset(dielectric_data, 0xFF, sizeof(uint16_t) * BSDF_LUT_SIZE * BSDF_LUT_SIZE * BSDF_LUT_SIZE);

  uint16_t* dielectric_inv_data;
  __FAILURE_HANDLE(host_malloc(&dielectric_inv_data, sizeof(uint16_t) * BSDF_LUT_SIZE * BSDF_LUT_SIZE * BSDF_LUT_SIZE));
  memset(dielectric_inv_data, 0xFF, sizeof(uint16_t) * BSDF_LUT_SIZE * BSDF_LUT_SIZE * BSDF_LUT_SIZE);

  __FAILURE_HANDLE(texture_create(&lut->conductor, BSDF_LUT_SIZE, BSDF_LUT_SIZE, 1, conductor_data, TexDataUINT16, 1));
  __FAILURE_HANDLE(texture_create(&lut->specular, BSDF_LUT_SIZE, BSDF_LUT_SIZE, 1, specular_data, TexDataUINT16, 1));
  __FAILURE_HANDLE(texture_create(&lut->dielectric, BSDF_LUT_SIZE, BSDF_LUT_SIZE, BSDF_LUT_SIZE, dielectric_data, TexDataUINT16, 1));
  __FAILURE_HANDLE(
    texture_create(&lut->dielectric_inv, BSDF_LUT_SIZE, BSDF_LUT_SIZE, BSDF_LUT_SIZE, dielectric_inv_data, TexDataUINT16, 1));

  return LUMINARY_SUCCESS;
}

LuminaryResult bsdf_lut_destroy(BSDFLUT** lut) {
  __CHECK_NULL_ARGUMENT(lut);

  __FAILURE_HANDLE(host_free(lut));

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_bsdf_lut_create(DeviceBSDFLUT** lut) {
  __CHECK_NULL_ARGUMENT(lut);

  __FAILURE_HANDLE(host_malloc(lut, sizeof(DeviceBSDFLUT)));
  memset(*lut, 0, sizeof(DeviceBSDFLUT));

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_bsdf_lut_update(DeviceBSDFLUT* lut, Device* device, const BSDFLUT* source_lut) {
  __CHECK_NULL_ARGUMENT(lut);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(source_lut);

  if (lut->conductor) {
    __FAILURE_HANDLE(device_texture_destroy(&lut->conductor));
  }

  if (lut->specular) {
    __FAILURE_HANDLE(device_texture_destroy(&lut->specular));
  }

  if (lut->dielectric) {
    __FAILURE_HANDLE(device_texture_destroy(&lut->dielectric));
  }

  if (lut->dielectric_inv) {
    __FAILURE_HANDLE(device_texture_destroy(&lut->dielectric_inv));
  }

  __FAILURE_HANDLE(device_texture_create(&lut->conductor, source_lut->conductor, device->stream_main));
  __FAILURE_HANDLE(device_texture_create(&lut->specular, source_lut->specular, device->stream_main));
  __FAILURE_HANDLE(device_texture_create(&lut->dielectric, source_lut->dielectric, device->stream_main));
  __FAILURE_HANDLE(device_texture_create(&lut->dielectric_inv, source_lut->dielectric_inv, device->stream_main));

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult device_bsdf_lut_destroy(DeviceBSDFLUT** lut) {
  __CHECK_NULL_ARGUMENT(lut);

  __FAILURE_HANDLE(host_free(lut));

  return LUMINARY_SUCCESS;
}
