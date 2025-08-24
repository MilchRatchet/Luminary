#include "device_bsdf.h"

#include "device.h"
#include "internal_error.h"
#include "kernel_args.h"

LuminaryResult bsdf_lut_create(BSDFLUT** lut) {
  __CHECK_NULL_ARGUMENT(lut);

  __FAILURE_HANDLE(host_malloc(lut, sizeof(BSDFLUT)));
  memset(*lut, 0, sizeof(BSDFLUT));

  void* conductor_data;
  __FAILURE_HANDLE(host_malloc(&conductor_data, BSDF_LUT_SIZE * BSDF_LUT_SIZE * sizeof(uint16_t)));

  void* specular_data;
  __FAILURE_HANDLE(host_malloc(&specular_data, BSDF_LUT_SIZE * BSDF_LUT_SIZE * sizeof(uint16_t)));

  void* dielectric_data;
  __FAILURE_HANDLE(host_malloc(&dielectric_data, BSDF_LUT_SIZE * BSDF_LUT_SIZE * BSDF_LUT_SIZE * sizeof(uint16_t)));

  void* dielectric_inv_data;
  __FAILURE_HANDLE(host_malloc(&dielectric_inv_data, BSDF_LUT_SIZE * BSDF_LUT_SIZE * BSDF_LUT_SIZE * sizeof(uint16_t)));

  __FAILURE_HANDLE(texture_create(&(*lut)->conductor));
  __FAILURE_HANDLE(texture_create(&(*lut)->specular));
  __FAILURE_HANDLE(texture_create(&(*lut)->dielectric));
  __FAILURE_HANDLE(texture_create(&(*lut)->dielectric_inv));

  __FAILURE_HANDLE(texture_fill((*lut)->conductor, BSDF_LUT_SIZE, BSDF_LUT_SIZE, 1, conductor_data, TEXTURE_DATA_TYPE_U16, 1));
  __FAILURE_HANDLE(texture_fill((*lut)->specular, BSDF_LUT_SIZE, BSDF_LUT_SIZE, 1, specular_data, TEXTURE_DATA_TYPE_U16, 1));
  __FAILURE_HANDLE(
    texture_fill((*lut)->dielectric, BSDF_LUT_SIZE, BSDF_LUT_SIZE, BSDF_LUT_SIZE, dielectric_data, TEXTURE_DATA_TYPE_U16, 1));
  __FAILURE_HANDLE(
    texture_fill((*lut)->dielectric_inv, BSDF_LUT_SIZE, BSDF_LUT_SIZE, BSDF_LUT_SIZE, dielectric_inv_data, TEXTURE_DATA_TYPE_U16, 1));

  (*lut)->conductor->wrap_mode_R = TEXTURE_WRAPPING_MODE_CLAMP;
  (*lut)->conductor->wrap_mode_S = TEXTURE_WRAPPING_MODE_CLAMP;
  (*lut)->conductor->wrap_mode_T = TEXTURE_WRAPPING_MODE_CLAMP;

  (*lut)->specular->wrap_mode_R = TEXTURE_WRAPPING_MODE_CLAMP;
  (*lut)->specular->wrap_mode_S = TEXTURE_WRAPPING_MODE_CLAMP;
  (*lut)->specular->wrap_mode_T = TEXTURE_WRAPPING_MODE_CLAMP;

  (*lut)->dielectric->wrap_mode_R = TEXTURE_WRAPPING_MODE_CLAMP;
  (*lut)->dielectric->wrap_mode_S = TEXTURE_WRAPPING_MODE_CLAMP;
  (*lut)->dielectric->wrap_mode_T = TEXTURE_WRAPPING_MODE_CLAMP;

  (*lut)->dielectric_inv->wrap_mode_R = TEXTURE_WRAPPING_MODE_CLAMP;
  (*lut)->dielectric_inv->wrap_mode_S = TEXTURE_WRAPPING_MODE_CLAMP;
  (*lut)->dielectric_inv->wrap_mode_T = TEXTURE_WRAPPING_MODE_CLAMP;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _bsdf_lut_generate_lut(BSDFLUT* lut, DeviceBSDFLUT* device_lut, Device* device) {
  __CHECK_NULL_ARGUMENT(lut);
  __CHECK_NULL_ARGUMENT(device_lut);
  __CHECK_NULL_ARGUMENT(device);

  DEVICE uint16_t* conductor_data;
  __FAILURE_HANDLE(device_malloc(&conductor_data, BSDF_LUT_SIZE * BSDF_LUT_SIZE * sizeof(uint16_t)));

  DEVICE uint16_t* specular_data;
  __FAILURE_HANDLE(device_malloc(&specular_data, BSDF_LUT_SIZE * BSDF_LUT_SIZE * sizeof(uint16_t)));

  DEVICE uint16_t* dielectric_data;
  __FAILURE_HANDLE(device_malloc(&dielectric_data, BSDF_LUT_SIZE * BSDF_LUT_SIZE * BSDF_LUT_SIZE * sizeof(uint16_t)));

  DEVICE uint16_t* dielectric_inv_data;
  __FAILURE_HANDLE(device_malloc(&dielectric_inv_data, BSDF_LUT_SIZE * BSDF_LUT_SIZE * BSDF_LUT_SIZE * sizeof(uint16_t)));

  KernelArgsBSDFGenerateSSLUT ss_lut_args;
  ss_lut_args.dst = DEVICE_PTR(conductor_data);

  KernelArgsBSDFGenerateGlossyLUT glossy_lut_args;
  glossy_lut_args.dst           = DEVICE_PTR(specular_data);
  glossy_lut_args.src_energy_ss = DEVICE_PTR(conductor_data);

  KernelArgsBSDFGenerateDielectricLUT dielectric_lut_args;
  dielectric_lut_args.dst     = DEVICE_PTR(dielectric_data);
  dielectric_lut_args.dst_inv = DEVICE_PTR(dielectric_inv_data);

  __FAILURE_HANDLE(device_sync_constant_memory(device));
  __FAILURE_HANDLE(
    kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_BSDF_GENERATE_SS_LUT], &ss_lut_args, device->stream_main));
  __FAILURE_HANDLE(
    kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_BSDF_GENERATE_GLOSSY_LUT], &glossy_lut_args, device->stream_main));
  __FAILURE_HANDLE(kernel_execute_with_args(
    device->cuda_kernels[CUDA_KERNEL_TYPE_BSDF_GENERATE_DIELECTRIC_LUT], &dielectric_lut_args, device->stream_main));

  __FAILURE_HANDLE(
    device_download(lut->conductor->data, conductor_data, 0, sizeof(uint16_t) * BSDF_LUT_SIZE * BSDF_LUT_SIZE, device->stream_main));
  __FAILURE_HANDLE(
    device_download(lut->specular->data, specular_data, 0, sizeof(uint16_t) * BSDF_LUT_SIZE * BSDF_LUT_SIZE, device->stream_main));
  __FAILURE_HANDLE(device_download(
    lut->dielectric->data, dielectric_data, 0, BSDF_LUT_SIZE * BSDF_LUT_SIZE * BSDF_LUT_SIZE * sizeof(uint16_t), device->stream_main));
  __FAILURE_HANDLE(device_download(
    lut->dielectric_inv->data, dielectric_inv_data, 0, BSDF_LUT_SIZE * BSDF_LUT_SIZE * BSDF_LUT_SIZE * sizeof(uint16_t),
    device->stream_main));

  __FAILURE_HANDLE(device_free(&conductor_data));
  __FAILURE_HANDLE(device_free(&specular_data));
  __FAILURE_HANDLE(device_free(&dielectric_data));
  __FAILURE_HANDLE(device_free(&dielectric_inv_data));

  return LUMINARY_SUCCESS;
}

DEVICE_CTX_FUNC LuminaryResult bsdf_lut_generate(BSDFLUT* lut, Device* device) {
  __CHECK_NULL_ARGUMENT(lut);
  __CHECK_NULL_ARGUMENT(device);

  DeviceBSDFLUT* device_lut = device->bsdf_lut;

  __FAILURE_HANDLE(_bsdf_lut_generate_lut(lut, device_lut, device));
  __FAILURE_HANDLE(device_bsdf_lut_update(device_lut, device, lut));

  return LUMINARY_SUCCESS;
}

LuminaryResult bsdf_lut_destroy(BSDFLUT** lut) {
  __CHECK_NULL_ARGUMENT(lut);

  if ((*lut)->conductor) {
    __FAILURE_HANDLE(texture_destroy(&(*lut)->conductor));
  }

  if ((*lut)->specular) {
    __FAILURE_HANDLE(texture_destroy(&(*lut)->specular));
  }

  if ((*lut)->dielectric) {
    __FAILURE_HANDLE(texture_destroy(&(*lut)->dielectric));
  }

  if ((*lut)->dielectric_inv) {
    __FAILURE_HANDLE(texture_destroy(&(*lut)->dielectric_inv));
  }

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

  if ((*lut)->conductor) {
    __FAILURE_HANDLE(device_texture_destroy(&(*lut)->conductor));
  }

  if ((*lut)->specular) {
    __FAILURE_HANDLE(device_texture_destroy(&(*lut)->specular));
  }

  if ((*lut)->dielectric) {
    __FAILURE_HANDLE(device_texture_destroy(&(*lut)->dielectric));
  }

  if ((*lut)->dielectric_inv) {
    __FAILURE_HANDLE(device_texture_destroy(&(*lut)->dielectric_inv));
  }

  __FAILURE_HANDLE(host_free(lut));

  return LUMINARY_SUCCESS;
}
