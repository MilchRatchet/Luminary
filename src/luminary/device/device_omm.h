#ifndef LUMINARY_DEVICE_OMM_H
#define LUMINARY_DEVICE_OMM_H

#include "device_utils.h"

struct Device typedef Device;

struct OpacityMicromap {
  OptixBuildInputOpacityMicromap optix_build_input;
  DEVICE void* buffer;
} typedef OpacityMicromap;

LuminaryResult omm_create(OpacityMicromap** omm);
DEVICE_CTX_FUNC LuminaryResult omm_build(OpacityMicromap* omm, Device* device, const DeviceMesh* mesh);
LuminaryResult omm_destroy(OpacityMicromap** omm);

#endif /* LUMINARY_DEVICE_OMM_H */
