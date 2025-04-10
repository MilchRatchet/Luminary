#ifndef LUMINARY_DEVICE_PACKING_H
#define LUMINARY_DEVICE_PACKING_H

#include "device_utils.h"

uint32_t device_pack_normal(const vec3 normal);
uint32_t device_pack_uv(const UV uv);

#endif /* LUMINARY_DEVICE_PACKING_H */