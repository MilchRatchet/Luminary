#ifndef LUMINARY_DEVICE_PACKING_H
#define LUMINARY_DEVICE_PACKING_H

#include "device_utils.h"

enum DevicePackFloatRoundingMode {
  DEVICE_PACK_FLOAT_ROUNDING_MODE_ROUND,
  DEVICE_PACK_FLOAT_ROUNDING_MODE_CEIL,
  DEVICE_PACK_FLOAT_ROUNDING_MODE_FLOOR
} typedef DevicePackFloatRoundingMode;

uint32_t device_pack_normal(const vec3 normal);
uint32_t device_pack_uv(const UV uv);
uint16_t device_pack_float(const float val, const DevicePackFloatRoundingMode mode);
float device_unpack_float(const uint16_t val);

#endif /* LUMINARY_DEVICE_PACKING_H */
