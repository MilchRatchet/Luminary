#include "device_packing.h"

#include <math.h>

// Octahedron encoding, for example: https://www.shadertoy.com/view/clXXD8
uint32_t device_pack_normal(const vec3 normal) {
  double x = normal.x;
  double y = normal.y;
  double z = normal.z;

  const double recip_norm = 1.0 / (fabs(x) + fabs(y) + fabs(z));

  x *= recip_norm;
  y *= recip_norm;
  z *= recip_norm;

  const double t = fmax(fmin(-z, 1.0f), 0.0f);

  x += (x >= 0.0) ? t : -t;
  y += (y >= 0.0) ? t : -t;

  x = fmax(fmin(x, 1.0), -1.0);
  y = fmax(fmin(y, 1.0), -1.0);

  x = (x + 1.0) * 0.5;
  y = (y + 1.0) * 0.5;

  const uint32_t x_u16 = (uint32_t) (x * 0xFFFF + 0.5);
  const uint32_t y_u16 = (uint32_t) (y * 0xFFFF + 0.5);

  return (y_u16 << 16) | x_u16;
}

/*
 * Each component gets 1 sign bit, 8 exponent bit and 7 mantissa bits.
 */
uint32_t device_pack_uv(const UV uv) {
  const uint32_t u = *((uint32_t*) (&uv.u));
  const uint32_t v = *((uint32_t*) (&uv.v));

  const uint32_t compressed = (u & 0xFFFF0000) | (v >> 16);

  return compressed;
}

uint16_t device_pack_float(const float val, const DevicePackFloatRoundingMode mode) {
  union {
    float a;
    uint32_t b;
  } converter;
  converter.a = val;

  switch (mode) {
    case DEVICE_PACK_FLOAT_ROUNDING_MODE_ROUND:
      converter.b += (1 << 15);
      break;
    case DEVICE_PACK_FLOAT_ROUNDING_MODE_CEIL:
      if (val >= 0.0f)
        converter.b += (1 << 16) - 1;
      break;
    case DEVICE_PACK_FLOAT_ROUNDING_MODE_FLOOR:
      if (val < 0.0f)
        converter.b += (1 << 16) - 1;
      break;
    default:
      break;
  }

  return (converter.b >> 16);
}

float device_unpack_float(const uint16_t val) {
  union {
    float a;
    uint32_t b;
  } converter;
  converter.b = val;

  converter.b <<= 16;

  return converter.a;
}
