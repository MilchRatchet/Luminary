#ifndef LUMINARY_HOST_INTRINSICS_H
#define LUMINARY_HOST_INTRINSICS_H

#include <immintrin.h>
#include <stdbool.h>

#define LUMINARY_X86_INTRINSICS

struct Vec128 {
  union {
    struct {
      float x;
      float y;
      float z;
      float w;
    };
    float data[4];
    __m128 _imm;
    __m128i _immi;
  };
} typedef Vec128;
_STATIC_ASSERT(sizeof(Vec128) == 16);

inline Vec128 vec128_set_1(const float a) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128){._imm = _mm_set1_ps(a)};
#else
  return (Vec128){.x = a, .y = a, .z = a, .w = a};
#endif
}

inline Vec128 vec128_set(const float x, const float y, const float z, const float w) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128){._imm = _mm_set_ps(x, y, z, w)};
#else
  return (Vec128){.x = x, .y = y, .z = z, .w = w};
#endif
}

inline bool vec128_is_equal(const Vec128 a, const Vec128 b) {
#ifdef LUMINARY_X86_INTRINSICS
  return _mm_cmpistrc(a._immi, b._immi, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_NEGATIVE_POLARITY);
#else
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
#endif
}

inline Vec128 vec128_add(const Vec128 a, const Vec128 b) {
  return (Vec128){.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z, .w = a.w + b.w};
}

inline Vec128 vec128_mul(const Vec128 a, const Vec128 b) {
  return (Vec128){.x = a.x * b.x, .y = a.y * b.y, .z = a.z * b.z, .w = a.w * b.w};
}

inline Vec128 vec128_cross(const Vec128 a, const Vec128 b) {
  return (Vec128){.x = a.y * b.z - a.z * b.y, .y = a.z * b.x - a.x * b.z, .z = a.x * b.y - a.y * b.x, .w = 0.0f};
}

inline float vec128_hsum(const Vec128 a) {
  return a.x + a.y + a.z + a.w;
}

inline Vec128 vec128_min(const Vec128 a, const Vec128 b) {
  return (Vec128){.x = fminf(a.x, b.x), .y = fminf(a.y, b.y), .z = fminf(a.z, b.z), .w = fminf(a.w, b.w)};
}

inline Vec128 vec128_max(const Vec128 a, const Vec128 b) {
  return (Vec128){.x = fmaxf(a.x, b.x), .y = fmaxf(a.y, b.y), .z = fmaxf(a.z, b.z), .w = fmaxf(a.w, b.w)};
}

#endif /* LUMINARY_HOST_INTRINSICS_H */
