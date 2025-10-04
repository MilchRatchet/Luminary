#ifndef LUMINARY_HOST_INTRINSICS_H
#define LUMINARY_HOST_INTRINSICS_H

#include <float.h>
#include <math.h>
#include <stdbool.h>

#ifdef __x86_64__
#include <immintrin.h>
#define LUMINARY_X86_INTRINSICS
#endif /* __x86_64__ */

struct Vec128 {
  union {
    struct {
      float x;
      float y;
      float z;
      float w;
    };
    float data[4];
#ifdef LUMINARY_X86_INTRINSICS
    __m128 _imm;
    __m128i _immi;
#endif
  };
} typedef Vec128;
_STATIC_ASSERT(sizeof(Vec128) == 16);

#ifndef __cplusplus

inline Vec128 vec128_set_1(const float a) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128) {._imm = _mm_set1_ps(a)};
#else
  return (Vec128) {.x = a, .y = a, .z = a, .w = a};
#endif
}

inline Vec128 vec128_set(const float x, const float y, const float z, const float w) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128) {._imm = _mm_set_ps(w, z, y, x)};
#else
  return (Vec128) {.x = x, .y = y, .z = z, .w = w};
#endif
}

inline bool vec128_is_equal(const Vec128 a, const Vec128 b) {
#ifdef LUMINARY_X86_INTRINSICS
  const __m128i diff = _mm_xor_si128(a._immi, b._immi);
  return _mm_testz_si128(diff, diff);
#else
  return memcmp(&a, &b, sizeof(Vec128)) == 0;
#endif
}

inline Vec128 vec128_add(const Vec128 a, const Vec128 b) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128) {._imm = _mm_add_ps(a._imm, b._imm)};
#else
  return (Vec128) {.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z, .w = a.w + b.w};
#endif
}

inline Vec128 vec128_sub(const Vec128 a, const Vec128 b) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128) {._imm = _mm_sub_ps(a._imm, b._imm)};
#else
  return (Vec128) {.x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z, .w = a.w - b.w};
#endif
}

inline Vec128 vec128_mul(const Vec128 a, const Vec128 b) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128) {._imm = _mm_mul_ps(a._imm, b._imm)};
#else
  return (Vec128) {.x = a.x * b.x, .y = a.y * b.y, .z = a.z * b.z, .w = a.w * b.w};
#endif
}

inline Vec128 vec128_scale(const Vec128 a, const float b) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128) {._imm = _mm_mul_ps(a._imm, vec128_set_1(b)._imm)};
#else
  return (Vec128) {.x = a.x * b, .y = a.y * b, .z = a.z * b, .w = a.w * b};
#endif
}

inline Vec128 vec128_fmadd(const Vec128 a, const Vec128 b, const Vec128 c) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128) {._imm = _mm_fmadd_ps(a._imm, b._imm, c._imm)};
#else
  return (Vec128) {.x = a.x * b.x + c.x, .y = a.y * b.y + c.y, .z = a.z * b.z + c.z, .w = a.w * b.w + c.w};
#endif
}

inline Vec128 vec128_cross(const Vec128 a, const Vec128 b) {
  // TODO: Implement this with shuffles.
  return (Vec128) {.x = a.y * b.z - a.z * b.y, .y = a.z * b.x - a.x * b.z, .z = a.x * b.y - a.y * b.x, .w = 0.0f};
}

inline float vec128_hsum(const Vec128 a) {
#ifdef LUMINARY_X86_INTRINSICS
  const __m128 m1 = _mm_shuffle_ps(a._imm, a._imm, _MM_SHUFFLE(0, 0, 3, 2));
  const __m128 m2 = _mm_add_ps(a._imm, m1);
  const __m128 m3 = _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(0, 0, 0, 1));
  const __m128 m4 = _mm_add_ps(m2, m3);
  return _mm_cvtss_f32(m4);
#else
  return a.x + a.y + a.z + a.w;
#endif
}

inline Vec128 vec128_min(const Vec128 a, const Vec128 b) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128) {._imm = _mm_min_ps(a._imm, b._imm)};
#else
  return (Vec128) {.x = fminf(a.x, b.x), .y = fminf(a.y, b.y), .z = fminf(a.z, b.z), .w = fminf(a.w, b.w)};
#endif
}

inline Vec128 vec128_max(const Vec128 a, const Vec128 b) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128) {._imm = _mm_max_ps(a._imm, b._imm)};
#else
  return (Vec128) {.x = fmaxf(a.x, b.x), .y = fmaxf(a.y, b.y), .z = fmaxf(a.z, b.z), .w = fmaxf(a.w, b.w)};
#endif
}

inline Vec128 vec128_load(const float* ptr) {
#ifdef LUMINARY_X86_INTRINSICS
  return (Vec128) {._imm = _mm_loadu_ps(ptr)};
#else
  return (Vec128) {.x = ptr[0], .y = ptr[1], .z = ptr[2], .w = ptr[3]};
#endif
}

inline void vec128_store(float* ptr, const Vec128 a) {
#ifdef LUMINARY_X86_INTRINSICS
  _mm_storeu_ps(ptr, a._imm);
#else
  ptr[0] = a.x;
  ptr[1] = a.y;
  ptr[2] = a.z;
  ptr[3] = a.w;
#endif
}

/*
 * Implicitely assumes that index < 4.
 */
#ifdef LUMINARY_X86_INTRINSICS
#define vec128_get_1(a, index) _mm_cvtss_f32(_mm_shuffle_ps(a._imm, a._imm, _MM_SHUFFLE(0, 0, 0, index)))
#else
inline float vec128_get_1(const Vec128 a, const uint32_t index) {
  return a.data[index];
}
#endif

inline Vec128 vec128_set_w_to_0(const Vec128 a) {
#ifdef LUMINARY_X86_INTRINSICS
  Vec128 zero;
  zero._imm = _mm_xor_ps(zero._imm, zero._imm);
  return (Vec128) {._imm = _mm_blend_ps(a._imm, zero._imm, 0b1000)};
#else
  return (Vec128) {.x = a.x, .y = a.y, .z = a.z, .w = 0.0f};
#endif
}

inline float vec128_hmax(const Vec128 a) {
#ifdef LUMINARY_X86_INTRINSICS
  const __m128 m1 = _mm_shuffle_ps(a._imm, a._imm, _MM_SHUFFLE(0, 0, 3, 2));
  const __m128 m2 = _mm_max_ps(a._imm, m1);
  const __m128 m3 = _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(0, 0, 0, 1));
  const __m128 m4 = _mm_max_ps(m2, m3);
  return _mm_cvtss_f32(m4);
#else
  return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
#endif
}

inline float vec128_norm2(const Vec128 a) {
  return sqrtf(vec128_hsum(vec128_mul(a, a)));
}

inline float vec128_dot(const Vec128 a, const Vec128 b) {
  return vec128_hsum(vec128_mul(a, b));
}

/*
 * Implicitely assumes w is 0. Call vec128_set_w_to_0 first if that is not given.
 */
inline float vec128_box_area(const Vec128 a) {
#ifdef LUMINARY_X86_INTRINSICS
  const Vec128 b = (Vec128) {._imm = _mm_shuffle_ps(a._imm, a._imm, _MM_SHUFFLE(3, 1, 0, 0))};
  const Vec128 c = (Vec128) {._imm = _mm_shuffle_ps(a._imm, a._imm, _MM_SHUFFLE(3, 2, 2, 1))};
  return vec128_hsum(vec128_mul(b, c));
#else
  return a.x * a.y + a.x * a.z + a.y * a.z;
#endif
}

/*
 * Rotate a 3 component vector given by a using the quaternion given by q.
 */
inline Vec128 vec128_rotate_quaternion(const Vec128 a, const Vec128 q) {
  const float dot_qa = a.x * q.x + a.y * q.y + a.z * q.z;
  const float dot_qq = q.x * q.x + q.y * q.y + q.z * q.z;

  const Vec128 cross = vec128_cross(q, a);

  Vec128 result = vec128_scale(q, 2.0f * dot_qa);
  result        = vec128_add(result, vec128_scale(a, q.w * q.w - dot_qq));
  result        = vec128_add(result, vec128_scale(cross, 2.0f * q.w));

  return result;
}

#endif /* !__cplusplus */

#endif /* LUMINARY_HOST_INTRINSICS_H */
