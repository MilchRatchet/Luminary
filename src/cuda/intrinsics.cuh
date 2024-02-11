#ifndef CU_INTRINSICS_H
#define CU_INTRINSICS_H

#include "utils.cuh"

LUM_DEVICE_FUNC unsigned int sign_extend_s8x4(unsigned int a) {
  unsigned int result;
  asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(result) : "r"(a));
  return result;
}

LUM_DEVICE_FUNC unsigned int __bfind(unsigned int a) {
  unsigned int result;
  asm("bfind.u32 %0, %1; " : "=r"(result) : "r"(a));
  return result;
}

/*
 * Semantic:
 * __slct(a,b,c) = (c >= 0) ? a : b;
 */
LUM_DEVICE_FUNC float __fslctf(const float a, const float b, const float c) {
  float result;
  asm("slct.f32.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
  return result;
}

/*
 * Semantic:
 * ____uswap16p(a) = (a >> 16) | (a << 16);
 */
LUM_DEVICE_FUNC uint32_t __uswap16p(const uint32_t a) {
  uint32_t result;
  asm("prmt.b32 %0, %1, %2, 0b0001000000110010;" : "=r"(result) : "r"(a), "r"(a));
  return result;
}

/*
 * Semantic:
 * __slct(a,b,c) = (c >= 0) ? a : b;
 */
LUM_DEVICE_FUNC unsigned int __uslctf(const unsigned int a, const unsigned int b, const float c) {
  unsigned int result;
  asm("slct.u32.f32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "f"(c));
  return result;
}

LUM_DEVICE_FUNC int __min_min(const int a, const int b, const int c) {
  int v;
  asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
  return v;
}

LUM_DEVICE_FUNC int __min_max(const int a, const int b, const int c) {
  int v;
  asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
  return v;
}

LUM_DEVICE_FUNC int __max_min(const int a, const int b, const int c) {
  int v;
  asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
  return v;
}

LUM_DEVICE_FUNC int __max_max(const int a, const int b, const int c) {
  int v;
  asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
  return v;
}

/*
 * Semantic:
 * __fmin_fmin(a,b,c) = fminf(a, fminf(b,c));
 *
 * Note that this runs on the ALU on Ampere.
 */
LUM_DEVICE_FUNC float __fmin_fmin(const float a, const float b, const float c) {
  return __int_as_float(__min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

LUM_DEVICE_FUNC float __fmin_fmax(const float a, const float b, const float c) {
  return __int_as_float(__min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

LUM_DEVICE_FUNC float __fmax_fmin(const float a, const float b, const float c) {
  return __int_as_float(__max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

/*
 * Semantic:
 * __fmax_fmax(a,b,c) = fmaxf(a, fmaxf(b,c));
 *
 * Note that this runs on the ALU on Ampere.
 */
LUM_DEVICE_FUNC float __fmax_fmax(const float a, const float b, const float c) {
  return __int_as_float(__max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

#endif /* CU_INTRINSICS_H */
