#ifndef CU_MINMAX_H
#define CU_MINMAX_H

__device__
int min_min(const int a, const int b, const int c) {
    int v;
    asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
    return v;
}

__device__
int min_max(const int a, const int b, const int c) {
    int v;
    asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
    return v;
}

__device__
int max_min(const int a, const int b, const int c) {
    int v;
    asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
    return v;
}

__device__
int max_max(const int a, const int b, const int c) {
    int v;
    asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
    return v;
}

__device__
float fmin_fmin(const float a, const float b, const float c) {
    return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

__device__
float fmin_fmax(const float a, const float b, const float c) {
    return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

__device__
float fmax_fmin(const float a, const float b, const float c) {
    return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

__device__
float fmax_fmax(const float a, const float b, const float c) {
    return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

__device__
float max7(const float a0, const float a1, const float b0, const float b1, const float c0, const float c1, const float d)
{
	const float t1 = fmin_fmax(a0, a1, d);
	const float t2 = fmin_fmax(b0, b1, t1);
	const float t3 = fmin_fmax(c0, c1, t2);
	return t3;
}

__device__
float min7(const float a0, const float a1, const float b0, const float b1, const float c0, const float c1, const float d)
{
	const float t1 = fmax_fmin(a0, a1, d);
	const float t2 = fmax_fmin(b0, b1, t1);
	const float t3 = fmax_fmin(c0, c1, t2);
	return t3;
}

#endif /* CU_MINMAX_H */
