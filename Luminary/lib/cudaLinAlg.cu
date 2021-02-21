#include "primitives.h"
#include <cuda_runtime_api.h>

__device__
vec3 cross_product(const vec3 a, const vec3 b) {
    vec3 result;

    result.x = a.y*b.z - a.z*b.y;
    result.y = a.z*b.x - a.x*b.z;
    result.z = a.x*b.y - a.y*b.x;

    return result;
}

__device__
float dot_product(const vec3 a, const vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
vec3 vec_diff(const vec3 a, const vec3 b) {
    vec3 result;

    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;

    return result;
}
