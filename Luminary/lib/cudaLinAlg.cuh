#ifndef CUDALINALG_H
#define CUDALINALG_H

#include "primitives.h"

__device__ vec3 cross_product(const vec3 a, const vec3 b);
__device__ float dot_product(const vec3 a, const vec3 b);
__device__ vec3 vec_diff(const vec3 a, const vec3 b);

#endif /* CUDALINALG_H */
