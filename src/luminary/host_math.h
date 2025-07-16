#ifndef LUMINARY_HOST_MATH_H
#define LUMINARY_HOST_MATH_H

#include "utils.h"

Quaternion rotation_euler_angles_to_quaternion(vec3 rotation);
vec3 rotation_quaternion_to_euler_angles(Quaternion q);

#endif /* LUMINARY_HOST_MATH_H */
