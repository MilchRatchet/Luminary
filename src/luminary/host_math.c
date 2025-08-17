#include "host_math.h"

#define _USE_MATH_DEFINES
#include <math.h>

Quaternion rotation_euler_angles_to_quaternion(vec3 rotation) {
  const float cr = cosf(rotation.x * 0.5f);
  const float sr = sinf(rotation.x * 0.5f);
  const float cp = cosf(rotation.y * 0.5f);
  const float sp = sinf(rotation.y * 0.5f);
  const float cy = cosf(rotation.z * 0.5f);
  const float sy = sinf(rotation.z * 0.5f);

  Quaternion q;
  q.w = cr * cp * cy + sr * sp * sy;
  q.x = sr * cp * cy - cr * sp * sy;
  q.y = cr * sp * cy + sr * cp * sy;
  q.z = cr * cp * sy - sr * sp * cy;

  return q;
}

vec3 rotation_quaternion_to_euler_angles(Quaternion q) {
  vec3 angles;

  const float sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
  const float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
  angles.x              = atan2f(sinr_cosp, cosr_cosp);

  const float sinp = sqrtf(1.0f + 2.0f * (q.w * q.y - q.x * q.z));
  const float cosp = sqrtf(1.0f - 2.0f * (q.w * q.y - q.x * q.z));
  angles.y         = 2.0f * atan2f(sinp, cosp) - M_PI / 2.0f;

  // yaw (z-axis rotation)
  const float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
  const float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
  angles.z              = atan2f(siny_cosp, cosy_cosp);

  return angles;
}
