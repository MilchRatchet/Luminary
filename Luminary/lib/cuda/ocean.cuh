/*
 * Wave generation based on a shadertoy by Alexander Alekseev aka TDM (2014)
 * The shadertoy can be found on https://www.shadertoy.com/view/Ms2SD1
 */
#ifndef CU_OCEAN_H
#define CU_OCEAN_H

#include "math.cuh"

#define FAST_ITERATIONS 3
#define SLOW_ITERATIONS 5

__device__ float ocean_hash(const UV uv) {
  const float x = uv.u * 127.1f + uv.v * 311.7f;
  return fractf(sinf(x) * 43758.5453123f);
}

__device__ float ocean_noise(const UV uv) {
  UV integral;
  integral.u = floorf(uv.u);
  integral.v = floorf(uv.v);

  UV fractional;
  fractional.u = fractf(uv.u);
  fractional.v = fractf(uv.v);

  fractional.u *= fractional.u * (3.0f - 2.0f * fractional.u);
  fractional.v *= fractional.v * (3.0f - 2.0f * fractional.v);

  const float hash1 = ocean_hash(integral);
  integral.u += 1.0f;
  const float hash2 = ocean_hash(integral);
  integral.v += 1.0f;
  const float hash4 = ocean_hash(integral);
  integral.u -= 1.0f;
  const float hash3 = ocean_hash(integral);

  const float a = lerp(hash1, hash2, fractional.u);
  const float b = lerp(hash3, hash4, fractional.u);

  return -1.0f + 2.0f * lerp(a, b, fractional.v);
}

__device__ float ocean_octave(UV uv, const float choppyness) {
  const float offset = ocean_noise(uv);
  uv.u += offset;
  uv.v += offset;

  UV wave1;
  wave1.u = 1.0f - fabsf(sinf(uv.u));
  wave1.v = 1.0f - fabsf(sinf(uv.v));

  UV wave2;
  wave2.u = fabsf(cosf(uv.u));
  wave2.v = fabsf(cosf(uv.v));

  wave1.u = lerp(wave1.u, wave2.u, wave1.u);
  wave1.v = lerp(wave1.v, wave2.v, wave1.v);

  return powf(1.0f - powf(wave1.u * wave1.v, 0.65f), choppyness);
}

__device__ float get_ocean_height(const vec3 p, const int steps) {
  float amplitude  = device_scene.ocean.amplitude;
  float choppyness = device_scene.ocean.choppyness;
  float frequency  = device_scene.ocean.frequency;

  UV uv;
  uv.u = p.x * 0.75f;
  uv.v = p.z;

  float d = 0.0f;
  float h = 0.0f;

  float t = 1.0f + device_scene.ocean.time * device_scene.ocean.speed;

  for (int i = 0; i < steps; i++) {
    UV temp;
    temp.u = (uv.u + t) * frequency;
    temp.v = (uv.v + t) * frequency;
    d      = ocean_octave(temp, choppyness);

    temp.u = (uv.u - t) * frequency;
    temp.v = (uv.v - t) * frequency;
    d += ocean_octave(temp, choppyness);

    h += d * amplitude;

    const float u = uv.u;
    const float v = uv.v;
    uv.u          = 1.6f * u + -1.2f * v;
    uv.v          = 1.2f * u + 1.6f * v;

    frequency *= 1.9f;
    amplitude *= 0.22f;
    choppyness = lerp(choppyness, 1.0f, 0.2f);
  }

  return p.y - h - device_scene.ocean.height;
}

__device__ vec3 get_ocean_normal(vec3 p, const float diff) {
  vec3 normal;
  normal.y = get_ocean_height(p, SLOW_ITERATIONS);
  p.x += diff;
  normal.x = get_ocean_height(p, SLOW_ITERATIONS) - normal.y;
  p.x -= diff;
  p.z += diff;
  normal.z = get_ocean_height(p, SLOW_ITERATIONS) - normal.y;
  normal.y = diff;

  return normalize_vector(normal);
}

__device__ float get_intersection_ocean(const vec3 origin, const vec3 ray, float max) {
  float min = 0.0f;

  vec3 p;
  p.x = origin.x + max * ray.x;
  p.y = origin.y + max * ray.y;
  p.z = origin.z + max * ray.z;

  float height_at_max = get_ocean_height(p, FAST_ITERATIONS);
  if (height_at_max > 0.0f)
    return FLT_MAX;

  float height_at_min = get_ocean_height(origin, FAST_ITERATIONS);
  if (height_at_min < 0.0f)
    return FLT_MAX;

  float mid = 0.0f;

  for (int i = 0; i < 8; i++) {
    mid = lerp(min, max, height_at_min / (height_at_min - height_at_max));
    p.x = origin.x + mid * ray.x;
    p.y = origin.y + mid * ray.y;
    p.z = origin.z + mid * ray.z;

    float height_at_mid = get_ocean_height(p, FAST_ITERATIONS);

    if (height_at_mid < 0.0f) {
      max           = mid;
      height_at_max = height_at_mid;
    }
    else {
      min           = mid;
      height_at_min = height_at_mid;
    }
  }

  return mid;
}

#endif /* CU_OCEAN_H */
