#ifndef CU_LUMINARY_MATERIAL_H
#define CU_LUMINARY_MATERIAL_H

#include "math.cuh"
#include "utils.cuh"

enum MaterialType { MATERIAL_GEOMETRY, MATERIAL_VOLUME, MATERIAL_PARTICLE } typedef MaterialType;

template <MaterialType TYPE>
struct MaterialContext {};

template <>
struct MaterialContext<MATERIAL_GEOMETRY> {
  uint32_t instance_id;
  uint32_t tri_id;
  RGBAF albedo;
  RGBF emission;
  vec3 position;
  vec3 V;
  vec3 normal;
  float roughness;
  uint16_t state;
  uint8_t flags;
  /* IOR of medium in direction of V. */
  float ior_in;
  /* IOR of medium on the other side. */
  float ior_out;
};

template <>
struct MaterialContext<MATERIAL_VOLUME> {
  VolumeDescriptor descriptor;
  vec3 position;
  vec3 V;
  uint16_t state;
};

template <>
struct MaterialContext<MATERIAL_PARTICLE> {
  uint32_t particle_id;
  vec3 position;
  vec3 V;
  vec3 normal;
  uint16_t state;
  uint8_t flags;
};

typedef MaterialContext<MATERIAL_GEOMETRY> MaterialContextGeometry;
typedef MaterialContext<MATERIAL_VOLUME> MaterialContextVolume;
typedef MaterialContext<MATERIAL_PARTICLE> MaterialContextParticle;

__device__ MaterialContextGeometry material_get_default_context() {
  MaterialContextGeometry ctx;

  ctx.instance_id = HIT_TYPE_INVALID;
  ctx.tri_id      = 0;
  ctx.albedo      = get_RGBAF(1.0f, 1.0f, 1.0f, 1.0f);
  ctx.emission    = get_color(0.0f, 0.0f, 0.0f);
  ctx.normal      = get_vector(0.0f, 0.0f, 1.0f);
  ctx.position    = get_vector(0.0f, 0.0f, 0.0f);
  ctx.V           = get_vector(0.0f, 0.0f, 1.0f);
  ctx.roughness   = 0.5f;
  ctx.state       = 0;
  ctx.flags       = 0;
  ctx.ior_in      = 1.0f;
  ctx.ior_out     = 1.0f;

  return ctx;
}

#endif /* CU_LUMINARY_MATERIAL_H */
