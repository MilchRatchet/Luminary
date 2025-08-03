#ifndef CU_LUMINARY_MATERIAL_H
#define CU_LUMINARY_MATERIAL_H

#include "math.cuh"
#include "utils.cuh"

enum MaterialType { MATERIAL_GEOMETRY, MATERIAL_VOLUME, MATERIAL_PARTICLE } typedef MaterialType;

enum MaterialParamType {
  MATERIAL_PARAM_TYPE_NORM_FLOAT,         // [0,1]
  MATERIAL_PARAM_TYPE_SIGNED_NORM_FLOAT,  // [-1,1]
  MATERIAL_PARAM_TYPE_IOR,                // [1,3]
  MATERIAL_PARAM_TYPE_ABBE,               // [9,91]
  MATERIAL_PARAM_TYPE_NORM_COLOR,         // [0,1]^3
  MATERIAL_PARAM_TYPE_COLOR,              // [0,1000]^3
} typedef MaterialParamType;

#define MATERIAL_GEOMETRY_PARAM_ALLOCATE(__name, __type, __bitcount)                                                                    \
  MATERIAL_GEOMETRY_PARAM_ALLOCATION_START_##__name,                                                                                    \
    MATERIAL_GEOMETRY_PARAM_ALLOCATION_SIZE_##__name = (__bitcount),                                                                    \
    MATERIAL_GEOMETRY_PARAM_##__name =                                                                                                  \
      MATERIAL_GEOMETRY_PARAM_ALLOCATION_START_##__name + 0x100u * MATERIAL_GEOMETRY_PARAM_ALLOCATION_SIZE_##__name + 0x10000 * __type, \
    MATERIAL_GEOMETRY_PARAM_ALLOCATION_END_##__name =                                                                                   \
      MATERIAL_GEOMETRY_PARAM_ALLOCATION_START_##__name + MATERIAL_GEOMETRY_PARAM_ALLOCATION_SIZE_##__name - 1,

#define MATERIAL_GEOMETRY_PARAM_GET_OFFSET(__param) (__param & 0xFF)
#define MATERIAL_GEOMETRY_PARAM_GET_SIZE(__param) ((__param >> 8) & 0xFF)
#define MATERIAL_GEOMETRY_PARAM_GET_TYPE(__param) ((MaterialParamType) ((__param >> 16) & 0xFF))

// Fixed sizes
#define MATERIAL_PARAM_TYPE_NORM_COLOR_SIZE 30
#define MATERIAL_PARAM_TYPE_COLOR_SIZE 32

enum MaterialGeometryParam : uint32_t {
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(ALBEDO, MATERIAL_PARAM_TYPE_NORM_COLOR, MATERIAL_PARAM_TYPE_NORM_COLOR_SIZE)  //
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(OPACITY, MATERIAL_PARAM_TYPE_NORM_FLOAT, 8)                                   //
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(EMISSION, MATERIAL_PARAM_TYPE_COLOR, MATERIAL_PARAM_TYPE_COLOR_SIZE)          //
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(ROUGHNESS, MATERIAL_PARAM_TYPE_NORM_FLOAT, 10)                                //
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(IOR_IN, MATERIAL_PARAM_TYPE_IOR, 8)                                           //
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(IOR_OUT, MATERIAL_PARAM_TYPE_IOR, 8)                                          //

  MATERIAL_GEOMETRY_PARAM_BITS_COUNT
} typedef MaterialGeometryParam;

#define MATERIAL_GEOM_NUM_UINTS ((MATERIAL_GEOMETRY_PARAM_BITS_COUNT + ((1 << 5) - 1)) >> 5)

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
  uint16_t state;
  uint8_t flags;
  /* IOR of medium in direction of V. */
  float ior_in;
  /* IOR of medium on the other side. */
  float ior_out;
  VolumeType volume_type;

  uint32_t data[MATERIAL_GEOM_NUM_UINTS];
};

template <>
struct MaterialContext<MATERIAL_VOLUME> {
  VolumeDescriptor descriptor;
  vec3 position;
  vec3 V;
  uint16_t state;
  VolumeType volume_type;
};

template <>
struct MaterialContext<MATERIAL_PARTICLE> {
  uint32_t particle_id;
  vec3 position;
  vec3 V;
  vec3 normal;
  uint16_t state;
  uint8_t flags;
  VolumeType volume_type;
};

typedef MaterialContext<MATERIAL_GEOMETRY> MaterialContextGeometry;
typedef MaterialContext<MATERIAL_VOLUME> MaterialContextVolume;
typedef MaterialContext<MATERIAL_PARTICLE> MaterialContextParticle;

template <MaterialGeometryParam PARAM>
__device__ uint32_t material_param_get_data(const MaterialContext<MATERIAL_GEOMETRY>& ctx) {
  constexpr uint32_t OFFSET = MATERIAL_GEOMETRY_PARAM_GET_OFFSET(PARAM);
  constexpr uint32_t SIZE   = MATERIAL_GEOMETRY_PARAM_GET_SIZE(PARAM);

  constexpr uint32_t first_index = OFFSET >> 5;
  constexpr uint32_t first_mask  = (SIZE < 32u) ? ((1u << SIZE) - 1u) : 0xFFFFFFFFu;
  constexpr uint32_t first_shift = OFFSET & 0x1Fu;

  uint32_t result = (ctx.data[first_index] >> first_shift) & first_mask;

  if constexpr (((OFFSET & 0x1Fu) + SIZE) > 0x20u) {
    constexpr uint32_t second_index = first_index + 1u;
    constexpr uint32_t second_mask  = (1u << (((OFFSET & 0x1Fu) + SIZE) - 5u)) - 1u;
    constexpr uint32_t second_shift = 32u - (OFFSET & 0x1Fu);

    result |= (ctx.data[second_index] & second_mask) << second_shift;
  }

  return result;
}

template <MaterialGeometryParam PARAM>
__device__ void material_param_set_data(MaterialContext<MATERIAL_GEOMETRY>& ctx, const uint32_t value) {
  constexpr uint32_t OFFSET = MATERIAL_GEOMETRY_PARAM_GET_OFFSET(PARAM);
  constexpr uint32_t SIZE   = MATERIAL_GEOMETRY_PARAM_GET_SIZE(PARAM);

  constexpr uint32_t first_index    = OFFSET >> 5;
  constexpr uint32_t first_src_mask = (SIZE < 32u) ? ((1u << SIZE) - 1u) : 0xFFFFFFFFu;
  constexpr uint32_t first_shift    = OFFSET & 0x1Fu;

  ctx.data[first_index] = (ctx.data[first_index] & ~(first_src_mask << first_shift)) | ((value & first_src_mask) << first_shift);

  if constexpr (((OFFSET & 0x1Fu) + SIZE) > 0x20u) {
    constexpr uint32_t second_index = first_index + 1u;
    constexpr uint32_t second_shift = 32u - (OFFSET & 0x1Fu);

    ctx.data[second_index] = (ctx.data[second_index] & ~(first_src_mask >> second_shift)) | ((value & first_src_mask) >> second_shift);
  }
}

template <MaterialGeometryParam PARAM>
__device__ float material_get_float(const MaterialContext<MATERIAL_GEOMETRY>& ctx) {
  static_assert(
    MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) != MATERIAL_PARAM_TYPE_NORM_COLOR
      && MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) != MATERIAL_PARAM_TYPE_COLOR,
    "PARAM must not be color type.");

  constexpr uint32_t SIZE          = MATERIAL_GEOMETRY_PARAM_GET_SIZE(PARAM);
  constexpr MaterialParamType TYPE = MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM);

  const uint32_t data = material_param_get_data<PARAM>(ctx);

  float result = data * (1.0f / ((1u << SIZE) - 1));

  if constexpr (TYPE == MATERIAL_PARAM_TYPE_NORM_FLOAT) {
    // Nothing
  }
  else if constexpr (TYPE == MATERIAL_PARAM_TYPE_SIGNED_NORM_FLOAT) {
    result = result * 2.0f - 1.0f;
  }
  else if constexpr (TYPE == MATERIAL_PARAM_TYPE_IOR) {
    result = result * 2.0f + 1.0f;
  }
  else if constexpr (TYPE == MATERIAL_PARAM_TYPE_ABBE) {
    result = result * 82.0f + 9.0f;
  }

  return result;
}

template <MaterialGeometryParam PARAM>
__device__ RGBF material_get_color(const MaterialContext<MATERIAL_GEOMETRY>& ctx) {
  static_assert(
    MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) == MATERIAL_PARAM_TYPE_NORM_COLOR
      || MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) == MATERIAL_PARAM_TYPE_COLOR,
    "PARAM must be color type.");

  const uint32_t data = material_param_get_data<PARAM>(ctx);

  RGBF result;
  if constexpr (MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) == MATERIAL_PARAM_TYPE_NORM_COLOR) {
    const uint32_t data_red   = data & 0x3FF;
    const uint32_t data_green = (data >> 10) & 0x3FF;
    const uint32_t data_blue  = data >> 20;

    result.r = data_red * (1.0f / 0x3FF);
    result.g = data_green * (1.0f / 0x3FF);
    result.b = data_blue * (1.0f / 0x3FF);
  }
  else {
  }

  return result;
}

template <MaterialGeometryParam PARAM>
__device__ void material_set_float(MaterialContext<MATERIAL_GEOMETRY>& ctx, const float value) {
  static_assert(
    MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) != MATERIAL_PARAM_TYPE_NORM_COLOR
      && MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) != MATERIAL_PARAM_TYPE_COLOR,
    "PARAM must not be color type.");

  constexpr uint32_t SIZE          = MATERIAL_GEOMETRY_PARAM_GET_SIZE(PARAM);
  constexpr MaterialParamType TYPE = MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM);

  float value_remapped = value;

  if constexpr (TYPE == MATERIAL_PARAM_TYPE_NORM_FLOAT) {
    // Nothing
  }
  else if constexpr (TYPE == MATERIAL_PARAM_TYPE_SIGNED_NORM_FLOAT) {
    value_remapped = (value_remapped + 1.0f) * 0.5f;
  }
  else if constexpr (TYPE == MATERIAL_PARAM_TYPE_IOR) {
    value_remapped = (value_remapped - 1.0f) * 0.5f;
  }
  else if constexpr (TYPE == MATERIAL_PARAM_TYPE_ABBE) {
    value_remapped = (value_remapped - 9.0f) * (1.0f / 82.0f);
  }

  value_remapped = __saturatef(value_remapped) * ((1u << SIZE) - 1) + 0.5f;

  const uint32_t data = (uint32_t) value_remapped;

  material_param_set_data<PARAM>(ctx, data);
}

template <MaterialGeometryParam PARAM>
__device__ void material_set_color(MaterialContext<MATERIAL_GEOMETRY>& ctx, const RGBF value) {
  static_assert(
    MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) == MATERIAL_PARAM_TYPE_NORM_COLOR
      || MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) == MATERIAL_PARAM_TYPE_COLOR,
    "PARAM must be color type.");

  uint32_t data;
  if constexpr (MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) == MATERIAL_PARAM_TYPE_NORM_COLOR) {
    const uint32_t data_red   = (uint32_t) (value.r * 0x3FF + 0.5f);
    const uint32_t data_green = (uint32_t) (value.g * 0x3FF + 0.5f);
    const uint32_t data_blue  = (uint32_t) (value.b * 0x3FF + 0.5f);

    data = data_red | (data_green << 10) | (data_blue << 20);
  }
  else {
  }

  material_param_set_data<PARAM>(ctx, data);
}

__device__ MaterialContextGeometry material_get_default_context() {
  MaterialContextGeometry ctx;

  ctx.instance_id = HIT_TYPE_INVALID;
  ctx.tri_id      = 0;
  ctx.albedo      = get_RGBAF(1.0f, 1.0f, 1.0f, 1.0f);
  ctx.emission    = get_color(0.0f, 0.0f, 0.0f);
  ctx.normal      = get_vector(0.0f, 0.0f, 1.0f);
  ctx.position    = get_vector(0.0f, 0.0f, 0.0f);
  ctx.V           = get_vector(0.0f, 0.0f, 1.0f);
  ctx.state       = 0;
  ctx.flags       = 0;
  ctx.ior_in      = 1.0f;
  ctx.ior_out     = 1.0f;

  material_set_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(ctx, 0.5f);

  return ctx;
}

#endif /* CU_LUMINARY_MATERIAL_H */
