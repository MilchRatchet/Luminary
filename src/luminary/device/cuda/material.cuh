#ifndef CU_LUMINARY_MATERIAL_H
#define CU_LUMINARY_MATERIAL_H

#include "math.cuh"
#include "utils.cuh"

enum MaterialType { MATERIAL_GEOMETRY, MATERIAL_VOLUME, MATERIAL_PARTICLE } typedef MaterialType;

enum MaterialParamType {
  MATERIAL_PARAM_TYPE_NORM_FLOAT,         // [0,1]
  MATERIAL_PARAM_TYPE_SIGNED_NORM_FLOAT,  // [-1,1]
  MATERIAL_PARAM_TYPE_IOR,                // [0,3]
  MATERIAL_PARAM_TYPE_ABBE,               // [9,91]
  MATERIAL_PARAM_TYPE_NORM_COLOR,         // [0,1]^3
  MATERIAL_PARAM_TYPE_COLOR,              // [0,1023]^3
  MATERIAL_PARAM_TYPE_PACKED_NORMAL       // [-1,1]^3
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
#define MATERIAL_PARAM_TYPE_PACKED_NORMAL_SIZE 32

enum MaterialGeometryParam : uint32_t {
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(FACE_NORMAL, MATERIAL_PARAM_TYPE_PACKED_NORMAL, MATERIAL_PARAM_TYPE_PACKED_NORMAL_SIZE)  //
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(EMISSION, MATERIAL_PARAM_TYPE_COLOR, MATERIAL_PARAM_TYPE_COLOR_SIZE)                     //
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(ALBEDO, MATERIAL_PARAM_TYPE_NORM_COLOR, MATERIAL_PARAM_TYPE_NORM_COLOR_SIZE)             //
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(OPACITY, MATERIAL_PARAM_TYPE_NORM_FLOAT, 8)                                              //
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(ROUGHNESS, MATERIAL_PARAM_TYPE_NORM_FLOAT, 10)                                           //
  MATERIAL_GEOMETRY_PARAM_ALLOCATE(IOR, MATERIAL_PARAM_TYPE_IOR, 8)                                                         //

  MATERIAL_GEOMETRY_PARAM_BITS_COUNT
} typedef MaterialGeometryParam;

#define MATERIAL_GEOM_NUM_UINTS ((MATERIAL_GEOMETRY_PARAM_BITS_COUNT + ((1 << 5) - 1)) >> 5)

template <MaterialType TYPE>
struct MaterialContext {};

template <>
struct MaterialContext<MATERIAL_GEOMETRY> {
  using RANDOM_GI         = RandomSet::BSDF<0>;
  using RANDOM_DL_SUN     = RandomSet::LIGHT_SUN<0>;
  using RANDOM_DL_GEO     = RandomSet::LIGHT_GEO<0>;
  using RANDOM_DL_AMBIENT = RandomSet::BSDF<1>;

  uint32_t instance_id;
  uint32_t tri_id;
  vec3 position;
  vec3 V;
  vec3 normal;
  uint16_t state;
  uint8_t flags;
  VolumeType volume_type;

  uint32_t data[MATERIAL_GEOM_NUM_UINTS];

  __device__ TriangleHandle get_handle() const {
    return triangle_handle_get(instance_id, tri_id);
  }
};

template <>
struct MaterialContext<MATERIAL_VOLUME> {
  using RANDOM_GI         = RandomSet::BSDF<0>;
  using RANDOM_DL_SUN     = RandomSet::LIGHT_SUN<1>;
  using RANDOM_DL_GEO     = RandomSet::LIGHT_GEO<1>;
  using RANDOM_DL_AMBIENT = RandomSet::BSDF<2>;

  VolumeDescriptor descriptor;
  vec3 position;
  vec3 V;
  uint16_t state;
  VolumeType volume_type;
  float max_dist;

  __device__ TriangleHandle get_handle() const {
    return triangle_handle_get(VOLUME_TYPE_TO_HIT(descriptor.type), 0);
  }
};

template <>
struct MaterialContext<MATERIAL_PARTICLE> {
  using RANDOM_GI         = RandomSet::BSDF<0>;
  using RANDOM_DL_SUN     = RandomSet::LIGHT_SUN<0>;
  using RANDOM_DL_GEO     = RandomSet::LIGHT_GEO<0>;
  using RANDOM_DL_AMBIENT = RandomSet::BSDF<1>;

  uint32_t particle_id;
  vec3 position;
  vec3 V;
  vec3 normal;
  uint16_t state;
  uint8_t flags;
  VolumeType volume_type;

  __device__ TriangleHandle get_handle() const {
    return triangle_handle_get(particle_id, 0);
  }
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

  if constexpr (((OFFSET & 0x1Fu) + SIZE) > 32u) {
    constexpr uint32_t second_index = first_index + 1u;
    constexpr uint32_t second_mask  = (1u << (((OFFSET & 0x1Fu) + SIZE) - 32u)) - 1u;
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

  if constexpr (((OFFSET & 0x1Fu) + SIZE) > 32u) {
    constexpr uint32_t second_index = first_index + 1u;
    constexpr uint32_t second_shift = 32u - (OFFSET & 0x1Fu);

    ctx.data[second_index] = (ctx.data[second_index] & ~(first_src_mask >> second_shift)) | ((value & first_src_mask) >> second_shift);
  }
}

template <MaterialGeometryParam PARAM>
__device__ float material_get_float(const MaterialContext<MATERIAL_GEOMETRY>& ctx) {
  static_assert(
    MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) != MATERIAL_PARAM_TYPE_NORM_COLOR
      && MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) != MATERIAL_PARAM_TYPE_COLOR
      && MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) != MATERIAL_PARAM_TYPE_PACKED_NORMAL,
    "PARAM must be float type.");

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
    result = result * 3.0f;
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
    const uint32_t data_max_value       = data & 0x3FFF;
    const uint32_t data_lower_relative  = (data >> 14) & 0xFF;
    const uint32_t data_higher_relative = (data >> 22) & 0xFF;
    const uint32_t max_component        = data >> 30;

    float max_value       = (data_max_value > 0) ? __uint_as_float((data_max_value << 14) | 0x30000000) * (1023.0f / 2.0f) : 0.0f;
    float lower_relative  = data_lower_relative * (1.0f / 0xFF) * max_value;
    float higher_relative = data_higher_relative * (1.0f / 0xFF) * max_value;

    if (max_component == 0) {
      result.r = max_value;
      result.g = lower_relative;
      result.b = higher_relative;
    }
    else if (max_component == 1) {
      result.r = lower_relative;
      result.g = max_value;
      result.b = higher_relative;
    }
    else {
      result.r = lower_relative;
      result.g = higher_relative;
      result.b = max_value;
    }
  }

  return result;
}

template <MaterialGeometryParam PARAM>
__device__ vec3 material_get_normal(const MaterialContext<MATERIAL_GEOMETRY>& ctx) {
  static_assert(MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) == MATERIAL_PARAM_TYPE_PACKED_NORMAL, "PARAM must be packed normal type.");

  const uint32_t data = material_param_get_data<PARAM>(ctx);

  return normal_unpack(data);
}

template <MaterialGeometryParam PARAM>
__device__ void material_set_float(MaterialContext<MATERIAL_GEOMETRY>& ctx, const float value) {
  static_assert(
    MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) != MATERIAL_PARAM_TYPE_NORM_COLOR
      && MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) != MATERIAL_PARAM_TYPE_COLOR
      && MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) != MATERIAL_PARAM_TYPE_PACKED_NORMAL,
    "PARAM must be float type.");

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
    value_remapped = value_remapped * (1.0f / 3.0f);
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
    const uint32_t data_red   = (uint32_t) (__saturatef(value.r) * 0x3FF + 0.5f);
    const uint32_t data_green = (uint32_t) (__saturatef(value.g) * 0x3FF + 0.5f);
    const uint32_t data_blue  = (uint32_t) (__saturatef(value.b) * 0x3FF + 0.5f);

    data = data_red | (data_green << 10) | (data_blue << 20);
  }
  else {
    uint32_t max_component;
    float max_value;
    float lower_relative;
    float higher_relative;

    if (value.r > value.g && value.r > value.b) {
      max_component   = 0;
      max_value       = value.r;
      lower_relative  = value.g;
      higher_relative = value.b;
    }
    else if (value.g > value.b) {
      max_component   = 1;
      max_value       = value.g;
      lower_relative  = value.r;
      higher_relative = value.b;
    }
    else {
      max_component   = 2;
      max_value       = value.b;
      lower_relative  = value.r;
      higher_relative = value.g;
    }

    max_value       = __saturatef(max_value * (1.0f / 1023.0f)) * 2.0f;  // Map to [0,2]
    lower_relative  = __saturatef(lower_relative * (2.0f / 1023.0f) * (1.0f / max_value));
    higher_relative = __saturatef(higher_relative * (2.0f / 1023.0f) * (1.0f / max_value));

    uint32_t data_max_value       = (__float_as_uint(max_value) >= 0x30000000) ? (__float_as_uint(max_value) >> 14) & 0x3FFF : 0;
    uint32_t data_lower_relative  = (uint32_t) (lower_relative * 0xFF + 0.5f);
    uint32_t data_higher_relative = (uint32_t) (higher_relative * 0xFF + 0.5f);

    data = data_max_value | (data_lower_relative << 14) | (data_higher_relative << 22) | (max_component << 30);
  }

  material_param_set_data<PARAM>(ctx, data);
}

template <MaterialGeometryParam PARAM>
__device__ void material_set_normal(MaterialContext<MATERIAL_GEOMETRY>& ctx, const vec3 value) {
  static_assert(MATERIAL_GEOMETRY_PARAM_GET_TYPE(PARAM) == MATERIAL_PARAM_TYPE_PACKED_NORMAL, "PARAM must be packed normal type.");

  const uint32_t data = normal_pack(value);

  material_param_set_data<PARAM>(ctx, data);
}

__device__ MaterialContextGeometry material_get_default_context() {
  MaterialContextGeometry ctx;

  ctx.instance_id = HIT_TYPE_INVALID;
  ctx.tri_id      = 0;
  ctx.normal      = get_vector(0.0f, 0.0f, 1.0f);
  ctx.position    = get_vector(0.0f, 0.0f, 0.0f);
  ctx.V           = get_vector(0.0f, 0.0f, 1.0f);
  ctx.state       = 0;
  ctx.flags       = 0;

  material_set_normal<MATERIAL_GEOMETRY_PARAM_FACE_NORMAL>(ctx, get_vector(0.0f, 0.0f, 1.0f));
  material_set_color<MATERIAL_GEOMETRY_PARAM_ALBEDO>(ctx, splat_color(1.0f));
  material_set_float<MATERIAL_GEOMETRY_PARAM_OPACITY>(ctx, 1.0f);
  material_set_float<MATERIAL_GEOMETRY_PARAM_ROUGHNESS>(ctx, 0.5f);
  material_set_color<MATERIAL_GEOMETRY_PARAM_EMISSION>(ctx, splat_color(0.0f));
  material_set_float<MATERIAL_GEOMETRY_PARAM_IOR>(ctx, 1.0f);

  return ctx;
}

#endif /* CU_LUMINARY_MATERIAL_H */
