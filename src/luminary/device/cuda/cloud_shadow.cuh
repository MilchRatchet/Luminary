#ifndef CLOUD_SHADOW_CUH
#define CLOUD_SHADOW_CUH

#include "cloud_utils.cuh"
#include "math.cuh"
#include "random.cuh"
#include "sky_defines.h"
#include "texture_utils.cuh"
#include "utils.cuh"

#define CLOUD_SHADOW_MAP_HALF_EXTENT 256.0f

LUMINARY_FUNCTION Mat3x3 cloud_shadow_basis(const vec3 sun_dir) {
  return create_basis(sun_dir);
}

LUMINARY_FUNCTION UV cloud_shadow_world_to_uv(const vec3 pos, const Mat3x3& basis) {
  const vec3 center = get_vector(0.0f, SKY_EARTH_RADIUS, 0.0f);
  const vec3 rel    = sub_vector(pos, center);
  const vec3 u1     = get_vector(basis.f11, basis.f21, basis.f31);
  const vec3 u2     = get_vector(basis.f12, basis.f22, basis.f32);

  const float ox  = dot_product(rel, u1);
  const float oy  = dot_product(rel, u2);
  const float inv = 1.0f / (2.0f * CLOUD_SHADOW_MAP_HALF_EXTENT);

  return get_uv(ox * inv + 0.5f, oy * inv + 0.5f);
}

LUMINARY_FUNCTION vec3 cloud_shadow_uv_to_origin(const float u, const float v, const Mat3x3& basis, const vec3 sun_dir) {
  const vec3 center = get_vector(0.0f, SKY_EARTH_RADIUS, 0.0f);
  const vec3 u1     = get_vector(basis.f11, basis.f21, basis.f31);
  const vec3 u2     = get_vector(basis.f12, basis.f22, basis.f32);

  const float ox = (u - 0.5f) * (2.0f * CLOUD_SHADOW_MAP_HALF_EXTENT);
  const float oy = (v - 0.5f) * (2.0f * CLOUD_SHADOW_MAP_HALF_EXTENT);

  vec3 origin = add_vector(center, add_vector(scale_vector(u1, ox), scale_vector(u2, oy)));

  if (get_length(origin) < SKY_EARTH_RADIUS) {
    const float earth_exit = sph_ray_int_p0(sun_dir, origin, SKY_EARTH_RADIUS);
    if (earth_exit != FLT_MAX) {
      origin = add_vector(origin, scale_vector(sun_dir, earth_exit + 0.01f));
    }
  }

  return origin;
}

template <CloudLayerType LAYER_TYPE>
LUMINARY_FUNCTION float cloud_shadow_layer_transmittance(const vec3 origin, const vec3 ray, const int step_count) {
  float2 cloud_layer_intersect;
  float max_dist;

  switch (LAYER_TYPE) {
    case CLOUD_LAYER_LOW: {
      cloud_layer_intersect = cloud_get_lowlayer_intersection(origin, ray, FLT_MAX);
      max_dist              = 8.0f * (device.cloud.low.height_max - device.cloud.low.height_min);
    } break;
    case CLOUD_LAYER_MID: {
      cloud_layer_intersect = cloud_get_midlayer_intersection(origin, ray, FLT_MAX);
      max_dist              = 8.0f * (device.cloud.mid.height_max - device.cloud.mid.height_min);
    } break;
    case CLOUD_LAYER_TOP: {
      cloud_layer_intersect = cloud_get_toplayer_intersection(origin, ray, FLT_MAX);
      max_dist              = 8.0f * (device.cloud.top.height_max - device.cloud.top.height_min);
    } break;
    default:
      LUMINARY_UNREACHABLE;
      return 1.0f;
  }

  const float start = cloud_layer_intersect.x;
  const float dist  = fminf(cloud_layer_intersect.y, max_dist);

  if (start == FLT_MAX || dist <= 0.0f) {
    return 1.0f;
  }

  const float step_size = dist / step_count;
  float reach           = start + 0.5f * step_size;
  float optical_depth   = 0.0f;

  for (int i = 0; i < step_count; i++) {
    const vec3 pos = add_vector(origin, scale_vector(ray, reach));
    reach += step_size;

    const float height = cloud_height<LAYER_TYPE>(pos);

    if (height < 0.0f || height > 1.0f) {
      continue;
    }

    const CloudWeather weather = cloud_weather<LAYER_TYPE>(pos, height);

    if (cloud_significant_point<LAYER_TYPE>(height, weather)) {
      optical_depth -= cloud_density<LAYER_TYPE>(pos, height, weather, 1.0f) * step_size * CLOUD_EXTINCTION_DENSITY;
      if (optical_depth < -64.0f) {
        return 0.0f;
      }
    }
  }

  return expf(optical_depth);
}

#ifndef OPTIX_KERNEL

LUMINARY_KERNEL void cloud_compute_shadow_map(const KernelArgsCloudComputeShadowMap args) {
  const uint32_t amount = args.dim * args.dim;

  const vec3 sun_dir = normalize_vector(device.sky.sun_pos);
  const Mat3x3 basis = cloud_shadow_basis(sun_dir);

  const uint32_t low_steps = max((uint32_t) device.cloud.shadow_steps * 2, 32u);
  const uint32_t mid_steps = max((uint32_t) device.cloud.shadow_steps, 16u);
  const uint32_t top_steps = max((uint32_t) ((device.cloud.shadow_steps + 1) / 2), 8u);

  for (uint32_t id = THREAD_ID; id < amount; id += blockDim.x * gridDim.x) {
    const uint32_t x = id % args.dim;
    const uint32_t y = id / args.dim;

    if (device.cloud.active == false) {
      args.tex[y * args.ld + x] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
      continue;
    }

    const float u = ((float) x + 0.5f) / (float) args.dim;
    const float v = ((float) y + 0.5f) / (float) args.dim;

    const vec3 origin = cloud_shadow_uv_to_origin(u, v, basis, sun_dir);

    const float t_low = (device.cloud.low_active) ? cloud_shadow_layer_transmittance<CLOUD_LAYER_LOW>(origin, sun_dir, low_steps) : 1.0f;
    const float t_mid = (device.cloud.mid_active) ? cloud_shadow_layer_transmittance<CLOUD_LAYER_MID>(origin, sun_dir, mid_steps) : 1.0f;
    const float t_top = (device.cloud.top_active) ? cloud_shadow_layer_transmittance<CLOUD_LAYER_TOP>(origin, sun_dir, top_steps) : 1.0f;

    args.tex[y * args.ld + x] = make_float4(t_low, t_mid, t_top, 1.0f);
  }
}

#endif  // !OPTIX_KERNEL

LUMINARY_FUNCTION float cloud_shadow_fast(const vec3 origin) {
  if (device.cloud.active == false) {
    return 1.0f;
  }

  const float height = sky_height(origin);

  float max_active_height = 0.0f;
  if (device.cloud.low_active) {
    max_active_height = fmaxf(max_active_height, device.cloud.low.height_max);
  }
  if (device.cloud.mid_active) {
    max_active_height = fmaxf(max_active_height, device.cloud.mid.height_max);
  }
  if (device.cloud.top_active) {
    max_active_height = fmaxf(max_active_height, device.cloud.top.height_max);
  }

  if (height >= max_active_height) {
    return 1.0f;
  }

  const vec3 sun_dir = normalize_vector(device.sky.sun_pos);
  const Mat3x3 basis = cloud_shadow_basis(sun_dir);
  UV uv              = cloud_shadow_world_to_uv(origin, basis);

  if (uv.u < 0.0f || uv.u > 1.0f || uv.v < 0.0f || uv.v > 1.0f) {
    return 1.0f;
  }

  const float edge        = fminf(fminf(uv.u, 1.0f - uv.u), fminf(uv.v, 1.0f - uv.v));
  const float border_fade = __saturatef(edge * 16.0f);

  TextureLoadArgs tex_load_args = texture_get_default_args();
  tex_load_args.flip_v          = false;
  tex_load_args.apply_gamma     = false;
  tex_load_args.default_result  = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

  const float4 t = texture_load(device.cloud_shadow_tex, uv, tex_load_args);

  float transmittance = 1.0f;
  if (device.cloud.low_active) {
    const float h = __saturatef((height - device.cloud.low.height_min) / (device.cloud.low.height_max - device.cloud.low.height_min));
    transmittance *= lerp(t.x, 1.0f, h);
  }
  if (device.cloud.mid_active) {
    const float h = __saturatef((height - device.cloud.mid.height_min) / (device.cloud.mid.height_max - device.cloud.mid.height_min));
    transmittance *= lerp(t.y, 1.0f, h);
  }
  if (device.cloud.top_active) {
    const float h = __saturatef((height - device.cloud.top.height_min) / (device.cloud.top.height_max - device.cloud.top.height_min));
    transmittance *= lerp(t.z, 1.0f, h);
  }

  return lerp(1.0f, transmittance, border_fade);
}

#endif /* CLOUD_SHADOW_CUH */
