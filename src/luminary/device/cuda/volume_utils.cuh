#ifndef CU_VOLUME_UTILS_H
#define CU_VOLUME_UTILS_H

#include "material.cuh"
#include "ocean_utils.cuh"
#include "utils.cuh"

#define FOG_DENSITY (0.001f * device.fog.density)

__device__ RGBF volume_get_transmittance(const VolumeDescriptor volume) {
  return add_color(volume.absorption, volume.scattering);
}

__device__ VolumeDescriptor volume_get_descriptor_preset_fog() {
  VolumeDescriptor volume;

  volume.type           = VOLUME_TYPE_FOG;
  volume.absorption     = get_color(0.0f, 0.0f, 0.0f);
  volume.scattering     = get_color(FOG_DENSITY, FOG_DENSITY, FOG_DENSITY);
  volume.max_scattering = FOG_DENSITY;
  volume.dist           = device.fog.dist;
  volume.max_height     = device.fog.height;
  volume.min_height     = (device.ocean.active) ? OCEAN_MAX_HEIGHT : -65535.0f;

  return volume;
}

__device__ VolumeDescriptor volume_get_descriptor_preset_ocean() {
  VolumeDescriptor volume;

  volume.type       = VOLUME_TYPE_OCEAN;
  volume.absorption = ocean_jerlov_absorption_coefficient((JerlovWaterType) device.ocean.water_type);
  volume.scattering = ocean_jerlov_scattering_coefficient((JerlovWaterType) device.ocean.water_type);
  volume.dist       = 10000.0f;
  volume.max_height = 65535.0f;
  volume.min_height = -65535.0f;

  volume.max_scattering = color_importance(volume.scattering);

  return volume;
}

__device__ VolumeDescriptor volume_get_descriptor_preset(const VolumeType type) {
  switch (type) {
    case VOLUME_TYPE_FOG:
      return volume_get_descriptor_preset_fog();
    case VOLUME_TYPE_OCEAN:
      return volume_get_descriptor_preset_ocean();
    default:
      return {};
  }
}

struct VolumePath {
  union {
    float2 data;
    struct {
      float start;
      float length;
    };
  };
} typedef VolumePath;

__device__ VolumePath make_volume_path(const float start, const float length) {
  VolumePath path;

  path.start  = start;
  path.length = length;

  return path;
}

/*
 * Computes the start and length of a ray path through a volume.
 * The output path is only valid if the start is non-negative.
 *
 * @param origin Start point of ray in world space.
 * @param ray Direction of ray.
 * @param limit Maximum distance a ray may travel in world space.
 * @result Two floats:
 *                  - [x] = Start in world space.
 *                  - [y] = Distance through fog in world space.
 */
__device__ VolumePath volume_compute_path(
  const VolumeDescriptor volume, const vec3 origin, const vec3 ray, const float limit, const bool ocean_fast_path = false) {
  if (limit <= 0.0f)
    return make_volume_path(-FLT_MAX, 0.0f);

  if (volume.max_height <= volume.min_height)
    return make_volume_path(-FLT_MAX, 0.0f);

  if (volume.type == VOLUME_TYPE_NONE)
    return make_volume_path(-FLT_MAX, 0.0f);

  // Vertical intersection
  float start_y;
  float end_y;
  if (volume.type == VOLUME_TYPE_OCEAN) {
    start_y = 0.0f;
    end_y   = (ocean_fast_path == false) ? ocean_intersection_distance(origin, ray, limit) : limit;
  }
  else {
    if (fabsf(ray.y) < 0.005f) {
      if (origin.y >= volume.min_height && origin.y <= volume.max_height) {
        start_y = 0.0f;
        end_y   = volume.dist;
      }
      else {
        return make_volume_path(-FLT_MAX, 0.0f);
      }
    }
    else {
      const float dy1 = volume.min_height - origin.y;
      const float dy2 = volume.max_height - origin.y;

      const float sy1 = dy1 / ray.y;
      const float sy2 = dy2 / ray.y;

      start_y = fmaxf(fminf(sy1, sy2), 0.0f);
      end_y   = fmaxf(sy1, sy2);
    }
  }

  // Horizontal intersection
  const float rn = 1.0f / sqrtf(ray.x * ray.x + ray.z * ray.z);
  const float rx = ray.x * rn;
  const float rz = ray.z * rn;

  const float dx = origin.x - device.camera.pos.x;
  const float dz = origin.z - device.camera.pos.z;

  const float dot = dx * rx + dz * rz;
  const float r2  = volume.dist * volume.dist;
  const float c   = (dx * dx + dz * dz) - r2;

  const float kx = dx - rx * dot;
  const float kz = dz - rz * dot;

  const float d = r2 - (kx * kx + kz * kz);

  if (d < 0.0f)
    return make_volume_path(-FLT_MAX, 0.0f);

  const float sd = sqrtf(d);
  const float q  = -dot - copysignf(sd, dot);

  const float t0 = fmaxf(0.0f, c / q);
  const float t1 = fmaxf(0.0f, q);

  const float start_xz = fminf(t0, t1);
  const float end_xz   = fmaxf(t0, t1);

  if (end_xz < start_xz || limit < start_xz)
    return make_volume_path(-FLT_MAX, 0.0f);

  const float start = fmaxf(start_xz, start_y);
  const float dist  = fminf(fminf(end_xz, end_y) - start, limit - start);

  if (dist < 0.0f)
    return make_volume_path(-FLT_MAX, 0.0f);

  return make_volume_path(start, dist);
}

/*
 * Computes a random volume intersection point by perfectly importance sampling the transmittance
 * based pdf.
 *
 * @param volume VolumeDescriptor of the corresponding volume.
 * @param start Start offset of ray.
 * @param max_length Maximum distance ray may travel after start.
 * @result Distance of intersection point and origin in world space.
 */
__device__ float volume_sample_intersection(const VolumeDescriptor volume, const float start, const float max_length, const float random) {
  // [FonWKH17] Equation 15
  const float t = (-logf(random)) / volume.max_scattering;

  if (t > max_length)
    return FLT_MAX;

  return start + t;
}

__device__ float volume_sample_intersection_pdf(const VolumeDescriptor volume, const float start, const float t) {
  return volume.max_scattering * expf(-volume.max_scattering * (t - start));
}

__device__ float volume_sample_intersection_miss_probability(const VolumeDescriptor volume, const float depth) {
  return expf(-volume.max_scattering * depth);
}

__device__ float volume_sample_intersection_bounded(const VolumeDescriptor volume, const float max_length, const float random) {
  const float prob_hit_at_max = 1.0f - expf(-volume.max_scattering * max_length);

  // [FonWKH17] Equation 15
  const float t = -logf(1.0f - random * prob_hit_at_max) / volume.max_scattering;

  return t;
}

__device__ float volume_sample_intersection_bounded_pdf(const VolumeDescriptor volume, const float max_length, const float t) {
  const float prob_hit_at_max = 1.0f - expf(-volume.max_scattering * max_length);

  return volume.max_scattering * expf(-volume.max_scattering * t) / prob_hit_at_max;
}

template <MaterialType TYPE>
__device__ float volume_phase_evaluate(const MaterialContext<TYPE> ctx, const vec3 ray);

template <>
__device__ float volume_phase_evaluate<MATERIAL_VOLUME>(const MaterialContextVolume ctx, const vec3 ray) {
  const float cos_angle = -dot_product(ctx.V, ray);

  float phase;
  if (ctx.descriptor.type == VOLUME_TYPE_OCEAN) {
    phase = ocean_phase(cos_angle);
  }
  else {
    const float diameter            = device.fog.droplet_diameter;
    const JendersieEonParams params = jendersie_eon_phase_parameters(diameter);
    phase                           = jendersie_eon_phase_function(cos_angle, params);
  }

  return phase;
}

template <>
__device__ float volume_phase_evaluate<MATERIAL_PARTICLE>(const MaterialContextParticle ctx, const vec3 ray) {
  const float cos_angle = -dot_product(ctx.V, ray);

  const float diameter            = device.particles.phase_diameter;
  const JendersieEonParams params = jendersie_eon_phase_parameters(diameter);
  return jendersie_eon_phase_function(cos_angle, params);
}

__device__ RGBF volume_integrate_transmittance_precomputed(const VolumeDescriptor volume, const float length) {
  const RGBF volume_transmittance = volume_get_transmittance(volume);

  RGBF result;
  result.r = expf(-length * volume_transmittance.r);
  result.g = expf(-length * volume_transmittance.g);
  result.b = expf(-length * volume_transmittance.b);

  return result;
}

__device__ RGBF volume_integrate_transmittance_ocean(const vec3 origin, const vec3 ray, const float depth, const bool force_path = false) {
  RGBF ocean_transmittance = get_color(1.0f, 1.0f, 1.0f);

  if (device.ocean.active) {
    const VolumeDescriptor volume = volume_get_descriptor_preset_ocean();
    const VolumePath path         = (force_path) ? make_volume_path(0.0f, depth) : volume_compute_path(volume, origin, ray, depth);

    if (path.start >= 0.0f) {
      RGBF volume_transmittance = volume_get_transmittance(volume);

      ocean_transmittance.r = expf(-path.length * volume_transmittance.r);
      ocean_transmittance.g = expf(-path.length * volume_transmittance.g);
      ocean_transmittance.b = expf(-path.length * volume_transmittance.b);
    }
  }

  return ocean_transmittance;
}

__device__ float volume_integrate_transmittance_fog(const vec3 origin, const vec3 ray, const float depth) {
  float fog_transmittance = 1.0f;

  if (device.fog.active) {
    const VolumeDescriptor volume = volume_get_descriptor_preset_fog();
    const VolumePath path         = volume_compute_path(volume, origin, ray, depth);

    if (path.start >= 0.0f) {
      fog_transmittance = expf(-path.length * volume.max_scattering);
    }
  }

  return fog_transmittance;
}

__device__ RGBF volume_integrate_transmittance(const VolumeType volume_type, const vec3 origin, const vec3 ray, const float depth) {
  const VolumeDescriptor volume = volume_get_descriptor_preset(volume_type);

  const VolumePath path = volume_compute_path(volume, origin, ray, depth);

  RGBF transmittance = splat_color(1.0f);

  if (path.start >= 0.0f) {
    RGBF volume_transmittance = volume_get_transmittance(volume);

    transmittance.r = expf(-path.length * volume_transmittance.r);
    transmittance.g = expf(-path.length * volume_transmittance.g);
    transmittance.b = expf(-path.length * volume_transmittance.b);
  }

  return transmittance;
}

__device__ MaterialContextVolume volume_get_context(const DeviceTask task, const VolumeDescriptor volume, const float max_dist) {
  MaterialContextVolume ctx;
  ctx.descriptor  = volume;
  ctx.position    = task.origin;
  ctx.V           = scale_vector(task.ray, -1.0f);
  ctx.state       = task.state;
  ctx.volume_type = volume.type;
  ctx.max_dist    = max_dist;

  return ctx;
}

__device__ float volume_sample_sky_dl_initial_vertex_dist(MaterialContextVolume& ctx, const ushort2 pixel) {
  const float random = random_1D(RANDOM_TARGET_LIGHT_SUN_INITIAL_VERTEX, pixel);

  const float dist = volume_sample_intersection_bounded(ctx.descriptor, ctx.max_dist, random);

  ctx.position = add_vector(ctx.position, scale_vector(ctx.V, -dist));

  return dist;
}

__device__ RGBF volume_sample_sky_dl_initial_vertex_weight(MaterialContextVolume& ctx, const float dist) {
  const RGBF volume_transmittance = volume_get_transmittance(ctx.descriptor);

  RGBF weight;
  weight.r = expf(-dist * volume_transmittance.r) * ctx.descriptor.scattering.r;
  weight.g = expf(-dist * volume_transmittance.g) * ctx.descriptor.scattering.g;
  weight.b = expf(-dist * volume_transmittance.b) * ctx.descriptor.scattering.b;

  weight = scale_color(weight, 1.0f / volume_sample_intersection_bounded_pdf(ctx.descriptor, ctx.max_dist, dist));

  return weight;
}

__device__ void volume_sample_sky_dl_initial_vertex(MaterialContextVolume& ctx, const ushort2 pixel, RGBF& weight) {
  const float dist = volume_sample_sky_dl_initial_vertex_dist(ctx, pixel);
  weight           = volume_sample_sky_dl_initial_vertex_weight(ctx, dist);
}

#endif /* CU_VOLUME_UTILS_H */
