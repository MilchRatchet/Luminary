#ifndef CU_VOLUME_UTILS_H
#define CU_VOLUME_UTILS_H

#include "ocean_utils.cuh"
#include "utils.cuh"

#define FOG_DENSITY (0.001f * device.fog.density)

struct VolumeDescriptor {
  VolumeType type;
  RGBF absorption;
  RGBF scattering;
  float max_scattering;
  float dist;
  float max_height;
  float min_height;
} typedef VolumeDescriptor;

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
  volume.max_height = OCEAN_MAX_HEIGHT;
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

__device__ VolumeType volume_get_type_at_position(const vec3 pos) {
  VolumeType type = VOLUME_TYPE_NONE;

  if (device.fog.active) {
    const VolumeDescriptor volume_fog = volume_get_descriptor_preset_fog();

    if (pos.y <= volume_fog.max_height && pos.y >= volume_fog.min_height) {
      type = VOLUME_TYPE_FOG;
    }
  }

  if (device.ocean.active) {
    const VolumeDescriptor volume_ocean = volume_get_descriptor_preset_ocean();

    if (pos.y <= volume_ocean.max_height && pos.y >= volume_ocean.min_height) {
      type = VOLUME_TYPE_OCEAN;
    }
  }

  return type;
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
__device__ float2 volume_compute_path(const VolumeDescriptor volume, const vec3 origin, const vec3 ray, const float limit) {
  if (limit <= 0.0f)
    return make_float2(-FLT_MAX, 0.0f);

  if (volume.max_height <= volume.min_height)
    return make_float2(-FLT_MAX, 0.0f);

  // Vertical intersection
  float start_y;
  float end_y;
  if (volume.type == VOLUME_TYPE_OCEAN) {
    const bool above_surface = ocean_get_relative_height(origin, OCEAN_ITERATIONS_INTERSECTION) > 0.0f;

    // Without loss of generality, we can simply assume that the end of the volume is at our closest intersection
    // as long as we are below the surface.
#ifdef SHADING_KERNEL
    const float surface_intersect = ocean_intersection_distance(origin, ray, limit);
#else
    const float surface_intersect = (above_surface) ? ocean_intersection_distance(origin, ray, limit) : limit;
#endif

    if (above_surface) {
      start_y = surface_intersect;
      end_y   = FLT_MAX;
    }
    else {
      start_y = 0.0f;
      end_y   = surface_intersect;
    }
  }
  else {
    if (fabsf(ray.y) < 0.005f) {
      if (origin.y >= volume.min_height && origin.y <= volume.max_height) {
        start_y = 0.0f;
        end_y   = volume.dist;
      }
      else {
        return make_float2(-FLT_MAX, 0.0f);
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
    return make_float2(-FLT_MAX, 0.0f);

  const float sd = sqrtf(d);
  const float q  = -dot - copysignf(sd, dot);

  const float t0 = fmaxf(0.0f, c / q);
  const float t1 = fmaxf(0.0f, q);

  const float start_xz = fminf(t0, t1);
  const float end_xz   = fmaxf(t0, t1);

  if (end_xz < start_xz || limit < start_xz)
    return make_float2(-FLT_MAX, 0.0f);

  const float start = fmaxf(start_xz, start_y);
  const float dist  = fminf(fminf(end_xz, end_y) - start, limit - start);

  if (dist < 0.0f)
    return make_float2(-FLT_MAX, 0.0f);

  return make_float2(start, dist);
}

/*
 * Computes a random volume intersection point by perfectly importance sampling the transmittance
 * based pdf.
 *
 * @param volume VolumeDescriptor of the corresponding volume.
 * @param start Start offset of ray.
 * @param max_dist Maximum dist ray may travel after start.
 * @result Distance of intersection point and origin in world space.
 */
__device__ float volume_sample_intersection(const VolumeDescriptor volume, const float start, const float max_dist, const float random) {
  // [FonWKH17] Equation 15
  const float t = (-logf(random)) / volume.max_scattering;

  if (t > max_dist)
    return FLT_MAX;

  return start + t;
}

__device__ float volume_sample_intersection_pdf(const VolumeDescriptor volume, const float start, const float t) {
  return volume.max_scattering * expf(-volume.max_scattering * (t - start));
}

__device__ RGBF volume_phase_evaluate(const GBufferData data, const VolumeType volume_hit_type, const vec3 ray) {
  const float cos_angle = -dot_product(data.V, ray);

  float phase;
  if (volume_hit_type == VOLUME_TYPE_OCEAN) {
    phase = ocean_phase(cos_angle);
  }
  else {
    const float diameter            = (volume_hit_type == VOLUME_TYPE_FOG) ? device.fog.droplet_diameter : device.particles.phase_diameter;
    const JendersieEonParams params = jendersie_eon_phase_parameters(diameter);
    phase                           = jendersie_eon_phase_function(cos_angle, params);
  }

  return scale_color(opaque_color(data.albedo), phase);
}

__device__ RGBF volume_integrate_transmittance_ocean(const vec3 origin, const vec3 ray, const float depth, const bool force_path = false) {
  RGBF ocean_transmittance = get_color(1.0f, 1.0f, 1.0f);

  if (device.ocean.active) {
    const VolumeDescriptor volume = volume_get_descriptor_preset_ocean();
    const float2 path             = (force_path) ? make_float2(0.0f, depth) : volume_compute_path(volume, origin, ray, depth);

    if (path.x >= 0.0f) {
      RGBF volume_transmittance = volume_get_transmittance(volume);

      ocean_transmittance.r = expf(-path.y * volume_transmittance.r);
      ocean_transmittance.g = expf(-path.y * volume_transmittance.g);
      ocean_transmittance.b = expf(-path.y * volume_transmittance.b);
    }
  }

  return ocean_transmittance;
}

__device__ float volume_integrate_transmittance_fog(const vec3 origin, const vec3 ray, const float depth) {
  float fog_transmittance = 1.0f;

  if (device.fog.active) {
    const VolumeDescriptor volume = volume_get_descriptor_preset_fog();
    const float2 path             = volume_compute_path(volume, origin, ray, depth);

    if (path.x >= 0.0f) {
      fog_transmittance = expf(-path.y * volume.max_scattering);
    }
  }

  return fog_transmittance;
}

__device__ RGBF volume_integrate_transmittance(const vec3 origin, const vec3 ray, const float depth) {
  const float fog_transmittance  = volume_integrate_transmittance_fog(origin, ray, depth);
  const RGBF ocean_transmittance = volume_integrate_transmittance_ocean(origin, ray, depth);

  return scale_color(ocean_transmittance, fog_transmittance);
}

__device__ GBufferData
  volume_generate_g_buffer(const DeviceTask task, const uint32_t instance_id, const int pixel, const VolumeDescriptor volume) {
  const float scattering_normalization = 1.0f / fmaxf(0.0001f, volume.max_scattering);

  const float ray_ior = ior_stack_interact(1.0f, pixel, IOR_STACK_METHOD_PEEK_CURRENT);

  GBufferData data;
  data.instance_id = instance_id;
  data.tri_id      = 0;
  data.albedo      = RGBAF_set(
    volume.scattering.r * scattering_normalization, volume.scattering.g * scattering_normalization,
    volume.scattering.b * scattering_normalization, 0.0f);
  data.emission  = get_color(0.0f, 0.0f, 0.0f);
  data.normal    = get_vector(0.0f, 0.0f, 0.0f);
  data.position  = task.origin;
  data.V         = scale_vector(task.ray, -1.0f);
  data.roughness = device.fog.droplet_diameter;
  data.state     = task.state;
  data.flags     = 0;
  data.ior_in    = ray_ior;
  data.ior_out   = ray_ior;

  return data;
}

#endif /* CU_VOLUME_UTILS_H */
