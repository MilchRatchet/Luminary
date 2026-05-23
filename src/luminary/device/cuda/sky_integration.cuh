#ifndef CU_LUMINARY_SKY_INTEGRATION_H
#define CU_LUMINARY_SKY_INTEGRATION_H

#include "cloud_shadow.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "ocean_utils.cuh"
#include "sky_utils.cuh"
#include "utils.cuh"

////////////////////////////////////////////////////////////////////
// Sky Utils
////////////////////////////////////////////////////////////////////

// [Wil21]
LUMINARY_FUNCTION float sky_mie_density(const float height) {
  // INSO (insoluble = dust-like particles)
  const float INSO = expf(-height * (1.0f / device.sky.mie_falloff));

  // WASO (water soluble = biogenic particles, organic carbon)
  float WASO = 0.0f;
  if (height < 2.0f) {
    WASO = 1.0f + 0.125f * (2.0f - height);
  }
  else if (height < 3.0f) {
    WASO = 3.0f - height;
  }
  WASO *= 60.0f / device.sky.ground_visibility;

  return device.sky.base_density * (INSO + WASO);
}

LUMINARY_FUNCTION float sky_ozone_density(const float height) {
  if (!device.sky.ozone_absorption)
    return 0.0f;

  const float min_val = (height > 25.0f) ? 0.0f : 0.1f;
  return device.sky.base_density * fmaxf(min_val, 1.0f - fabsf(height - 25.0f) / device.sky.ozone_layer_thickness);
}

/*
 * Computes the start and length of a ray path through atmosphere.
 * @param origin Start point of ray in sky space.
 * @param ray Direction of ray
 * @result 2 floats, first value is the start, second value is the length of the path.
 */
LUMINARY_FUNCTION float2 sky_compute_path(const vec3 origin, const vec3 ray, const float min_height, const float max_height) {
  const float height = get_length(origin);

  if (height <= min_height)
    return make_float2(0.0f, -FLT_MAX);

  const float earth_dist = sph_ray_int_p0(ray, origin, min_height);
  const float atmo_dist  = sph_ray_int_p0(ray, origin, max_height);

  float distance;
  float start;
  if (height > max_height) {
    const float atmo_dist2 = sph_ray_int_back_p0(ray, origin, max_height);

    distance = fminf(earth_dist - atmo_dist, atmo_dist2 - atmo_dist);
    start    = atmo_dist;
  }
  else {
    distance = fminf(earth_dist, atmo_dist);
    start    = 0.0f;
  }

  return make_float2(start, distance);
}

////////////////////////////////////////////////////////////////////
// Atmosphere Integration
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION Spectrum sky_compute_atmosphere(
  Spectrum& transmittance_out, const vec3 origin, const vec3 ray, const float limit, const bool celestials, const bool cloud_shadows,
  const int steps, const PathID& path_id) {
  Spectrum result = spectrum_set1(0.0f);

  const float2 path = sky_compute_path(origin, ray, SKY_EARTH_RADIUS, SKY_ATMO_RADIUS);

  const float start    = path.x;
  const float distance = fminf(path.y, limit - start);

  Spectrum transmittance = spectrum_get_ident();

  if (distance > 0.0f) {
    float reach = start;
    float step_size;

    const float light_angle   = sample_sphere_solid_angle(device.sky.sun_pos, SKY_SUN_RADIUS, origin);
    const float random_offset = random_1D(RANDOM_TARGET_SKY_STEP_OFFSET, path_id);

    const JendersieEonParams mie_params = jendersie_eon_phase_parameters(device.sky.mie_diameter);

    for (int i = 0; i < steps; i++) {
      const float new_reach = start + distance * (i + random_offset) / steps;
      step_size             = new_reach - reach;
      reach                 = new_reach;

      const vec3 pos     = add_vector(origin, scale_vector(ray, reach));
      const float height = sky_height(pos);

      const vec3 ray_scatter       = normalize_vector(sub_vector(device.sky.sun_pos, pos));
      const float cos_angle        = dot_product(ray, ray_scatter);
      const float zenith_cos_angle = dot_product(normalize_vector(pos), ray_scatter);
      const float phase_rayleigh   = sky_rayleigh_phase(cos_angle);
      const float phase_mie        = sky_mie_phase(cos_angle, mie_params);

      float shadow;
      if (cloud_shadows) {
        shadow = sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS) ? 0.0f : cloud_shadow(pos, ray_scatter);
      }
      else {
        shadow = sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS) ? 0.0f : 1.0f;
      }

      TextureLoadArgs tex_load_args = texture_get_default_args();
      tex_load_args.flip_v          = false;
      tex_load_args.apply_gamma     = false;

      const UV transmittance_uv       = sky_transmittance_lut_uv(height, zenith_cos_angle);
      const float4 transmittance_low  = texture_load(device.sky_lut_transmission_low_tex, transmittance_uv, tex_load_args);
      const float4 transmittance_high = texture_load(device.sky_lut_transmission_high_tex, transmittance_uv, tex_load_args);
      const Spectrum extinction_sun   = spectrum_merge(transmittance_low, transmittance_high);

      const float density_rayleigh = sky_rayleigh_density(height) * device.sky.rayleigh_density;
      const float density_mie      = sky_mie_density(height) * device.sky.mie_density;
      const float density_ozone    = sky_ozone_density(height) * device.sky.ozone_density;

      const Spectrum scattering_rayleigh = spectrum_scale(SKY_RAYLEIGH_SCATTERING, density_rayleigh);
      const float scattering_mie         = SKY_MIE_SCATTERING * density_mie;

      const Spectrum extinction_rayleigh = spectrum_scale(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
      const float extinction_mie         = SKY_MIE_EXTINCTION * density_mie;
      const Spectrum extinction_ozone    = spectrum_scale(SKY_OZONE_EXTINCTION, density_ozone);

      const Spectrum scattering = spectrum_add(scattering_rayleigh, spectrum_set1(scattering_mie));
      const Spectrum extinction = spectrum_add(spectrum_add(extinction_rayleigh, spectrum_set1(extinction_mie)), extinction_ozone);
      const Spectrum phase_times_scattering =
        spectrum_add(spectrum_scale(scattering_rayleigh, phase_rayleigh), spectrum_set1(scattering_mie * phase_mie));

      const Spectrum ss_radiance = spectrum_scale(spectrum_mul(extinction_sun, phase_times_scattering), shadow * light_angle);

      const UV multiscattering_uv        = get_uv(zenith_cos_angle * 0.5f + 0.5f, height / SKY_ATMO_HEIGHT);
      const float4 multiscattering_low   = texture_load(device.sky_lut_multiscattering_low_tex, multiscattering_uv, tex_load_args);
      const float4 multiscattering_high  = texture_load(device.sky_lut_multiscattering_high_tex, multiscattering_uv, tex_load_args);
      const Spectrum multiscattering_tex = spectrum_merge(multiscattering_low, multiscattering_high);
      const Spectrum ms_radiance         = spectrum_mul(multiscattering_tex, scattering);

      const Spectrum S = spectrum_add(ss_radiance, ms_radiance);

      Spectrum step_transmittance = extinction;
      step_transmittance          = spectrum_scale(step_transmittance, -step_size);
      step_transmittance          = spectrum_exp(step_transmittance);

      const Spectrum Sint = spectrum_mul(spectrum_sub(S, spectrum_mul(S, step_transmittance)), spectrum_inv(extinction));

      result        = spectrum_add(result, spectrum_mul(Sint, transmittance));
      transmittance = spectrum_mul(transmittance, step_transmittance);
    }

    const Spectrum sun_radiance = spectrum_scale(SKY_SUN_RADIANCE, device.sky.sun_strength);
    result                      = spectrum_mul(result, sun_radiance);
  }

  if (celestials) {
    const float sun_hit   = sphere_ray_intersection(ray, origin, device.sky.sun_pos, SKY_SUN_RADIUS);
    const float earth_hit = sph_ray_int_p0(ray, origin, SKY_EARTH_RADIUS);
    const float moon_hit  = sphere_ray_intersection(ray, origin, device.sky.moon_pos, SKY_MOON_RADIUS);

    if (earth_hit > sun_hit && moon_hit > sun_hit) {
      const Spectrum S = spectrum_mul(transmittance, spectrum_scale(SKY_SUN_RADIANCE, device.sky.sun_strength));

      result = spectrum_add(result, S);
    }
    else if (earth_hit > moon_hit) {
      const vec3 moon_pos   = add_vector(origin, scale_vector(ray, moon_hit));
      const vec3 bounce_ray = normalize_vector(sub_vector(device.sky.sun_pos, moon_pos));

      if (sphere_ray_hit_outside(bounce_ray, moon_pos, get_vector(0.0f, 0.0f, 0.0f), SKY_EARTH_RADIUS) == false) {
        vec3 normal = normalize_vector(sub_vector(moon_pos, device.sky.moon_pos));

        const float tex_u = 0.5f + device.sky.moon_tex_offset + atan2f(normal.z, normal.x) * (1.0f / (2.0f * PI));
        const float tex_v = 0.5f + asinf(normal.y) * (1.0f / PI);

        const UV uv = get_uv(tex_u, tex_v);

        const Mat3x3 tangent_space = create_basis(normal);

        const float4 normal_vals = texture_load(device.moon_normal_tex, uv);

        vec3 map_normal = get_vector(normal_vals.x, normal_vals.y, normal_vals.z);
        map_normal      = scale_vector(map_normal, 2.0f);
        map_normal      = sub_vector(map_normal, get_vector(1.0f, 1.0f, 1.0f));

        normal = normalize_vector(transform_vec3(tangent_space, map_normal));

        const float NdotL = dot_product(normal, bounce_ray);

        if (NdotL > 0.0f) {
          const float albedo = texture_load(device.moon_albedo_tex, uv).x;

          const float light_angle = sample_sphere_solid_angle(device.sky.sun_pos, SKY_SUN_RADIUS, moon_pos);
          const float weight      = albedo * device.sky.sun_strength * NdotL * light_angle / (2.0f * PI);

          result =
            spectrum_add(result, spectrum_mul(transmittance, spectrum_mul(SKY_MOON_SOLAR_FLUX, spectrum_scale(SKY_SUN_RADIANCE, weight))));
        }
      }
    }

    if ((device.ptrs.stars != (Star*) 0) && sun_hit == FLT_MAX && earth_hit == FLT_MAX && moon_hit == FLT_MAX) {
      const float ray_altitude = asinf(ray.y);
      const float ray_azimuth  = atan2f(-ray.z, -ray.x) + PI;

      const uint32_t x = (uint32_t) (ray_azimuth * 10.0f);
      const uint32_t y = (uint32_t) ((ray_altitude + PI * 0.5f) * 10.0f);

      const uint32_t grid = x + y * STARS_GRID_LD;

      const uint32_t a = __ldg(device.ptrs.stars_offsets + grid);
      const uint32_t b = __ldg(device.ptrs.stars_offsets + grid + 1);

      for (uint32_t i = a; i < b; i++) {
        const Star star     = star_load(i);
        const vec3 star_pos = angles_to_direction(star.altitude, star.azimuth);

        if (sphere_ray_hit_outside(ray, get_vector(0.0f, 0.0f, 0.0f), star_pos, star.radius)) {
          result = spectrum_add(result, spectrum_scale(transmittance, star.intensity * device.sky.stars_intensity));
        }
      }
    }
  }

  transmittance_out = spectrum_mul(transmittance_out, transmittance);

  return result;
}

////////////////////////////////////////////////////////////////////
// Wrapper
////////////////////////////////////////////////////////////////////

LUMINARY_FUNCTION RGBF
  sky_get_color(const vec3 origin, const vec3 ray, const float limit, const bool celestials, const int steps, const PathID& path_id) {
  Spectrum unused = spectrum_set1(0.0f);

  const Spectrum radiance = sky_compute_atmosphere(unused, origin, ray, limit, celestials, false, steps, path_id);

  return sky_evaluate_radiance_from_spectrum(radiance);
}

LUMINARY_FUNCTION RGBF sky_trace_inscattering(const vec3 origin, const vec3 ray, const float limit, RGBF& record, const PathID& path_id) {
  Spectrum transmittance = spectrum_set1(1.0f);

  const float base_range = (IS_PRIMARY_RAY) ? 40.0f : 80.0f;

  const int steps =
    fminf(fmaxf(0.5f, limit / base_range), 2.0f) * (device.sky.steps / 6) + random_1D(RANDOM_TARGET_SKY_INSCATTERING_STEP, path_id) - 0.5f;

  const Spectrum radiance = sky_compute_atmosphere(transmittance, origin, ray, limit, false, true, steps, path_id);

  const RGBF inscattering = mul_color(sky_evaluate_radiance_from_spectrum(radiance), record);

  record = mul_color(record, sky_evaluate_transmittance_from_spectrum(transmittance));

  return inscattering;
}

LUMINARY_FUNCTION RGBF sky_color_no_compute(const vec3 origin, const vec3 ray, const uint8_t state) {
  RGBF sky;
  switch (device.sky.mode) {
    default:
    case LUMINARY_SKY_MODE_DEFAULT: {
      sky = splat_color(0.0f);
    } break;
    case LUMINARY_SKY_MODE_HDRI: {
      sky = sky_hdri_sample(ray);

      const bool include_sun = state & (STATE_FLAG_CAMERA_DIRECTION);
      if (include_sun) {
        const vec3 sky_origin = world_to_sky_transform(origin);

        // HDRI does not include the sun, compute sun visibility
        const bool ray_hits_sun   = sphere_ray_hit_outside(ray, sky_origin, device.sky.sun_pos, SKY_SUN_RADIUS);
        const bool ray_hits_earth = (ocean_is_underwater(origin) == false) ? sph_ray_hit_p0(ray, sky_origin, SKY_EARTH_RADIUS) : false;

        if (ray_hits_sun && ray_hits_earth == false) {
          const RGBF sun_color = sky_get_sun_color(sky_origin, ray);

          sky = add_color(sky, sun_color);
        }
      }
    } break;
    case LUMINARY_SKY_MODE_CONSTANT_COLOR: {
      sky = device.sky.constant_color;
    } break;
  }

  return sky;
}

LUMINARY_FUNCTION RGBF sky_color_main(const vec3 origin, const vec3 ray, const uint8_t state, const PathID& path_id) {
  RGBF sky;
  switch (device.sky.mode) {
    default:
    case LUMINARY_SKY_MODE_DEFAULT: {
      const vec3 sky_origin  = world_to_sky_transform(origin);
      const bool include_sun = state & (STATE_FLAG_CAMERA_DIRECTION);

      sky = sky_get_color(sky_origin, ray, FLT_MAX, include_sun, device.sky.steps, path_id);
    } break;
    case LUMINARY_SKY_MODE_HDRI: {
      sky = sky_hdri_sample(ray);

      const bool include_sun = state & (STATE_FLAG_CAMERA_DIRECTION);
      if (include_sun) {
        const vec3 sky_origin = world_to_sky_transform(origin);

        // HDRI does not include the sun, compute sun visibility
        const bool ray_hits_sun   = sphere_ray_hit_outside(ray, sky_origin, device.sky.sun_pos, SKY_SUN_RADIUS);
        const bool ray_hits_earth = sph_ray_hit_p0(ray, sky_origin, SKY_EARTH_RADIUS);

        if (ray_hits_sun && !ray_hits_earth) {
          const RGBF sun_color = sky_get_sun_color(sky_origin, ray);

          sky = add_color(sky, sun_color);
        }
      }
    } break;
    case LUMINARY_SKY_MODE_CONSTANT_COLOR: {
      sky = device.sky.constant_color;
    } break;
  }

  return sky;
}

#endif /* CU_LUMINARY_SKY_INTEGRATION_H */
