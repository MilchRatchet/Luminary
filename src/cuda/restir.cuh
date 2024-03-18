#ifndef CU_RESTIR_H
#define CU_RESTIR_H

#include "bsdf.cuh"
#include "memory.cuh"
#include "utils.cuh"
#include "volume_utils.cuh"

//
// This file implements light sampling based on ReSTIR. However, I ultimately decided to only use
// weighted reservoir sampling using the BRDF as the target pdf. This is because I want to avoid
// introducing any bias here. Initially, the plan was to reuse visibility that was computed in
// the previous frame. However, to achieve an unbiased implementation we need to evaluate the target
// pdf of the chosen sample for all neighbours. If I include visibility then I must reevaluate visibility
// for each neighbour which is not feasible. [Wym21] suggested not computing visibility but using
// heuristics and MIS weights to keep the bias to a minimum but I am not happy with that.
//
// Since I don't use visibility, there is no need for temporal resampling, the main point behind temporal resampling is
// that the chosen candidate from the initial candidate sampling may be occluded and hence through temporal
// resampling we increase the likelihood that a pixel has a non occluded sample ready for spatial resampling.
// Spatial resampling then redistributes those samples so that any single pixel has a high chance of visiting
// a new and likely unoccluded light. Obviously, we also additionally increase the number of samples any pixel
// has visited exponentially with a non exponential cost.
//
// The increased sample quality through spatial resampling turns out to not be worth much in my case where
// no visibility is used in the target pdf. Hence both temporal and spatial resampling are not part of this
// algorithm. This also has some additional benefits in that it reduces memory consumption since we don't need
// two light sample buffers and because light sampling is equally good for all bounce depths.
//

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Bit20]
// B. Bitterli, C. Wyman, M. Pharr, P. Shirley, A. Lefohn, W. Jarosz,
// "Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting",
// ACM Transactions on Graphics (Proceedings of SIGGRAPH), 39(4), 2020.

// [Wym21]
// C. Wyman, A. Panteleev, "Rearchitecting Spatiotemporal Resampling for Production",
// High-Performance Graphics - Symposium Papers, pp. 23-41, 2021

/**
 * Returns an empty light sample.
 * @result Empty light sample.
 */
__device__ LightSample restir_sample_empty() {
  LightSample s;

  s.seed          = 0;
  s.presampled_id = LIGHT_ID_NONE;
  s.id            = LIGHT_ID_NONE;
  s.weight        = 0.0f;

  return s;
}

__device__ float restir_sample_volume_extinction(const GBufferData data, const vec3 ray, const float max_dist) {
  float extinction = 1.0f;

  if (device.scene.fog.active) {
    const VolumeDescriptor volume = volume_get_descriptor_preset_fog();
    const float2 path             = volume_compute_path(volume, data.position, ray, max_dist);

    if (path.x >= 0.0f) {
      extinction *= expf(-path.y * volume.max_scattering);
    }
  }

  return extinction;
}

/*
 * Surface sample a triangle and return its area and emission luminance.
 * @param triangle Triangle.
 * @param origin Point to sample from.
 * @param area Solid angle of the triangle.
 * @param seed Random seed used to sample the triangle.
 * @param lum Output emission luminance of the triangle at the sampled point.
 * @result Normalized direction to the point on the triangle.
 *
 * Robust triangle sampling.
 */
__device__ vec3
  restir_sample_triangle(const TriangleLight triangle, const vec3 origin, const float2 random, float& solid_angle, RGBF& lum, float& r) {
  float r1 = sqrtf(random.x);
  float r2 = random.y;

  // Map random numbers uniformly into [0.025,0.975].
  r1 = 0.025f + 0.95f * r1;
  r2 = 0.025f + 0.95f * r2;

  const float u = 1.0f - r1;
  const float v = r1 * r2;

  const vec3 p   = add_vector(triangle.vertex, add_vector(scale_vector(triangle.edge1, u), scale_vector(triangle.edge2, v)));
  const vec3 dir = vector_direction_stable(p, origin);

  r = get_length(sub_vector(p, origin));

  const vec3 cross         = cross_product(triangle.edge1, triangle.edge2);
  const float cross_length = get_length(cross);

  // Use that surface * cos_term = 0.5 * |a x b| * |normal(a x b)^Td| = 0.5 * |a x b| * |(a x b)^Td|/|a x b| = 0.5 * |(a x b)^Td|.
  const float surface_cos_term = 0.5f * fabsf(dot_product(cross, dir));

  solid_angle = surface_cos_term / (r * r);

  if (isnan(solid_angle) || isinf(solid_angle) || solid_angle < 1e-7f) {
    solid_angle = 0.0f;
    lum         = get_color(0.0f, 0.0f, 0.0f);
    return get_vector(0.0f, 0.0f, 0.0f);
  }

  const uint16_t illum_tex = device.scene.materials[triangle.material_id].luminance_map;

  const UV tex_coords   = load_triangle_tex_coords(triangle.triangle_id, make_float2(u, v));
  const float4 emission = texture_load(device.ptrs.luminance_atlas[illum_tex], tex_coords);

  lum = scale_color(get_color(emission.x, emission.y, emission.z), device.scene.material.default_material.b * emission.w);

  return dir;
}

__device__ vec3 restir_apply_sample(LightSample light, vec3 pos, const ushort2 pixel, float& solid_angle, RGBF& lum, float& sample_dist) {
  const float2 random = quasirandom_sequence_2D(light.seed, pixel);

  vec3 ray;
  switch (light.presampled_id) {
    case LIGHT_ID_NONE:
    case LIGHT_ID_SUN: {
      vec3 sky_pos = world_to_sky_transform(pos);
      ray          = sample_sphere(device.sun_pos, SKY_SUN_RADIUS, sky_pos, random, solid_angle);
      lum          = scale_color(get_color(1.0f, 1.0f, 1.0f), 2e+04f * device.scene.sky.sun_strength);
      sample_dist  = FLT_MAX;
    } break;
    case LIGHT_ID_TOY:
      ray         = toy_sample_ray(pos, random);
      solid_angle = toy_get_solid_angle(pos);
      lum         = scale_color(device.scene.toy.emission, device.scene.toy.material.b);
      // Approximation, it is not super important what the actual distance is
      sample_dist = get_length(sub_vector(pos, device.scene.toy.position));
      break;
    default: {
      const TriangleLight triangle = load_triangle_light(device.restir.presampled_triangle_lights, light.presampled_id);
      ray                          = restir_sample_triangle(triangle, pos, random, solid_angle, lum, sample_dist);
    } break;
  }

  return ray;
}

__device__ vec3
  restir_apply_sample_shading(const GBufferData data, LightSample light, const ushort2 pixel, RGBF& weight, bool& is_transparent_pass) {
  const float2 random = quasirandom_sequence_2D(light.seed, pixel);

  vec3 ray;
  switch (light.presampled_id) {
    case LIGHT_ID_NONE:
    case LIGHT_ID_SUN: {
      float solid_angle;
      vec3 sky_pos = world_to_sky_transform(data.position);
      ray          = sample_sphere(device.sun_pos, SKY_SUN_RADIUS, sky_pos, random, solid_angle);
    } break;
    case LIGHT_ID_TOY:
      ray = toy_sample_ray(data.position, random);
      break;
    default: {
      const TriangleLight triangle = load_triangle_light(device.restir.presampled_triangle_lights, light.presampled_id);
      ray                          = sample_triangle(triangle, data.position, random);
    } break;
  }

  if (data.flags & G_BUFFER_VOLUME_HIT) {
    is_transparent_pass = true;
    weight              = scale_color(volume_phase_evaluate(data, VOLUME_HIT_TYPE(data.hit_id), ray), light.weight);
  }
  else {
    // TODO: Document why 2PI is the correct 1/PDF.
    weight = scale_color(bsdf_evaluate(data, ray, BSDF_SAMPLING_GENERAL, is_transparent_pass, 2.0f * PI), light.weight);
  }

  return ray;
}

/**
 * Compute the target PDF of a light sample. Currently, this is basically BRDF weight multiplied by general light intensity.
 * @param x Light sample
 * @param data Data to compute the target pdf from.
 * @result Target PDF of light sample.
 */
__device__ float restir_sample_target_pdf(
  LightSample x, const GBufferData data, const RGBF record, const ushort2 pixel, float& primitive_pdf) {
  if (x.presampled_id == LIGHT_ID_NONE) {
    primitive_pdf = 1.0f;
    return 0.0f;
  }

  // We overwrite the local scope copy of the light sample
  x.weight = 1.0f;

  float solid_angle, sample_dist;
  RGBF emission;
  const vec3 ray = restir_apply_sample(x, data.position, pixel, solid_angle, emission, sample_dist);

  primitive_pdf = (solid_angle > 0.0f) ? 1.0f / solid_angle : 0.0f;

  RGBF bsdf_weight;
  if (data.flags & G_BUFFER_VOLUME_HIT) {
    bsdf_weight = volume_phase_evaluate(data, VOLUME_HIT_TYPE(data.hit_id), ray);
  }
  else {
    bool is_transparent_pass;
    bsdf_weight = bsdf_evaluate(data, ray, BSDF_SAMPLING_GENERAL, is_transparent_pass, solid_angle);
  }

  const RGBF color_value = mul_color(mul_color(emission, bsdf_weight), record);
  float value            = luminance(color_value);

  if (isinf(value) || isnan(value)) {
    value = 0.0f;
  }

  if (value > 0.0f) {
    value *= restir_sample_volume_extinction(data, ray, sample_dist);
  }

  return value;
}

/**
 * Samples a light sample using WRS and MIS.
 * @param data Data used for evaluating a lights importance.
 * @param seed Seed used for random number generation, the seed is overwritten on use.
 * @result Sampled light sample.
 */
__device__ LightSample restir_sample_reservoir(const GBufferData data, const RGBF record, const ushort2 pixel) {
  LightSample selected = restir_sample_empty();

  if (!(data.flags & G_BUFFER_REQUIRES_SAMPLING))
    return selected;

  const vec3 sky_pos = world_to_sky_transform(data.position);

  const int sun_visible = !sph_ray_hit_p0(normalize_vector(sub_vector(device.sun_pos, sky_pos)), sky_pos, SKY_EARTH_RADIUS);
  const int toy_visible = (device.scene.toy.active && device.scene.toy.emissive);

  float selection_target_pdf = 1.0f;

  // Importance sample the sun
  if (sun_visible) {
    LightSample sampled;
    sampled.seed          = QUASI_RANDOM_TARGET_RESTIR_DIR;
    sampled.presampled_id = LIGHT_ID_SUN;

    float primitive_pdf;
    const float sampled_target_pdf = restir_sample_target_pdf(sampled, data, record, pixel, primitive_pdf);
    const float sampled_pdf        = primitive_pdf;

    const float weight = (sampled_pdf > 0.0f) ? sampled_target_pdf / sampled_pdf : 0.0f;

    selected.weight += weight;
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RESTIR_CHOICE, pixel) * selected.weight < weight) {
      selected.id            = LIGHT_ID_SUN;
      selected.presampled_id = LIGHT_ID_SUN;
      selected.seed          = sampled.seed;
      selection_target_pdf   = sampled_target_pdf;
    }
  }

  // Importance sample the toy (but only if we are not originating from the toy)
  if (toy_visible && data.hit_id != HIT_TYPE_TOY) {
    LightSample sampled;
    sampled.seed          = QUASI_RANDOM_TARGET_RESTIR_DIR + 1;
    sampled.presampled_id = LIGHT_ID_TOY;

    float primitive_pdf;
    const float sampled_target_pdf = restir_sample_target_pdf(sampled, data, record, pixel, primitive_pdf);
    const float sampled_pdf        = primitive_pdf;

    const float weight = (sampled_pdf > 0.0f) ? sampled_target_pdf / sampled_pdf : 0.0f;

    selected.weight += weight;
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RESTIR_CHOICE + 1, pixel) * selected.weight < weight) {
      selected.id            = LIGHT_ID_TOY;
      selected.presampled_id = LIGHT_ID_TOY;
      selected.seed          = sampled.seed;
      selection_target_pdf   = sampled_target_pdf;
    }
  }

  const uint32_t light_count = (device.scene.material.lights_active) ? (1 << device.restir.light_candidate_pool_size_log2) : 0;
  const int reservoir_size   = (device.scene.triangle_lights_count > 0) ? min(device.restir.initial_reservoir_size, light_count) : 0;

  const float reservoir_sampling_pdf = (1.0f / device.scene.triangle_lights_count);

  // Don't allow triangles to sample themselves.
  uint32_t blocked_light_id = LIGHT_ID_TRIANGLE_ID_LIMIT + 1;
  if (data.hit_id <= LIGHT_ID_TRIANGLE_ID_LIMIT) {
    blocked_light_id = load_triangle_light_id(data.hit_id);
  }

  for (int i = 0; i < reservoir_size; i++) {
    LightSample sampled;
    sampled.seed          = QUASI_RANDOM_TARGET_RESTIR_DIR + 2 + i;
    sampled.presampled_id = quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RESTIR_GENERATION + i, pixel) * light_count;
    sampled.id            = device.ptrs.light_candidates[sampled.presampled_id];

    if (sampled.id == blocked_light_id)
      continue;

    float primitive_pdf;
    const float sampled_target_pdf = restir_sample_target_pdf(sampled, data, record, pixel, primitive_pdf);
    const float sampled_pdf        = reservoir_sampling_pdf * primitive_pdf;

    const float weight = (sampled_pdf > 0.0f) ? sampled_target_pdf / sampled_pdf : 0.0f;

    selected.weight += weight;
    if (quasirandom_sequence_1D(QUASI_RANDOM_TARGET_RESTIR_CHOICE + 2 + i, pixel) * selected.weight < weight) {
      selected.id            = sampled.id;
      selected.presampled_id = sampled.presampled_id;
      selected.seed          = sampled.seed;
      selection_target_pdf   = sampled_target_pdf;
    }
  }

  // Compute the shading weight of the selected light (Probability of selecting the light through WRS)
  if (selected.id == LIGHT_ID_NONE) {
    selected.weight = 0.0f;
  }
  else {
    // We use uniform MIS weights because the images of our distributions are a partition of the set of all lights.
    const float mis_weight = (selected.id <= LIGHT_ID_TRIANGLE_ID_LIMIT) ? 1.0f / reservoir_size : 1.0f;

    selected.weight = mis_weight * selected.weight / selection_target_pdf;
  }

  return selected;
}

__global__ void restir_candidates_pool_generation() {
  if (device.scene.triangle_lights_count == 0)
    return;

  int id = THREAD_ID;

  const int light_sample_bin_count = 1 << device.restir.light_candidate_pool_size_log2;
  while (id < light_sample_bin_count) {
    // TODO: Expose more white noise keys, maybe
    const uint32_t sampled_id =
      random_uint32_t_base(0xfcbd6e15, id + light_sample_bin_count * device.temporal_frames) % device.scene.triangle_lights_count;

    device.ptrs.light_candidates[id]             = sampled_id;
    device.restir.presampled_triangle_lights[id] = device.scene.triangle_lights[sampled_id];

    id += blockDim.x * gridDim.x;
  }
}

#endif /* CU_RESTIR_H */
