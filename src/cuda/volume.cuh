#ifndef CU_VOLUME_H
#define CU_VOLUME_H

#include "math.cuh"
#include "utils.cuh"

//
// This implements a volume renderer. These volumes are homogenous and bound by a maximum height relative to the planet's
// surface. Closed form tracking is used to solve any light interaction with such a volume.
// While this implementation is mostly generic, some details are handled simply with the constraint that Luminary only has
// two types of volumes, fog and ocean water.
//

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [FonWKH17]
// J. Fong, M. Wrenninge, C. Kulla, R. Habel, "Production Volume Rendering", SIGGRAPH 2017 Course, 2017
// https://graphics.pixar.com/library/ProductionVolumeRendering/

#define FOG_DENSITY (0.001f * device.scene.fog.density)

struct VolumeDescriptor {
  // TODO: Correctly pass descriptor to G-Buffer and use in ReSTIR.
  VolumeType type;
  float water_droplet_diameter;
  RGBF absorption;
  RGBF scattering;
  float avg_transmittance;
  float max_height; /* Sky space */
  float min_height;
} typedef VolumeDescriptor;

__device__ RGBF volume_get_transmittance(const VolumeDescriptor volume) {
  return add_color(volume.absorption, volume.scattering);
}

__device__ VolumeDescriptor volume_get_descriptor_preset_fog() {
  VolumeDescriptor volume;

  volume.type                   = VOLUME_TYPE_FOG;
  volume.water_droplet_diameter = device.scene.fog.droplet_diameter;
  volume.absorption             = get_color(0.0f, 0.0f, 0.0f);
  volume.scattering             = get_color(FOG_DENSITY, FOG_DENSITY, FOG_DENSITY);
  volume.avg_transmittance      = FOG_DENSITY;
  volume.max_height             = world_to_sky_scale(device.scene.fog.height);
  volume.min_height             = world_to_sky_scale(OCEAN_MAX_HEIGHT);

  return volume;
}

__device__ VolumeDescriptor volume_get_descriptor_preset_ocean() {
  VolumeDescriptor volume;

  volume.type                   = VOLUME_TYPE_OCEAN;
  volume.water_droplet_diameter = 50.0f;
  volume.absorption             = OCEAN_ABSORPTION;
  volume.scattering             = OCEAN_SCATTERING;
  volume.max_height             = world_to_sky_scale(OCEAN_MIN_HEIGHT);
  volume.min_height             = 0.0f;

  volume.avg_transmittance = RGBF_avg(volume_get_transmittance(volume));

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

/*
 * Computes the start and length of a ray path through a volume.
 * @param origin Start point of ray in world space.
 * @param ray Direction of ray.
 * @param limit Maximum distance a ray may travel in world space.
 * @result Two floats:
 *                  - [x] = Start in world space
 *                  - [y] = Distance through fog in world space
 */
__device__ float2 volume_compute_path(const VolumeDescriptor volume, const vec3 origin, const vec3 ray, const float limit) {
  const vec3 sky_origin = world_to_sky_transform(origin);
  const float2 path     = sky_compute_path(sky_origin, ray, SKY_EARTH_RADIUS, SKY_EARTH_RADIUS + volume.max_height);

  if (path.y == -FLT_MAX)
    return make_float2(-FLT_MAX, -FLT_MAX);

  const float start    = fmaxf(sky_to_world_scale(path.x), 0.0f);
  const float distance = fmaxf(fminf(sky_to_world_scale(path.y), limit - start), 0.0f);

  return make_float2(start, distance);
}

/*
 * Computes a random volume intersection point by perfectly importance sampling the transmittance
 * based pdf.
 *
 * @param volume VolumeDescriptor of the corresponding volume.
 * @param origin Origin of ray in world space.
 * @param ray Direction of ray.
 * @param limit Depth of ray without scattering in world space.
 * @result Distance of intersection point and origin in world space.
 */
__device__ float volume_sample_intersection(const VolumeDescriptor volume, const vec3 origin, const vec3 ray, const float limit = FLT_MAX) {
  const float2 path = volume_compute_path(volume, origin, ray, limit);

  if (path.y == -FLT_MAX)
    return FLT_MAX;

  const float start    = path.x;
  const float max_dist = path.y;

  // [FonWKH17] Equation 15
  const float t = (-logf(1.0f - white_noise())) / volume.avg_transmittance;

  if (t > max_dist)
    return FLT_MAX;

  return start + t;
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

__global__ void volume_generate_g_buffer() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count  = device.ptrs.task_counts[id * 6 + 4];
  const int task_offset = device.ptrs.task_offsets[id * 5 + 4];

  for (int i = 0; i < task_count; i++) {
    VolumeTask task = load_volume_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    LightEvalData data;
    data.flags     = LIGHT_EVAL_DATA_REQUIRES_SAMPLING | LIGHT_EVAL_DATA_VOLUME_HIT;
    data.normal    = get_vector(0.0f, 0.0f, 0.0f);
    data.position  = task.position;
    data.V         = scale_vector(ray, -1.0f);
    data.roughness = 1.0f;
    data.metallic  = 0.0f;

    store_light_eval_data(data, pixel);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 10) void volume_process_events() {
  const int task_count = device.trace_count[threadIdx.x + blockIdx.x * blockDim.x];

  for (int i = 0; i < task_count; i++) {
    const int offset    = get_task_address(i);
    TraceTask task      = load_trace_task(device.trace_tasks + offset);
    const float2 result = __ldcs((float2*) (device.ptrs.trace_results + offset));

    float depth     = result.x;
    uint32_t hit_id = __float_as_uint(result.y);

    const int pixel = task.index.x + task.index.y * device.width;
    RGBF record     = load_RGBF(device.records + pixel);

    if (device.scene.fog.active) {
      const VolumeDescriptor volume = volume_get_descriptor_preset_fog();
      if (device.iteration_type == TYPE_LIGHT) {
        const float t = volume_compute_path(volume, task.origin, task.ray, depth).y;

        record = scale_color(record, expf(-t * volume.avg_transmittance));
      }
      else {
        const float volume_dist = volume_sample_intersection(volume, task.origin, task.ray, depth);

        if (volume_dist < depth) {
          depth  = volume_dist;
          hit_id = VOLUME_FOG_HIT;
        }
      }
    }

    if (device.scene.ocean.active) {
      const VolumeDescriptor volume = volume_get_descriptor_preset_ocean();
      if (device.iteration_type == TYPE_LIGHT) {
        const float t = volume_compute_path(volume, task.origin, task.ray, depth).y;

        RGBF volume_transmittance = volume_get_transmittance(volume);

        record.r *= expf(-t * volume_transmittance.r);
        record.g *= expf(-t * volume_transmittance.g);
        record.b *= expf(-t * volume_transmittance.b);
      }
      else {
        const float volume_dist = volume_sample_intersection(volume, task.origin, task.ray, depth);

        if (volume_dist < depth) {
          depth  = volume_dist;
          hit_id = VOLUME_OCEAN_HIT;
        }
      }
    }

    __stcs((float2*) (device.ptrs.trace_results + offset), make_float2(depth, __uint_as_float(hit_id)));
    store_RGBF(device.records + pixel, record);
  }
}

__global__ __launch_bounds__(THREADS_PER_BLOCK, 9) void volume_process_tasks() {
  const int id = threadIdx.x + blockIdx.x * blockDim.x;

  const int task_count   = device.ptrs.task_counts[id * 6 + 4];
  const int task_offset  = device.ptrs.task_offsets[id * 5 + 4];
  int light_trace_count  = device.ptrs.light_trace_count[id];
  int bounce_trace_count = device.ptrs.bounce_trace_count[id];

  for (int i = 0; i < task_count; i++) {
    VolumeTask task = load_volume_task(device.trace_tasks + get_task_address(task_offset + i));
    const int pixel = task.index.y * device.width + task.index.x;

    vec3 ray;
    ray.x = cosf(task.ray_xz) * cosf(task.ray_y);
    ray.y = sinf(task.ray_y);
    ray.z = sinf(task.ray_xz) * cosf(task.ray_y);

    const VolumeDescriptor volume = volume_get_descriptor_preset((VolumeType) task.volume_type);

    RGBF record = device.records[pixel];

    {
      // TODO: Importance sample phase function using inverse CDF LUT.
      const vec3 bounce_ray = angles_to_direction(white_noise() * PI, white_noise() * 2.0f * PI);
      const float cos_angle = dot_product(ray, bounce_ray);
      const float phase     = jendersie_eon_phase_function(cos_angle, volume.water_droplet_diameter);

      const float weight = 4.0f * PI * phase;

      device.ptrs.bounce_records[pixel] = scale_color(record, weight);

      TraceTask bounce_task;
      bounce_task.origin = task.position;
      bounce_task.ray    = bounce_ray;
      bounce_task.index  = task.index;

      store_trace_task(device.ptrs.bounce_trace + get_task_address(bounce_trace_count++), bounce_task);
    }

    const int light_occupied = (device.ptrs.state_buffer[pixel] & STATE_LIGHT_OCCUPIED);

    if (!light_occupied) {
      LightSample light = load_light_sample(device.ptrs.light_samples, pixel);

      uint32_t light_history_buffer_entry = LIGHT_ID_ANY;

      BRDFInstance brdf = brdf_get_instance_scattering();

      if (light.weight > 0.0f) {
        BRDFInstance brdf_sample = brdf_apply_sample_scattering(brdf, light, task.position, volume.water_droplet_diameter);

        device.ptrs.light_records[pixel] = mul_color(record, brdf_sample.term);
        light_history_buffer_entry       = light.id;

        TraceTask light_task;
        light_task.origin = task.position;
        light_task.ray    = brdf_sample.L;
        light_task.index  = task.index;

        store_trace_task(device.ptrs.light_trace + get_task_address(light_trace_count++), light_task);
      }

      device.ptrs.light_sample_history[pixel] = light_history_buffer_entry;
    }
  }

  device.ptrs.light_trace_count[id]  = light_trace_count;
  device.ptrs.bounce_trace_count[id] = bounce_trace_count;
}

#endif /* CU_VOLUME_H */
