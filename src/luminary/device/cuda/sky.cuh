#ifndef CU_SKY_H
#define CU_SKY_H

#include "math.cuh"
#include "memory.cuh"
#include "sky_integration.cuh"
#include "sky_utils.cuh"
#include "utils.cuh"

//
// In this atmosphere rendering implementation, the single scattering is computed using ray-marching. The transmittance and multiscattering
// are precomputed as in [Bru17] and [Hil20] respectively. We extended the model using the model of [Wil21] as reference. For this, we
// modified their public GUI implementation so that we could compare our model to theirs. This modification is found in
// https://github.com/MilchRatchet/pragueskymodel. Our final model found in that repository has a lot more parameters than the
// implementation that we use in Luminary. Some of these parameters were found to be superfluous and possibly harmful for an efficient
// implementation. In the following, we summarize our additions over the common real-time methods of [Bru17] and [Hil20]. We added the layer
// of water soluble aerosols which contribute to the Mie scattering. Further, we found that using a spectrum of 8 different wavelengths with
// equal gaps in the range [415,635] allows us to obtain a visual look that is similar to that of the model in [Wil21]. Using only three
// wavelengths as in [Hil20] meant that the sky would look very purple during sunsets. Other spectrums could also work well for our
// purposes but we wanted a multiple of four to optimally use the texture sampling capabilities of the hardware and we did not want to
// actually try and solve the problem of finding four optimal wavelengths, if they exist.
//

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Hil20]
// Sébastien Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique", Computer Graphics Forum, 2020
// https://github.com/sebh/UnrealEngineSkyAtmosphere

// [Bru17]
// Eric Bruneton, "Precomputed Atmospheric Scattering: a New Implementation", 2017
// https://ebruneton.github.io/precomputed_atmospheric_scattering/

// [Wil21]
// Alexander Wilkie, Petr Vevoda, Thomas Bashford-Rogers, Lukas Hosek, Tomas Iser, Monika Kolarova, Tobias Rittig and Jaroslav Krivanek,
// "A Fitted Radiance and Attenuation Model for Realistic Atmospheres", Association for Computing Machinery, 40 (4), pp. 1-14, 2021
// https://cgg.mff.cuni.cz/publications/skymodel-2021/

////////////////////////////////////////////////////////////////////
// Sky LUT function
////////////////////////////////////////////////////////////////////

// [Bru17]
LUMINARY_FUNCTION Spectrum sky_compute_transmittance_optical_depth(const float r, const float mu) {
  const int steps = 2500;

  // Distance to top of atmosphere
  const float disc = r * r * (mu * mu - 1.0f) + SKY_ATMO_RADIUS * SKY_ATMO_RADIUS;
  const float dist = fmaxf(-r * mu + sqrtf(fmaxf(0.0f, disc)), 0.0f);

  const float step_size = dist / steps;

  Spectrum depth = spectrum_set1(0.0f);

  for (int i = 0; i <= steps; i++) {
    const float reach  = i * step_size;
    const float height = sqrtf(reach * reach + 2.0f * r * mu * reach + r * r) - SKY_EARTH_RADIUS;

    const float density_rayleigh = sky_rayleigh_density(height) * device.sky.rayleigh_density;
    const float density_mie      = sky_mie_density(height) * device.sky.mie_density;
    const float density_ozone    = sky_ozone_density(height) * device.sky.ozone_density;

    const Spectrum extinction_rayleigh = spectrum_scale(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
    const float extinction_mie         = SKY_MIE_EXTINCTION * density_mie;
    const Spectrum extinction_ozone    = spectrum_scale(SKY_OZONE_EXTINCTION, density_ozone);

    const Spectrum extinction = spectrum_add(spectrum_add(extinction_rayleigh, spectrum_set1(extinction_mie)), extinction_ozone);

    const float w = (i == 0 || i == steps) ? 0.5f : 1.0f;

    depth = spectrum_add(depth, spectrum_scale(extinction, w * step_size));
  }

  return depth;
}

// [Bru17]
LUMINARY_KERNEL void sky_compute_transmittance_lut(KernelArgsSkyComputeTransmittanceLUT args) {
  unsigned int id = THREAD_ID;

  const int amount = SKY_TM_TEX_WIDTH * SKY_TM_TEX_HEIGHT;

  while (id < amount) {
    const int y = id / SKY_TM_TEX_WIDTH;
    const int x = id - y * SKY_TM_TEX_WIDTH;

    float fx = ((float) x + 0.5f) / SKY_TM_TEX_WIDTH;
    float fy = ((float) y + 0.5f) / SKY_TM_TEX_HEIGHT;

    fx = sky_sub_to_unit_uv(fx, SKY_TM_TEX_WIDTH);
    fy = sky_sub_to_unit_uv(fy, SKY_TM_TEX_HEIGHT);

    const float H   = sqrtf(SKY_ATMO_RADIUS * SKY_ATMO_RADIUS - SKY_EARTH_RADIUS * SKY_EARTH_RADIUS);
    const float rho = H * fy;
    const float r   = sqrtf(rho * rho + SKY_EARTH_RADIUS * SKY_EARTH_RADIUS);

    const float d_min = SKY_ATMO_RADIUS - r;
    const float d_max = rho + H;
    const float d     = d_min + fx * (d_max - d_min);

    float mu = (d == 0.0f) ? 1.0f : (H * H - rho * rho - d * d) / (2.0f * r * d);
    mu       = fminf(1.0f, fmaxf(-1.0f, mu));

    const Spectrum optical_depth = sky_compute_transmittance_optical_depth(r, mu);
    const Spectrum transmittance = spectrum_exp(spectrum_scale(optical_depth, -1.0f));

    args.dst_low[x + y * SKY_TM_TEX_WIDTH]  = spectrum_split_low(transmittance);
    args.dst_high[x + y * SKY_TM_TEX_WIDTH] = spectrum_split_high(transmittance);

    id += blockDim.x * gridDim.x;
  }
}

struct msScatteringResult {
  Spectrum L;
  Spectrum multiScatterAs1;
} typedef msScatteringResult;

// [Hil20]
LUMINARY_FUNCTION msScatteringResult sky_compute_multiscattering_integration(
  const vec3 origin, const vec3 ray, const vec3 sun_pos, const DeviceTextureObject transmission_low,
  const DeviceTextureObject transmission_high) {
  msScatteringResult result;

  result.L               = spectrum_set1(0.0f);
  result.multiScatterAs1 = spectrum_set1(0.0f);

  const float2 path = sky_compute_path(origin, ray, SKY_EARTH_RADIUS, SKY_ATMO_RADIUS);

  if (path.y == -FLT_MAX) {
    return result;
  }

  const float start    = path.x;
  const float distance = path.y;

  if (distance > 0.0f) {
    const int steps = 500;
    float reach     = start;
    float step_size;

    const float light_angle = sample_sphere_solid_angle(sun_pos, SKY_SUN_RADIUS, origin);

    Spectrum transmittance = spectrum_set1(1.0f);

    const JendersieEonParams mie_params = jendersie_eon_phase_parameters(device.sky.mie_diameter);

    for (int i = 0; i < steps; i++) {
      const float newReach = start + distance * (i + 0.3f) / steps;
      step_size            = newReach - reach;
      reach                = newReach;

      const vec3 pos     = add_vector(origin, scale_vector(ray, reach));
      const float height = sky_height(pos);

      const vec3 ray_scatter     = normalize_vector(sub_vector(sun_pos, pos));
      const float cos_angle      = dot_product(ray, ray_scatter);
      const float phase_rayleigh = sky_rayleigh_phase(cos_angle);
      const float phase_mie      = sky_mie_phase(cos_angle, mie_params);

      const float zenith_cos_angle = dot_product(normalize_vector(pos), ray_scatter);

      TextureLoadArgs tex_load_args = texture_get_default_args();
      tex_load_args.flip_v          = false;
      tex_load_args.apply_gamma     = false;

      const UV transmittance_uv       = sky_transmittance_lut_uv(height, zenith_cos_angle);
      const float4 transmittance_low  = texture_load(transmission_low, transmittance_uv, tex_load_args);
      const float4 transmittance_high = texture_load(transmission_high, transmittance_uv, tex_load_args);
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
      const Spectrum phaseTimesScattering =
        spectrum_add(spectrum_scale(scattering_rayleigh, phase_rayleigh), spectrum_set1(scattering_mie * phase_mie));

      const float shadow = sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS) ? 0.0f : 1.0f;
      const Spectrum S   = spectrum_scale(spectrum_mul(extinction_sun, phaseTimesScattering), shadow * light_angle);

      Spectrum step_transmittance = extinction;
      step_transmittance          = spectrum_scale(step_transmittance, -step_size);
      step_transmittance          = spectrum_exp(step_transmittance);

      const Spectrum ssInt = spectrum_mul(spectrum_sub(S, spectrum_mul(S, step_transmittance)), spectrum_inv(extinction));
      const Spectrum msInt = spectrum_mul(spectrum_sub(scattering, spectrum_mul(scattering, step_transmittance)), spectrum_inv(extinction));

      result.L               = spectrum_add(result.L, spectrum_mul(ssInt, transmittance));
      result.multiScatterAs1 = spectrum_add(result.multiScatterAs1, spectrum_mul(msInt, transmittance));

      transmittance = spectrum_mul(transmittance, step_transmittance);
    }
  }

  return result;
}

// [Hil20]
// This kernel does not use default Luminary launch bounds, hence it may not be marked as LUMINARY_KERNEL
LUMINARY_KERNEL_NO_BOUNDS void sky_compute_multiscattering_lut(KernelArgsSkyComputeMultiscatteringLUT args) {
  const int x = blockIdx.x;
  const int y = blockIdx.y;

  float fx = ((float) x + 0.5f) / SKY_MS_TEX_SIZE;
  float fy = ((float) y + 0.5f) / SKY_MS_TEX_SIZE;

  fx = sky_sub_to_unit_uv(fx, SKY_MS_TEX_SIZE);
  fy = sky_sub_to_unit_uv(fy, SKY_MS_TEX_SIZE);

  __shared__ Spectrum luminance_shared[SKY_MS_ITER];
  __shared__ Spectrum multiscattering_shared[SKY_MS_ITER];

  const float cos_angle = fx * 2.0f - 1.0f;
  const vec3 sun_dir    = get_vector(0.0f, cos_angle, sqrtf(__saturatef(1.0f - cos_angle * cos_angle)));
  const float height    = SKY_EARTH_RADIUS + __saturatef(fy + SKY_HEIGHT_OFFSET) * (SKY_ATMO_HEIGHT - SKY_HEIGHT_OFFSET);

  const vec3 pos     = get_vector(0.0f, height, 0.0f);
  const vec3 sun_pos = scale_vector(sun_dir, SKY_SUN_DISTANCE);

  const float sqrt_sample = (float) SKY_MS_BASE;

  const float a     = threadIdx.x / SKY_MS_BASE;
  const float b     = (threadIdx.x - ((threadIdx.x / SKY_MS_BASE) * SKY_MS_BASE));
  const float randA = a / sqrt_sample;
  const float randB = b / sqrt_sample;
  const vec3 ray    = sample_ray_sphere(2.0f * randA - 1.0f, randB);

  msScatteringResult result =
    sky_compute_multiscattering_integration(pos, ray, sun_pos, args.transmission_low_tex, args.transmission_high_tex);

  luminance_shared[threadIdx.x]       = result.L;
  multiscattering_shared[threadIdx.x] = result.multiScatterAs1;

  for (int i = SKY_MS_ITER >> 1; i > 0; i = i >> 1) {
    __syncthreads();
    if (threadIdx.x < i) {
      luminance_shared[threadIdx.x]       = spectrum_add(luminance_shared[threadIdx.x], luminance_shared[threadIdx.x + i]);
      multiscattering_shared[threadIdx.x] = spectrum_add(multiscattering_shared[threadIdx.x], multiscattering_shared[threadIdx.x + i]);
    }
  }

  if (threadIdx.x > 0)
    return;

  Spectrum luminance       = spectrum_scale(luminance_shared[0], 1.0f / (sqrt_sample * sqrt_sample));
  Spectrum multiscattering = spectrum_scale(multiscattering_shared[0], 1.0f / (sqrt_sample * sqrt_sample));

  const Spectrum multiScatteringContribution = spectrum_inv(spectrum_sub(spectrum_set1(1.0f), multiscattering));

  const Spectrum L = spectrum_scale(spectrum_mul(luminance, multiScatteringContribution), device.sky.multiscattering_factor);

  args.dst_low[x + y * SKY_MS_TEX_SIZE]  = spectrum_split_low(L);
  args.dst_high[x + y * SKY_MS_TEX_SIZE] = spectrum_split_high(L);
}

////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////

LUMINARY_KERNEL void sky_process_tasks() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_SKY];
  const int task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_SKY];

  LUMINARY_ASSUME(task_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    const DeviceTask task            = task_load(task_base_address);

    if ((device.sky.mode != LUMINARY_SKY_MODE_DEFAULT) && ((task.state & STATE_FLAG_ALLOW_AMBIENT) == 0))
      continue;

    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

    RGBF sky = sky_color_main(task.origin, task.ray, task.state, task.path_id);
    sky      = mul_color(sky, record_unpack(throughput.record));

    write_beauty_buffer(sky, throughput.results_index);
  }
}

LUMINARY_KERNEL void sky_process_tasks_debug() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[TASK_ADDRESS_OFFSET_SKY];
  const int task_offset = device.ptrs.task_offsets[TASK_ADDRESS_OFFSET_SKY];

  LUMINARY_ASSUME(task_count <= MAXIMUM_TASKS_PER_THREAD);

  for (int i = 0; i < task_count; i++) {
    HANDLE_DEVICE_ABORT();

    const uint32_t task_base_address      = task_get_base_address(task_offset + i, TASK_STATE_BUFFER_INDEX_POSTSORT);
    const DeviceTask task                 = task_load(task_base_address);
    const DeviceTaskThroughput throughput = task_throughput_load(task_base_address);

    RGBF result;
    switch (device.settings.shading_mode) {
      case LUMINARY_SHADING_MODE_ALBEDO:
        result = sky_color_main(task.origin, task.ray, STATE_FLAG_CAMERA_DIRECTION, task.path_id);
        break;
      case LUMINARY_SHADING_MODE_IDENTIFICATION:
        result = get_color(0.0f, 0.63f, 1.0f);
        break;
      default:
        result = splat_color(0.0f);
        break;
    }

    write_beauty_buffer(result, throughput.results_index);
  }
}

#endif /* CU_SKY_H */
