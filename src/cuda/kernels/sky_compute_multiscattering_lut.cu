#include "math.cuh"
#include "sky.cuh"
#include "utils.cuh"

// [Hil20]
__global__ void sky_compute_multiscattering_lut(float4* multiscattering_tex_lower, float4* multiscattering_tex_higher) {
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

  msScatteringResult result = sky_compute_multiscattering_integration(pos, ray, sun_pos);

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

  const Spectrum L = spectrum_scale(spectrum_mul(luminance, multiScatteringContribution), device.scene.sky.multiscattering_factor);

  multiscattering_tex_lower[x + y * SKY_MS_TEX_SIZE]  = spectrum_split_low(L);
  multiscattering_tex_higher[x + y * SKY_MS_TEX_SIZE] = spectrum_split_high(L);
}
