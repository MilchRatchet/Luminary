#include "math.cuh"
#include "sky.cuh"
#include "utils.cuh"

// [Bru17]
__global__ void sky_compute_transmittance_lut(float4* transmittance_tex_lower, float4* transmittance_tex_higher) {
  unsigned int id = THREAD_ID;

  const int amount = SKY_TM_TEX_WIDTH * SKY_TM_TEX_HEIGHT;

  while (id < amount) {
    const int x = id % SKY_TM_TEX_WIDTH;
    const int y = id / SKY_TM_TEX_WIDTH;

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

    transmittance_tex_lower[x + y * SKY_TM_TEX_WIDTH]  = spectrum_split_low(transmittance);
    transmittance_tex_higher[x + y * SKY_TM_TEX_WIDTH] = spectrum_split_high(transmittance);

    id += blockDim.x * gridDim.x;
  }
}
