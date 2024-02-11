#include "brdf.cuh"
#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

__global__ void unittest_brdf_kernel(float* bounce, float* light, uint32_t total, uint32_t steps, uint32_t iterations) {
  unsigned int id = THREAD_ID;

  while (id < total) {
    const unsigned int x = id % steps;
    const unsigned int y = id / steps;

    const float smoothness = (1.0f / (steps - 1)) * x;
    const float metallic   = (1.0f / (steps - 1)) * y;

    float sum_bounce = 0.0f;
    float sum_light  = 0.0f;

    for (int i = 0; i < iterations; i++) {
      const float ran1 = 0.5f * PI * (1.0f - sqrtf(white_noise()));
      const float ran2 = 2.0f * PI * white_noise();
      const float ran3 = 0.5f * PI * (1.0f - sqrtf(white_noise()));
      const float ran4 = 2.0f * PI * white_noise();
      const vec3 V     = angles_to_direction(ran1, ran2);

      BRDFInstance brdf =
        brdf_get_instance(get_RGBAF(1.0f, 1.0f, 1.0f, 1.0f), V, get_vector(0.0f, 1.0f, 0.0f), 1.0f - smoothness, metallic);

      bool dummy;
      brdf = brdf_sample_ray(brdf, 0, dummy);

      float weight = luminance(brdf.term);

      sum_bounce += weight;

      brdf.L    = angles_to_direction(ran3, ran4);
      brdf.term = get_color(1.0f, 1.0f, 1.0f);

      brdf = brdf_evaluate(brdf);

      weight = luminance(brdf.term);

      sum_light += weight;
    }

    bounce[id] = sum_bounce / iterations;
    light[id]  = sum_light / iterations;

    id += blockDim.x * gridDim.x;
  }
}
