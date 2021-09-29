#ifndef CU_ASVGF_H
#define CU_ASVGF_H

/*
 * This is based on
 *
 * [Schied et al. 2018] Gradient Estimation for Real-Time Adaptive Temporal
 *  Filtering; Christoph Schied, Christoph Peters, Carsten Dachsbacher;
 *  Proceedings of the ACM on Computer Graphics and Interactive Techniques
 *  http://cg.ivd.kit.edu/atf.php
 *
 */

#include "math.cuh"
#include "utils.cuh"

__global__ void temporal_accumulate() {
  for (int offset = threadIdx.x + blockIdx.x * blockDim.x; offset < device_amount; offset += blockDim.x * gridDim.x) {
    RGBF buffer = device_frame_buffer[offset];

    vec3 hit = device_world_space_hit[offset];

    vec4 pos;
    pos.x = hit.x;
    pos.y = hit.y;
    pos.z = hit.z;
    pos.w = 1.0f;

    vec4 prev_pixel = transform_vec4(device_projection, transform_vec4(device_view_space, pos));

    prev_pixel.x /= -prev_pixel.w;
    prev_pixel.y /= -prev_pixel.w;

    prev_pixel.x = device_width * (1.0f - prev_pixel.x) * 0.5f;
    prev_pixel.y = device_height * (prev_pixel.y + 1.0f) * 0.5f;

    const int prev_x = prev_pixel.x - 0.5f;
    const int prev_y = prev_pixel.y - 0.5f;

    float2 w = make_float2(prev_pixel.x - 0.5f - floorf(prev_pixel.x - 0.5f), prev_pixel.y - 0.5f - floorf(prev_pixel.y - 0.5f));

    const float l = luminance(buffer);

    RGBF temporal       = get_color(0.0f, 0.0f, 0.0f);
    TraceResult trace   = device_frame_trace_buffer[offset];
    vec3 normal         = device_frame_normal_buffer[offset];
    float slope         = device_frame_slope_buffer[offset];
    float2 moments      = make_float2(l, l * l);
    float2 prev_moments = make_float2(0.0f, 0.0f);
    float sum_weights   = 0.0f;
    float history       = 0.0f;

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        const int x = prev_x + i;
        const int y = prev_y + j;

        if (x < 0 || x >= device_width || y < 0 || y >= device_height)
          continue;

        TraceResult prev_trace = device_frame_trace_temporal[y * device_width + x];

        if (!temporalTraceTest(trace, prev_trace, slope))
          continue;

        vec3 prev_normal = device_frame_normal_temporal[y * device_width + x];

        if (!temporalNormalTest(normal, prev_normal))
          continue;

        const float weight = ((i == 0) ? (1.0f - w.x) : w.x) * ((j == 0) ? (1.0f - w.y) : w.y);

        temporal = add_color(temporal, scale_color(device_frame_temporal[y * device_width + x], weight));
        history += device_frame_history_temporal[y * device_width + x] * weight;

        float2 m = device_frame_moments_temporal[y * device_width + x];
        prev_moments.x += m.x * weight;
        prev_moments.y += m.y * weight;

        sum_weights += weight;
      }
    }

    if (sum_weights > 0.01f) {
      temporal = scale_color(temporal, 1.0f / sum_weights);
      prev_moments.x *= 1.0f / sum_weights;
      prev_moments.y *= 1.0f / sum_weights;
      history *= 1.0f / sum_weights;

      float alpha         = fmaxf(0.0f, 1.0f / (history + 1.0f));
      float moments_alpha = fmaxf(0.6f, 1.0f / (history + 1.0f));
      buffer              = add_color(scale_color(buffer, alpha), scale_color(temporal, 1.0f - alpha));

      moments.x = lerp(moments.x, prev_moments.x, moments_alpha);
      moments.y = lerp(moments.y, prev_moments.y, moments_alpha);

      history = fminf(64.0f, history + 1.0f);
    }
    else {
      history = 1.0f;
    }

    device_frame_output[offset]           = buffer;
    device_frame_history_buffer[offset]   = history;
    device_frame_moments_temporal[offset] = moments;
  }
}

__global__ void estimate_variance() {
  for (int offset = threadIdx.x + blockIdx.x * blockDim.x; offset < device_amount; offset += blockDim.x * gridDim.x) {
    const int x = offset % device_width;
    const int y = offset / device_width;

    float2 moments    = device_frame_moments_temporal[offset];
    float history     = device_frame_history_buffer[offset];
    RGBF color        = device_frame_output[offset];
    const float slope = device_frame_slope_buffer[offset];

    const float history_threshold = 4.0f;

    RGBF color_spatial      = get_color(0.0f, 0.0f, 0.0f);
    float2 moments_spatial  = make_float2(0.0f, 0.0f);
    float2 moments_temporal = make_float2(0.0f, 0.0f);

    TraceResult trace = device_frame_trace_buffer[offset];

    if (trace.depth < 0.0f)
      continue;

    if (history < history_threshold) {
      float l = luminance(color);
      moments.x += l;
      moments.y += l * l;

      vec3 normal = device_frame_normal_buffer[offset];

      float sum_weights = 1.0f;
      const int r       = (history > 1.0f) ? 2 : 3;

      for (int i = -r; i <= r; i++) {
        for (int j = -r; j <= r; j++) {
          if (i != 0 || j != 0) {
            const int n_x = x + j;
            const int n_y = y + i;

            if (n_x < 0 || n_x >= device_width || n_y < 0 || n_y >= device_height)
              continue;

            const int n_offset = n_x + n_y * device_width;

            RGBF n_color = device_frame_output[n_offset];

            l = luminance(n_color);

            TraceResult n_trace = device_frame_trace_buffer[n_offset];
            vec3 n_normal       = device_frame_normal_buffer[n_offset];

            const float weight_depth  = fabsf(trace.depth - n_trace.depth) / (slope * i * i + j * j + 1e-2);
            const float weight_normal = powf(fmaxf(0.0f, dot_product(normal, n_normal)), 128.0f);

            float weight = expf(-weight_depth) * weight_normal * (trace.hit_id == n_trace.hit_id);

            if (isnan(weight))
              weight = 0.0f;

            sum_weights += weight;

            moments.x += l * weight;
            moments.y += l * l * weight;
            color = add_color(color, scale_color(n_color, weight));
          }
        }
      }

      moments.x *= 1.0f / sum_weights;
      color = scale_color(color, 1.0f / sum_weights);

      moments_spatial = moments;
      color_spatial   = color;

      device_frame_variance_buffer[offset] =
        (1.0f + 3.0f * (1.0f - history / history_threshold)) * fmaxf(0.0f, moments.y - moments.x * moments.x);
      device_frame_output[offset] = color;
    }
    else {
      device_frame_variance_buffer[offset] = fmaxf(0.0f, moments.y - moments.x * moments.x);
      device_frame_output[offset]          = color;
    }
  }
}

__constant__ float gaussian_kernel[3][3] = {
  {1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0},
  {1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0},
  {1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0}};

__device__ float sigma_luminance(const float variance, const int x, const int y) {
  float sum = variance * gaussian_kernel[1][1];

  for (int i = -1; i < 2; i++) {
    for (int j = -1; j < 2; j++) {
      if (i != 0 || j != 0) {
        const int n_x = x + j;
        const int n_y = y + i;

        if (n_x < 0 || n_x >= device_width || n_y < 0 || n_y >= device_height)
          continue;

        const int n_offset     = n_x + n_y * device_width;
        const float n_variance = device_frame_variance_buffer[n_offset];
        const float weight     = gaussian_kernel[j + 1][i + 1];
        sum += n_variance * weight;
      }
    }
  }

  return sqrtf(fmaxf(sum, 0.0f));
}

#define ASVGF_TAP(i, j, w)                                                                                     \
  {                                                                                                            \
    const int p_x = x + j * filter_scale;                                                                      \
    const int p_y = y + i * filter_scale;                                                                      \
                                                                                                               \
    if (p_x >= 0 && p_x < device_width && p_y >= 0 && p_y < device_height) {                                   \
      const int p_offset        = p_x + p_y * device_width;                                                    \
      const RGBF p_color        = device_frame_buffer[p_offset];                                               \
      const float p_variance    = device_frame_variance_buffer[p_offset];                                      \
      const vec3 p_normal       = device_frame_normal_buffer[p_offset];                                        \
      const TraceResult p_trace = device_frame_trace_buffer[p_offset];                                         \
      const float p_luminance   = luminance(p_color);                                                          \
      const float offset_len    = i * filter_scale * i * filter_scale + j * filter_scale * j * filter_scale;   \
                                                                                                               \
      const float weight_luminance = fabsf(p_luminance - l) / (sigma_l + 1e-10f);                              \
      const float weight_depth     = 3.0f * fabsf(p_trace.depth - trace.depth) / (slope * offset_len + 1e-2f); \
      const float weight_normal    = powf(fmaxf(0.0f, dot_product(p_normal, normal)), 128.0f);                 \
                                                                                                               \
      const float weight = expf(-weight_luminance * weight_luminance - weight_depth) * w * weight_normal;      \
                                                                                                               \
      sum_color = add_color(sum_color, scale_color(p_color, weight));                                          \
      sum_variance += weight * weight * p_variance;                                                            \
      sum_weights += weight;                                                                                   \
    }                                                                                                          \
  }

#define ASVGF_FILTER_BOX3X3()        \
  {                                  \
    for (int i = -1; i < 2; i++) {   \
      for (int j = -1; j < 2; j++) { \
        if (i != 0 || j != 0) {      \
          ASVGF_TAP(i, j, 1.0f);     \
        }                            \
      }                              \
    }                                \
  }

#define ASVGF_FILTER_ATROUS()       \
  {                                 \
    ASVGF_TAP(1, 0, 2.0f / 3.0f);   \
    ASVGF_TAP(0, 1, 2.0f / 3.0f);   \
    ASVGF_TAP(-1, 0, 2.0f / 3.0f);  \
    ASVGF_TAP(0, -1, 2.0f / 3.0f);  \
                                    \
    ASVGF_TAP(2, 0, 1.0f / 6.0f);   \
    ASVGF_TAP(0, 2, 1.0f / 6.0f);   \
    ASVGF_TAP(-2, 0, 1.0f / 6.0f);  \
    ASVGF_TAP(0, -2, 1.0f / 6.0f);  \
                                    \
    ASVGF_TAP(1, 1, 4.0f / 9.0f);   \
    ASVGF_TAP(-1, 1, 4.0f / 9.0f);  \
    ASVGF_TAP(1, -1, 4.0f / 9.0f);  \
    ASVGF_TAP(-1, -1, 4.0f / 9.0f); \
                                    \
    ASVGF_TAP(1, 2, 1.0f / 9.0f);   \
    ASVGF_TAP(-1, 2, 1.0f / 9.0f);  \
    ASVGF_TAP(1, -2, 1.0f / 9.0f);  \
    ASVGF_TAP(-1, -2, 1.0f / 9.0f); \
                                    \
    ASVGF_TAP(2, 1, 1.0f / 9.0f);   \
    ASVGF_TAP(-2, 1, 1.0f / 9.0f);  \
    ASVGF_TAP(2, -1, 1.0f / 9.0f);  \
    ASVGF_TAP(-2, -1, 1.0f / 9.0f); \
                                    \
    ASVGF_TAP(2, 2, 1.0f / 36.0f);  \
    ASVGF_TAP(2, 2, 1.0f / 36.0f);  \
    ASVGF_TAP(2, 2, 1.0f / 36.0f);  \
    ASVGF_TAP(2, 2, 1.0f / 36.0f);  \
  }

__global__ void filter_output(int filter_scale) {
  for (int offset = threadIdx.x + blockIdx.x * blockDim.x; offset < device_amount; offset += blockDim.x * gridDim.x) {
    const int x = offset % device_width;
    const int y = offset / device_width;

    RGBF color        = device_frame_output[offset];
    vec3 normal       = device_frame_normal_buffer[offset];
    float variance    = device_frame_variance_buffer[offset];
    TraceResult trace = device_frame_trace_buffer[offset];
    const float slope = device_frame_slope_buffer[offset];
    float l           = luminance(color);
    float sigma_l     = sigma_luminance(variance, x, y) * 3.0f;

    RGBF sum_color     = color;
    float sum_variance = variance;
    float sum_weights  = 1.0f;

    if (trace.depth > 0.0f) {
      ASVGF_FILTER_ATROUS();
    }

    sum_color = scale_color(sum_color, 1.0f / sum_weights);
    sum_variance *= (1.0f / sum_weights) * (1.0f / sum_weights);

    // device_frame_variance_buffer[offset] = sum_variance;
    device_frame_output[offset] = sum_color;
  }
}

#endif /* CU_ASVGF_H */
