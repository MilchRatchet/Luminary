#ifndef CU_DIRECTIVES_H
#define CU_DIRECTIVES_H

/*
 * Define WEIGHT_BASED_EXIT for the following to apply
 *
 * Rays with an accumulated weight below CUTOFF are cancelled
 * Higher values provide better performance at the cost of extreme bright lights not being shaded properly
 *
 * Change BRIGHTEST_EMISSION to the intensity of the brightest light for best performance without
 * visual degradation
 *
 * Rays with an accumulated weight below PROBABILISTIC_CUTOFF have a chance equal to
 * the inverse linear interpolation of the weight and CUTOFF and PROBABILISTIC_CUTOFF
 * Higher values provide much better performance at the cost of noisy/too dark areas
 * Setting PROBABILISTIC_CUTOFF >= CUTOFF will deactivate this option
 */
#define WEIGHT_BASED_EXIT
#define BRIGHTEST_EMISSION 40.0f
#define CUTOFF ((1.0f) / (BRIGHTEST_EMISSION * 255.0f))
#define PROBABILISTIC_CUTOFF ((1.0f) / (0.5f * BRIGHTEST_EMISSION * 255.0f))

/*
 * Define LOW_QUALITY_LONG_BOUNCES for the following to apply
 *
 * After each bounce there is a 1/(max_depth) chance for the sample to exit early
 * This provides much better performance at the cost of noisy indirectly lit areas
 *
 * MIN_BOUNCES controls the minimum amount of bounces that have to happen before this may happen
 */
#define LOW_QUALITY_LONG_BOUNCES
#define MIN_BOUNCES 1

/*
 * Define SINGLE_CONTRIBUTIONS_ONLY for the following to apply
 *
 * A sample exits early when light information have already been gathered
 * This improves performance at the cost of reflections in directly lit surfaces
 */
#define SINGLE_CONTRIBUTION_ONLY

__device__ int validate_trace_task(TraceTask task, RGBF record) {
  int valid = 1;

#ifdef WEIGHT_BASED_EXIT
  const float max = fmaxf(record.r, fmaxf(record.g, record.b));
  if (
    max < CUTOFF
    || (max < PROBABILISTIC_CUTOFF && sample_blue_noise(task.index.x, task.index.y, task.state, 20) > (max - CUTOFF) / (CUTOFF - PROBABILISTIC_CUTOFF))) {
    valid = 0;
  }
#endif

#ifdef LOW_QUALITY_LONG_BOUNCES
  if (
    ((task.state & DEPTH_LEFT) >> 16) <= (device_max_ray_depth - MIN_BOUNCES)
    && sample_blue_noise(task.index.x, task.index.y, task.state, 21) < 1.0f / (1 + device_max_ray_depth)) {
    valid = 0;
  }
#endif

#ifdef SINGLE_CONTRIBUTION_ONLY
  {
    RGBF color = device_frame_buffer[task.index.x + task.index.y * device_width];
    if (luminance(color) > eps)
      valid = 0;
  }
#endif

  return valid;
}

#endif /* CU_DIRECTIVES_H */
