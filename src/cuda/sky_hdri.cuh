#ifndef CU_SKY_HDRI_H
#define CU_SKY_HDRI_H

#include "utils.cuh"
#include "utils.h"

/*
 * For documentation, see the host version in raytrace.c which is used for the camera rays.
 */
LUM_DEVICE_FUNC float sky_hdri_tent_filter_importance_sample(const float x) {
  if (x > 0.5f) {
    return 1.0f - sqrtf(2.0f) * sqrtf(1.0f - x);
  }
  else {
    return -1.0f + sqrtf(2.0f) * sqrtf(x);
  }
}

#endif /* CU_SKY_HDRI_H */
