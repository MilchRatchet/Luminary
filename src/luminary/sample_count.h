#ifndef LUMINARY_SAMPLE_COUNT_H
#define LUMINARY_SAMPLE_COUNT_H

#include "utils.h"

struct SampleCountSlice {
  uint32_t current_sample_count;
  uint32_t end_sample_count;
} typedef SampleCountSlice;

LuminaryResult sample_count_get_default(SampleCountSlice* slice);
LuminaryResult sample_count_get_slice(SampleCountSlice* src, uint32_t slice_size, SampleCountSlice* dst);

#endif /* LUMINARY_SAMPLE_COUNT_H */
