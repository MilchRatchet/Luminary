#include "sample_count.h"

#include "internal_error.h"

LuminaryResult sample_count_get_default(SampleCountSlice* slice) {
  __CHECK_NULL_ARGUMENT(slice);

  slice->current_sample_count = 0;
  slice->end_sample_count     = 0xFFFFFFFFu;

  return LUMINARY_SUCCESS;
}

LuminaryResult sample_count_get_slice(SampleCountSlice* src, uint32_t slice_size, SampleCountSlice* dst) {
  __CHECK_NULL_ARGUMENT(src);
  __CHECK_NULL_ARGUMENT(dst);

  dst->current_sample_count = src->current_sample_count;

  // The src slice might not be large enough so we need to cap the slice size.
  const uint32_t max_sample_count = dst->current_sample_count + slice_size;
  dst->end_sample_count           = (src->end_sample_count < max_sample_count) ? src->end_sample_count : max_sample_count;

  // The slice size might have been capped, so we must compute it here.
  const uint32_t actual_slice_size = dst->end_sample_count - src->current_sample_count;
  src->current_sample_count += actual_slice_size;

  return LUMINARY_SUCCESS;
}
