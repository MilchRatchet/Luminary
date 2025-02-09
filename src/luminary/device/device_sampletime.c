#include "device_sampletime.h"

#include "internal_error.h"

LuminaryResult sample_time_create(SampleTime** sample_time) {
  __CHECK_NULL_ARGUMENT(sample_time);

  __FAILURE_HANDLE(host_malloc(sample_time, sizeof(SampleTime)));
  memset(*sample_time, 0, sizeof(SampleTime));

  return LUMINARY_SUCCESS;
}

LuminaryResult sample_time_set_time(SampleTime* sample_time, uint32_t device_id, double time) {
  __CHECK_NULL_ARGUMENT(sample_time);

  LUM_UNUSED(device_id);

  // TODO: Implement proper time keeping based on devices
  sample_time->time = time;

  return LUMINARY_SUCCESS;
}

LuminaryResult sample_time_get_time(SampleTime* sample_time, double* time) {
  __CHECK_NULL_ARGUMENT(sample_time);

  *time = sample_time->time;

  return LUMINARY_SUCCESS;
}

LuminaryResult sample_time_destroy(SampleTime** sample_time) {
  __CHECK_NULL_ARGUMENT(sample_time);

  __FAILURE_HANDLE(host_free(sample_time));

  return LUMINARY_SUCCESS;
}
