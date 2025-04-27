#include "device_sampletime.h"

#include "internal_error.h"

LuminaryResult sample_time_create(SampleTime** sample_time) {
  __CHECK_NULL_ARGUMENT(sample_time);

  __FAILURE_HANDLE(host_malloc(sample_time, sizeof(SampleTime)));
  memset(*sample_time, 0, sizeof(SampleTime));

  return LUMINARY_SUCCESS;
}

LuminaryResult sample_time_reset(SampleTime* sample_time) {
  __CHECK_NULL_ARGUMENT(sample_time);

  memset(sample_time->time, 0, sizeof(sample_time->time));

  return LUMINARY_SUCCESS;
}

LuminaryResult sample_time_set_time(SampleTime* sample_time, uint32_t device_id, double time) {
  __CHECK_NULL_ARGUMENT(sample_time);

  __DEBUG_ASSERT(device_id < LUMINARY_MAX_NUM_DEVICES);

  sample_time->time[device_id] = time;

  return LUMINARY_SUCCESS;
}

LuminaryResult sample_time_get_time(SampleTime* sample_time, double* time) {
  __CHECK_NULL_ARGUMENT(sample_time);

  double samples_per_ms = 0.0;

  for (uint32_t device_id = 0; device_id < LUMINARY_MAX_NUM_DEVICES; device_id++) {
    samples_per_ms += (sample_time->time[device_id] > 0.0) ? 1.0 / sample_time->time[device_id] : 0.0;
  }

  *time = (samples_per_ms > 0.0) ? 1.0 / samples_per_ms : 0.0;

  return LUMINARY_SUCCESS;
}

LuminaryResult sample_time_destroy(SampleTime** sample_time) {
  __CHECK_NULL_ARGUMENT(sample_time);

  __FAILURE_HANDLE(host_free(sample_time));

  return LUMINARY_SUCCESS;
}
