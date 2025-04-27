#ifndef LUMINARY_DEVICE_SAMPLETIME_H
#define LUMINARY_DEVICE_SAMPLETIME_H

#include "device_utils.h"

struct SampleTime {
  double time[LUMINARY_MAX_NUM_DEVICES];
} typedef SampleTime;

LuminaryResult sample_time_create(SampleTime** sample_time);
LuminaryResult sample_time_reset(SampleTime* sample_time);
LuminaryResult sample_time_set_time(SampleTime* sample_time, uint32_t device_id, double time);
LuminaryResult sample_time_get_time(SampleTime* sample_time, double* time);
LuminaryResult sample_time_destroy(SampleTime** sample_time);

#endif /* LUMINARY_DEVICE_SAMPLETIME_H */
