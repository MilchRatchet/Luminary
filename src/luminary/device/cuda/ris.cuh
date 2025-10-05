#ifndef CU_RIS_H
#define CU_RIS_H

#include "math.cuh"
#include "random.cuh"
#include "utils.cuh"

////////////////////////////////////////////////////////////////////
// Literature
////////////////////////////////////////////////////////////////////

// [Cik22]
// E. Ciklabakkal, A. Gruson, I. Georgiev, D. Nowrouzezahrai, T. Hachisuka, "Single-pass stratified importance resampling",
// Computer Graphics Forum (Proceedings of EGSR), 2022

// For light tree based technique, we stratify the RIS light tree random number dimensions using 1D latin hypercube sampling
// to achieve the desired ordered stratification. The other dimensions are entirely dependent on this first dimension so it makes no
// sense to apply any further stratification for those within RIS.
__device__ float ris_transform_stratum(const uint32_t index, const uint32_t num_samples, const float random) {
  const float section_length = 1.0f / num_samples;

  return (index + random) * section_length;
}

__device__ float2 ris_transform_stratum_2D(const uint32_t index, const uint32_t num_samples, const float2 random) {
  return make_float2(ris_transform_stratum(index, num_samples, random.x), random.y);
}

// #define RIS_COLLECT_SAMPLE_COUNT

struct RISReservoir {
  float sum_weight;
  float selected_target;
  float random;
#ifdef RIS_COLLECT_SAMPLE_COUNT
  uint32_t num_samples;
#endif  // RIS_COLLECT_SAMPLE_COUNT
} typedef RISReservoir;

__device__ void ris_reservoir_reset(RISReservoir& reservoir) {
  reservoir.sum_weight      = 0.0f;
  reservoir.selected_target = 0.0f;

#ifdef RIS_COLLECT_SAMPLE_COUNT
  reservoir.num_samples = 0;
#endif  // RIS_COLLECT_SAMPLE_COUNT
}

__device__ RISReservoir ris_reservoir_init(const float random) {
  RISReservoir reservoir;

  reservoir.random = random;

  ris_reservoir_reset(reservoir);

  return reservoir;
}

__device__ bool ris_reservoir_add_sample(RISReservoir& reservoir, const float target, const float sampling_weight) {
  const float weight = target * sampling_weight;

  reservoir.sum_weight += weight;

#ifdef RIS_COLLECT_SAMPLE_COUNT
  reservoir.num_samples++;
#endif  // RIS_COLLECT_SAMPLE_COUNT

  if (weight == 0.0f)
    return false;

  const float resampling_probability = weight / reservoir.sum_weight;

  const bool sample_accepted = (reservoir.random < resampling_probability);

  reservoir.selected_target = (sample_accepted) ? target : reservoir.selected_target;
  const float random_shift  = (sample_accepted) ? 0.0f : resampling_probability;
  const float random_scale  = (sample_accepted) ? resampling_probability : 1.0f - resampling_probability;

  reservoir.random = random_saturate((reservoir.random - random_shift) / random_scale);

  return sample_accepted;
}

__device__ float ris_reservoir_get_sampling_weight(const RISReservoir reservoir) {
  return (reservoir.selected_target > 0.0f) ? reservoir.sum_weight / reservoir.selected_target : 0.0f;
}

__device__ float ris_reservoir_get_sampling_prob(const RISReservoir reservoir) {
  return (reservoir.sum_weight > 0.0f) ? reservoir.selected_target / reservoir.sum_weight : 1.0f;
}

struct RISAggregator {
  float sum_weight;
} typedef RISAggregator;

struct RISSampleHandle {
  float resampling_probability;
  float target;
} typedef RISSampleHandle;

struct RISLane {
  float selected_target;
  float random;
} typedef RISLane;

__device__ void ris_aggregator_reset(RISAggregator& aggregator) {
  aggregator.sum_weight = 0.0f;
}

__device__ RISAggregator ris_aggregator_init(void) {
  RISAggregator aggregator;

  ris_aggregator_reset(aggregator);

  return aggregator;
}

__device__ RISSampleHandle ris_aggregator_add_sample(RISAggregator& aggregator, const float target, const float sampling_weight) {
  const float weight = target * sampling_weight;

  aggregator.sum_weight += weight;

  RISSampleHandle handle;
  handle.resampling_probability = (weight > 0.0f) ? weight / aggregator.sum_weight : 0.0f;
  handle.target                 = target;

  return handle;
}

__device__ void ris_lane_reset(RISLane& lane) {
  lane.selected_target = 0.0f;
}

__device__ RISLane ris_lane_init(const float random) {
  RISLane lane;

  lane.random = random;

  ris_lane_reset(lane);

  return lane;
}

__device__ bool ris_lane_add_sample(RISLane& lane, const RISSampleHandle sample) {
  const float resampling_probability = sample.resampling_probability;
  const bool sample_accepted         = (lane.random < resampling_probability);

  lane.selected_target     = (sample_accepted) ? sample.target : lane.selected_target;
  const float random_shift = (sample_accepted) ? 0.0f : resampling_probability;
  const float random_scale = (sample_accepted) ? resampling_probability : 1.0f - resampling_probability;

  lane.random = random_saturate((lane.random - random_shift) / random_scale);

  return sample_accepted;
}

__device__ float ris_lane_get_sampling_weight(const RISLane lane, const RISAggregator aggregator) {
  return (lane.selected_target > 0.0f) ? aggregator.sum_weight / lane.selected_target : 0.0f;
}

__device__ float ris_lane_get_sampling_prob(const RISLane lane, const RISAggregator aggregator) {
  return (aggregator.sum_weight > 0.0f) ? (lane.selected_target / aggregator.sum_weight) : 0.0f;
}

#endif /* CU_RIS_H */
