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

#define RIS_COLLECT_SAMPLE_COUNT

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

  reservoir.random = (reservoir.random - random_shift) / random_scale;

  return sample_accepted;
}

__device__ float ris_reservoir_get_sampling_weight(const RISReservoir reservoir) {
  return (reservoir.selected_target > 0.0f) ? reservoir.sum_weight / reservoir.selected_target : 0.0f;
}

template <uint32_t NUM_TARGETS>
struct MultiRISAggregator {
  float sum_weight[NUM_TARGETS];
  float resampling_probability[NUM_TARGETS];
};

template <uint32_t NUM_TARGETS>
__device__ void multi_ris_aggregator_reset(MultiRISAggregator<NUM_TARGETS>& aggregator) {
#pragma unroll
  for (uint32_t target_id = 0; target_id < NUM_TARGETS; target_id++) {
    aggregator.sum_weight[target_id]             = 0.0f;
    aggregator.resampling_probability[target_id] = 0.0f;
  }
}

template <uint32_t NUM_TARGETS>
__device__ MultiRISAggregator<NUM_TARGETS> multi_ris_aggregator_init(void) {
  MultiRISAggregator<NUM_TARGETS> aggregator;

  multi_ris_aggregator_reset<NUM_TARGETS>(aggregator);

  return aggregator;
}

template <uint32_t NUM_TARGETS>
__device__ void multi_ris_aggregator_add_sample(
  MultiRISAggregator<NUM_TARGETS>& aggregator, const float target[NUM_TARGETS], const float sampling_weight) {
#pragma unroll
  for (uint32_t target_id = 0; target_id < NUM_TARGETS; target_id++) {
    const float weight = target[target_id] * sampling_weight;

    aggregator.sum_weight[target_id] += weight;
    aggregator.resampling_probability[target_id] = (weight > 0.0f) ? weight / aggregator.sum_weight[target_id] : 0.0f;
  }
}

template <uint32_t NUM_TARGETS, uint32_t TARGET_ID>
struct MultiRISLane {
  float selected_target[NUM_TARGETS];
  float random;
};

template <uint32_t NUM_TARGETS, uint32_t TARGET_ID>
__device__ void multi_ris_lane_reset(MultiRISLane<NUM_TARGETS, TARGET_ID>& lane) {
#pragma unroll
  for (uint32_t target_id = 0; target_id < NUM_TARGETS; target_id++) {
    lane.selected_target[target_id] = 0.0f;
  }
}

template <uint32_t NUM_TARGETS, uint32_t TARGET_ID>
__device__ MultiRISLane<NUM_TARGETS, TARGET_ID> multi_ris_lane_init(const float random) {
  MultiRISLane<NUM_TARGETS, TARGET_ID> lane;

  lane.random = random;

  multi_ris_lane_reset<NUM_TARGETS, TARGET_ID>(lane);

  return lane;
}

template <uint32_t NUM_TARGETS, uint32_t TARGET_ID>
__device__ bool multi_ris_lane_add_sample(
  MultiRISLane<NUM_TARGETS, TARGET_ID>& lane, const MultiRISAggregator<NUM_TARGETS> aggregator, const float target[NUM_TARGETS]) {
  const float resampling_probability = aggregator.resampling_probability[TARGET_ID];
  const bool sample_accepted         = (lane.random < resampling_probability);

  if (sample_accepted) {
#pragma unroll
    for (uint32_t target_id = 0; target_id < NUM_TARGETS; target_id++) {
      lane.selected_target[target_id] = target[target_id];
    }
  }

  const float random_shift = (sample_accepted) ? 0.0f : resampling_probability;
  const float random_scale = (sample_accepted) ? resampling_probability : 1.0f - resampling_probability;

  lane.random = (lane.random - random_shift) / random_scale;

  return sample_accepted;
}

template <uint32_t NUM_TARGETS, uint32_t TARGET_ID>
__device__ float multi_ris_lane_get_sampling_weight(
  MultiRISLane<NUM_TARGETS, TARGET_ID>& lane, const MultiRISAggregator<NUM_TARGETS> aggregator, const uint32_t num_lanes[NUM_TARGETS]) {
  float probability_not_sampling = 1.0f;

#pragma unroll
  for (uint32_t target_id = 0; target_id < NUM_TARGETS; target_id++) {
    const float target             = lane.selected_target[target_id];
    const float sum_weight         = aggregator.sum_weight[target_id];
    const float target_probability = (sum_weight > 0.0f) ? target / sum_weight : 0.0f;
    const float anti_probability   = (1.0f - target_probability);

    probability_not_sampling *= powf(anti_probability, num_lanes[target_id]);
  }

  return 1.0f / (1.0f - probability_not_sampling);
}

#endif /* CU_RIS_H */
