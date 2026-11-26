#ifndef LUMINARY_DEVICE_RENDERER_H
#define LUMINARY_DEVICE_RENDERER_H

#include "device_callback.h"
#include "device_utils.h"
#include "kernel.h"
#include "optix_kernel.h"
#include "thread.h"

// #define DEVICE_RENDERER_DO_PER_KERNEL_TIMING

struct Device typedef Device;
struct AdaptiveSampler typedef AdaptiveSampler;

struct DeviceRendererQueueArgs {
  uint32_t max_depth;
  bool enable_adaptive_sampling;
  bool render_particles;
  bool render_volumes;
  bool render_clouds;
  bool render_ocean;
  bool render_inscattering;
  bool render_lights;
  bool render_procedural_sky;
  uint32_t adaptive_sampling_update_interval;
  ShadingMode shading_mode;
} typedef DeviceRendererQueueArgs;

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING

#define DEVICE_RENDERER_TIMED_KERNEL_TYPES (CUDA_KERNEL_TYPE_COUNT + OPTIX_KERNEL_TYPE_COUNT)
#define DEVICE_RENDERER_KERNEL_TIMING_SLOTS 2

struct DeviceRendererKernelTimingSlot {
  CUevent start;
  CUevent end;
  bool pending;
} typedef DeviceRendererKernelTimingSlot;

struct DeviceRendererKernelTimer {
  const char* name;
  DeviceRendererKernelTimingSlot slots[DEVICE_RENDERER_KERNEL_TIMING_SLOTS];
  double total_ms;
} typedef DeviceRendererKernelTimer;

struct DeviceRendererPerKernelTimings {
  DeviceRendererKernelTimer kernels[DEVICE_RENDERER_TIMED_KERNEL_TYPES];
} typedef DeviceRendererPerKernelTimings;

#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

#define DEVICE_RENDERER_TIMING_EVENTS_COUNT 128
#define DEVICE_RENDERER_TIMING_EVENTS_MASK (DEVICE_RENDERER_TIMING_EVENTS_COUNT - 1)

typedef uint32_t DeviceRendererStatusFlags;

enum DeviceRendererStatusFlag {
  DEVICE_RENDERER_STATUS_FLAG_NONE         = 0,
  DEVICE_RENDERER_STATUS_FLAG_READY        = (1 << 0),
  DEVICE_RENDERER_STATUS_FLAG_IN_PROGRESS  = (1 << 1),
  DEVICE_RENDERER_STATUS_FLAG_FINISHED     = (1 << 2),
  DEVICE_RENDERER_STATUS_FLAG_FIRST_SAMPLE = (1 << 3)
} typedef DeviceRendererStatusFlag;

struct DeviceRenderer {
  Device* device;
  Thread* thread;
  Queue* commands;
  Queue* replies;
  uint64_t host_id;
  CUstream stream;
  CUevent manager_ready;
  CUevent renderer_ready;
  CUevent abort_flag_uploaded;
  CUevent abort_complete;
  DeviceSampleAllocation sample_allocation;
  uint32_t executed_aggregate_sample_counts[ADAPTIVE_SAMPLER_NUM_STAGES + 1];
  DeviceRendererStatusFlags status_flags;
  DeviceRendererQueueArgs args;
  uint32_t pass_id;
  uint32_t* work_pending;
  CUhostFn registered_callback_continue_func;
  CUhostFn registered_callback_finished_func;
  DeviceCommonCallbackData common_callback_data;
  uint64_t render_id;
  DeviceRenderCallbackData callback_data[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
  uint32_t callback_id;
  uint32_t event_id;
  uint32_t timing_event_id;
  CUevent time_start[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
  CUevent time_end[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
  float total_render_time[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
  float last_time;
  bool shutdown;

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  DeviceRendererPerKernelTimings kernel_times[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */
} typedef DeviceRenderer;

DEVICE_CTX_FUNC LuminaryResult device_renderer_create(DeviceRenderer** renderer, Device* device, uint64_t host_id);
LuminaryResult device_renderer_handle_callback(DeviceRenderer* renderer, DeviceRenderCallbackData* data, bool* is_valid);
DEVICE_CTX_FUNC LuminaryResult device_renderer_register_callback(
  DeviceRenderer* renderer, CUhostFn callback_continue_func, CUhostFn callback_finished_func, DeviceCommonCallbackData callback_data);
LuminaryResult device_renderer_start(DeviceRenderer* renderer, const DeviceRendererQueueArgs* args);
LuminaryResult device_renderer_continue(DeviceRenderer* renderer);
LuminaryResult device_renderer_request_abort(DeviceRenderer* renderer);
LuminaryResult device_renderer_wait_for_abort_submission(DeviceRenderer* renderer);
LuminaryResult device_renderer_finish_iteration(DeviceRenderer* renderer, bool is_undersampling);
DEVICE_CTX_FUNC LuminaryResult device_renderer_update_render_time(DeviceRenderer* renderer, uint32_t target_event_id);
LuminaryResult device_renderer_allocate_sample(DeviceRenderer* renderer, AdaptiveSampler* sampler);
LuminaryResult device_renderer_externalize_samples(
  DeviceRenderer* renderer, uint32_t num_stage_executions[ADAPTIVE_SAMPLER_NUM_STAGES + 1]);
LuminaryResult device_renderer_register_external_samples(
  DeviceRenderer* renderer, uint32_t num_stage_executions[ADAPTIVE_SAMPLER_NUM_STAGES + 1]);
LuminaryResult device_renderer_get_render_time(DeviceRenderer* renderer, uint32_t event_id, float* time);
LuminaryResult device_renderer_get_latest_event_id(DeviceRenderer* renderer, uint32_t* event_id);
LuminaryResult device_renderer_get_status(DeviceRenderer* renderer, DeviceRendererStatusFlags* status_flags);
LuminaryResult device_renderer_get_tile_count(DeviceRenderer* renderer, Device* device, uint32_t undersampling_stage, uint32_t* tile_count);
LuminaryResult device_renderer_get_total_executed_samples(DeviceRenderer* renderer, uint32_t* aggregate_sample_count);
LuminaryResult device_renderer_shutdown(DeviceRenderer* renderer);
DEVICE_CTX_FUNC LuminaryResult device_renderer_destroy(DeviceRenderer** renderer);

#endif /* LUMINARY_DEVICE_RENDERER_H */
