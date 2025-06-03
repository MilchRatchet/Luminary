#ifndef LUMINARY_DEVICE_RENDERER_H
#define LUMINARY_DEVICE_RENDERER_H

#include "device_callback.h"
#include "device_utils.h"
#include "kernel.h"
#include "optix_kernel.h"
#include "spinlock.h"

// #define DEVICE_RENDERER_DO_PER_KERNEL_TIMING

struct Device typedef Device;

enum DeviceRendererQueueActionType {
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_CONST_MEM,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_TILE_ID,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_DEPTH,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_QUEUE_NEXT_SAMPLE,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_END_OF_SAMPLE
} typedef DeviceRendererQueueActionType;

struct DeviceRendererQueueActionMemUpdate {
  union {
    // DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_DEPTH
    struct {
      uint32_t depth;
    };
  };

} typedef DeviceRendererQueueActionMemUpdate;

struct DeviceRendererQueueAction {
  DeviceRendererQueueActionType type;
  union {
    CUDAKernelType cuda_type;
    OptixKernelType optix_type;
    DeviceRendererQueueActionMemUpdate mem_update;
  };
} typedef DeviceRendererQueueAction;

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING

#define DEVICE_RENDERER_MAX_TIMED_KERNELS 256

struct DeviceRendererKernelTimer {
  const char* name;
  CUevent time_start;
  CUevent time_end;
} typedef DeviceRendererKernelTimer;

struct DeviceRendererPerKernelTimings {
  ARRAY DeviceRendererKernelTimer* kernels;
  uint32_t num_kernel_launches;
} typedef DeviceRendererPerKernelTimings;

#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

#define DEVICE_RENDERER_TIMING_EVENTS_COUNT 128
#define DEVICE_RENDERER_TIMING_EVENTS_MASK (DEVICE_RENDERER_TIMING_EVENTS_COUNT - 1)

struct DeviceRenderer {
  uint32_t action_ptr;
  ARRAY DeviceRendererQueueAction* prepass_queue;
  ARRAY DeviceRendererQueueAction* queue;
  ARRAY DeviceRendererQueueAction* postpass_queue;
  CUhostFn registered_callback_continue_func;
  CUhostFn registered_callback_finished_func;
  DeviceCommonCallbackData common_callback_data;
  uint64_t render_id;
  DeviceRenderCallbackData callback_data[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
  uint32_t event_id;
  uint32_t timing_event_id;
  CUevent time_start[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
  CUevent time_end[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
  float total_render_time[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
  float last_time;

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  DeviceRendererPerKernelTimings kernel_times[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */
} typedef DeviceRenderer;

struct DeviceRendererQueueArgs {
  uint32_t max_depth;
  bool render_particles;
  bool render_volumes;
  bool render_clouds;
  bool render_ocean;
  bool render_inscattering;
  bool render_lights;
  bool render_procedural_sky;
  ShadingMode shading_mode;
} typedef DeviceRendererQueueArgs;

DEVICE_CTX_FUNC LuminaryResult device_renderer_create(DeviceRenderer** renderer);
LuminaryResult device_renderer_handle_callback(DeviceRenderer* renderer, DeviceRenderCallbackData* data, bool* is_valid);
DEVICE_CTX_FUNC LuminaryResult device_renderer_build_kernel_queue(DeviceRenderer* renderer, DeviceRendererQueueArgs* args);
DEVICE_CTX_FUNC LuminaryResult device_renderer_register_callback(
  DeviceRenderer* renderer, CUhostFn callback_continue_func, CUhostFn callback_finished_func, DeviceCommonCallbackData callback_data);
DEVICE_CTX_FUNC LuminaryResult device_renderer_init_new_render(DeviceRenderer* renderer);
DEVICE_CTX_FUNC LuminaryResult device_renderer_queue_sample(DeviceRenderer* renderer, Device* device, SampleCountSlice* sample_count);
DEVICE_CTX_FUNC LuminaryResult device_renderer_update_render_time(DeviceRenderer* renderer, uint32_t target_event_id);
LuminaryResult device_renderer_get_render_time(DeviceRenderer* renderer, uint32_t event_id, float* time);
LuminaryResult device_renderer_get_latest_event_id(DeviceRenderer* renderer, uint32_t* event_id);
DEVICE_CTX_FUNC LuminaryResult device_renderer_destroy(DeviceRenderer** renderer);

#endif /* LUMINARY_DEVICE_RENDERER_H */
