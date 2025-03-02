#ifndef LUMINARY_DEVICE_RENDERER_H
#define LUMINARY_DEVICE_RENDERER_H

#include "device_callback.h"
#include "device_utils.h"
#include "kernel.h"
#include "optix_kernel.h"
#include "spinlock.h"

struct Device typedef Device;

enum DeviceRendererQueueActionType {
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_CONST_MEM,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_DEPTH,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_QUEUE_NEXT_SAMPLE,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_END_OF_SAMPLE
} typedef DeviceRendererQueueActionType;

struct DeviceRendererQueueActionMemUpdate {
  uint32_t depth;
} typedef DeviceRendererQueueActionMemUpdate;

struct DeviceRendererQueueAction {
  DeviceRendererQueueActionType type;
  union {
    CUDAKernelType cuda_type;
    OptixKernelType optix_type;
    DeviceRendererQueueActionMemUpdate mem_update;
  };
} typedef DeviceRendererQueueAction;

#define DEVICE_RENDERER_TIMING_EVENTS_COUNT 2
#define DEVICE_RENDERER_TIMING_EVENTS_MASK (DEVICE_RENDERER_TIMING_EVENTS_COUNT - 1)

struct DeviceRenderer {
  uint32_t action_ptr;
  ARRAY DeviceRendererQueueAction* queue;
  CUhostFn registered_callback_func;
  DeviceCommonCallbackData callback_data;
  uint64_t render_id;
  RingBuffer* callback_data_ringbuffer;
  uint32_t event_id;
  CUevent time_start[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
  CUevent time_end[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
  float total_render_time[DEVICE_RENDERER_TIMING_EVENTS_COUNT];
  float last_time;
  SpinLockCounter callbacks_active_counter;
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
DEVICE_CTX_FUNC LuminaryResult
  device_renderer_register_callback(DeviceRenderer* renderer, CUhostFn callback_func, DeviceCommonCallbackData callback_data);
LuminaryResult device_renderer_clear_callback_data(DeviceRenderer* renderer);
DEVICE_CTX_FUNC LuminaryResult device_renderer_init_new_render(DeviceRenderer* renderer);
DEVICE_CTX_FUNC LuminaryResult device_renderer_queue_sample(DeviceRenderer* renderer, Device* device, SampleCountSlice* sample_count);
LuminaryResult device_renderer_get_render_time(DeviceRenderer* renderer, uint32_t event_id, float* time);
DEVICE_CTX_FUNC LuminaryResult device_renderer_destroy(DeviceRenderer** renderer);

#endif /* LUMINARY_DEVICE_RENDERER_H */
