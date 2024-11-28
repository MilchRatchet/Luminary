#ifndef LUMINARY_DEVICE_RENDERER_H
#define LUMINARY_DEVICE_RENDERER_H

#include "device_utils.h"
#include "kernel.h"
#include "optix_kernel.h"

struct Device typedef Device;

enum DeviceRendererQueueActionType {
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL       = 0,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL      = 1,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_CONST_MEM  = 2,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_QUEUE_NEXT_SAMPLE = 3,
  DEVICE_RENDERER_QUEUE_ACTION_TYPE_END_OF_SAMPLE     = 4
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

struct DeviceRenderer {
  SampleCountSlice sample_count;
  uint32_t action_ptr;
  ARRAY DeviceRendererQueueAction* queue;
  CUhostFn registered_callback_func;
  void* registered_callback_data;
} typedef DeviceRenderer;

struct DeviceRendererQueueArgs {
  uint32_t max_depth;
  bool render_particles;
  bool render_volumes;
  bool render_clouds;
  bool render_inscattering;
} typedef DeviceRendererQueueArgs;

DEVICE_CTX_FUNC LuminaryResult device_renderer_create(DeviceRenderer** renderer);
DEVICE_CTX_FUNC LuminaryResult device_renderer_build_kernel_queue(DeviceRenderer* renderer, DeviceRendererQueueArgs* args);
DEVICE_CTX_FUNC LuminaryResult device_renderer_register_callback(DeviceRenderer* renderer, CUhostFn callback_func, void* callback_data);
DEVICE_CTX_FUNC LuminaryResult device_renderer_set_sample_slice(DeviceRenderer* renderer, SampleCountSlice sample_count);
DEVICE_CTX_FUNC LuminaryResult device_renderer_queue_sample(DeviceRenderer* renderer, Device* device);
DEVICE_CTX_FUNC LuminaryResult device_renderer_destroy(DeviceRenderer** renderer);

#endif /* LUMINARY_DEVICE_RENDERER_H */
