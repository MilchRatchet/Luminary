#include "device_renderer.h"

#include "device.h"
#include "internal_error.h"

LuminaryResult device_renderer_create(DeviceRenderer** renderer) {
  __CHECK_NULL_ARGUMENT(renderer);

  __FAILURE_HANDLE(host_malloc(renderer, sizeof(DeviceRenderer)));
  memset(*renderer, 0, sizeof(DeviceRenderer));

  __FAILURE_HANDLE(array_create(&(*renderer)->queue, sizeof(DeviceRendererQueueAction), 16));

  __FAILURE_HANDLE(ringbuffer_create(&(*renderer)->callback_data_ringbuffer, sizeof(DeviceRenderCallbackData) * 64));

  for (uint32_t event_id = 0; event_id < DEVICE_RENDERER_TIMING_EVENTS_COUNT; event_id++) {
    __FAILURE_HANDLE(cuEventCreate(&(*renderer)->time_start[event_id], 0));
    __FAILURE_HANDLE(cuEventCreate(&(*renderer)->time_end[event_id], 0));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_handle_callback(DeviceRenderer* renderer, DeviceRenderCallbackData* data, bool* is_valid) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(data);
  __CHECK_NULL_ARGUMENT(is_valid);

  *is_valid = renderer->render_id == data->render_id;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_build_kernel_queue(DeviceRenderer* renderer, DeviceRendererQueueArgs* args) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(array_set_num_elements(&renderer->queue, 0));

  DeviceRendererQueueAction action;

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_CONST_MEM;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  for (uint32_t depth = 0; depth <= args->max_depth; depth++) {
    action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_DEPTH;
    action.mem_update = (DeviceRendererQueueActionMemUpdate){.depth = depth};
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    if (depth == 0) {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_GENERATE_TRACE_TASKS;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }
    else {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_BALANCE_TRACE_TASKS;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    // TODO: Figure out better ways of doing this.
    if (depth == 0) {
      action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_QUEUE_NEXT_SAMPLE;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
    action.optix_type = OPTIX_KERNEL_TYPE_RAYTRACE_GEOMETRY;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    if (args->render_particles) {
      action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
      action.optix_type = OPTIX_KERNEL_TYPE_RAYTRACE_PARTICLES;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    if (args->render_volumes) {
      // It is important to compute bridges lighting before sampling volume scattering events.
      // We want to sample over the whole unoccluded ray path so sampling scattering events before
      // that would shorten the path.

      action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
      action.optix_type = OPTIX_KERNEL_TYPE_SHADING_VOLUME_BRIDGES;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));

      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_VOLUME_PROCESS_EVENTS;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    if (args->render_clouds) {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_CLOUD_PROCESS_TASKS;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    if (args->render_inscattering) {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_SKY_PROCESS_INSCATTERING_EVENTS;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
    action.cuda_type = CUDA_KERNEL_TYPE_POSTPROCESS_TRACE_TASKS;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
    action.optix_type = OPTIX_KERNEL_TYPE_SHADING_GEOMETRY;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    if (args->render_volumes) {
      action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
      action.optix_type = OPTIX_KERNEL_TYPE_SHADING_VOLUME;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    if (args->render_particles) {
      action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
      action.optix_type = OPTIX_KERNEL_TYPE_SHADING_PARTICLES;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
    action.cuda_type = CUDA_KERNEL_TYPE_SKY_PROCESS_TASKS;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));
  }

  action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
  action.cuda_type = CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_END_OF_SAMPLE;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_register_callback(DeviceRenderer* renderer, CUhostFn callback_func, DeviceCommonCallbackData callback_data) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(callback_func);

  renderer->registered_callback_func = callback_func;

  renderer->callback_data = callback_data;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_clear_callback_data(DeviceRenderer* renderer) {
  __CHECK_NULL_ARGUMENT(renderer);

  __FAILURE_HANDLE(ringbuffer_release_entry(renderer->callback_data_ringbuffer, sizeof(DeviceRenderCallbackData)));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_init_new_render(DeviceRenderer* renderer) {
  __CHECK_NULL_ARGUMENT(renderer);

  renderer->render_id++;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_queue_sample(DeviceRenderer* renderer, Device* device, SampleCountSlice* sample_count) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);

  if (sample_count->current_sample_count == sample_count->end_sample_count)
    return LUMINARY_SUCCESS;

  uint32_t num_actions;
  __FAILURE_HANDLE(array_get_num_elements(renderer->queue, &num_actions));

  uint32_t event_id = renderer->event_id & DEVICE_RENDERER_TIMING_EVENTS_MASK;

  if (renderer->event_id >= DEVICE_RENDERER_TIMING_EVENTS_COUNT) {
    CUDA_FAILURE_HANDLE(cuEventElapsedTime(&renderer->last_time, renderer->time_start[event_id], renderer->time_end[event_id]));

    // Convert from milliseconds to seconds.
    renderer->last_time *= 0.001;
  }

  CUDA_FAILURE_HANDLE(cuEventRecord(renderer->time_start[event_id], device->stream_main));

  for (uint32_t action_id = renderer->action_ptr; action_id < num_actions; action_id++) {
    const DeviceRendererQueueAction* action = renderer->queue + action_id;

    switch (action->type) {
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL:
        __FAILURE_HANDLE(kernel_execute(device->cuda_kernels[action->cuda_type], device->stream_main));
        break;
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL:
        __FAILURE_HANDLE(optix_kernel_execute(device->optix_kernels[action->optix_type], device));
        break;
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_CONST_MEM:
        __FAILURE_HANDLE(device_update_dynamic_const_mem(device, sample_count->current_sample_count));
        break;
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_DEPTH:
        __FAILURE_HANDLE(device_update_depth_const_mem(device, action->mem_update.depth));
        break;
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_QUEUE_NEXT_SAMPLE:
        DeviceRenderCallbackData* callback_data;
        __FAILURE_HANDLE(
          ringbuffer_allocate_entry(renderer->callback_data_ringbuffer, sizeof(DeviceRenderCallbackData), (void**) &callback_data));

        callback_data->common    = renderer->callback_data;
        callback_data->render_id = renderer->render_id;

        CUDA_FAILURE_HANDLE(cuEventRecord(device->event_queue_render, device->stream_main));
        CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_callbacks, device->event_queue_render, CU_EVENT_WAIT_DEFAULT));
        CUDA_FAILURE_HANDLE(cuLaunchHostFunc(device->stream_callbacks, renderer->registered_callback_func, callback_data));
        break;
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_END_OF_SAMPLE:
        if ((device->undersampling_state & ~UNDERSAMPLING_FIRST_SAMPLE_MASK) == 0) {
          sample_count->current_sample_count++;
        }

        CUDA_FAILURE_HANDLE(cuEventRecord(renderer->time_end[event_id], device->stream_main));
        renderer->event_id++;
        return LUMINARY_SUCCESS;
    }
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_destroy(DeviceRenderer** renderer) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(*renderer);

  __FAILURE_HANDLE(array_destroy(&(*renderer)->queue));
  __FAILURE_HANDLE(ringbuffer_destroy(&(*renderer)->callback_data_ringbuffer));

  for (uint32_t event_id = 0; event_id < DEVICE_RENDERER_TIMING_EVENTS_COUNT; event_id++) {
    __FAILURE_HANDLE(cuEventDestroy((*renderer)->time_start[event_id]));
    __FAILURE_HANDLE(cuEventDestroy((*renderer)->time_end[event_id]));
  }

  __FAILURE_HANDLE(host_free(renderer));

  return LUMINARY_SUCCESS;
}
