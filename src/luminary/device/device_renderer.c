#include "device_renderer.h"

#include "device.h"
#include "internal_error.h"

LuminaryResult device_renderer_create(DeviceRenderer** renderer) {
  __CHECK_NULL_ARGUMENT(renderer);

  __FAILURE_HANDLE(host_malloc(renderer, sizeof(DeviceRenderer)));
  memset(*renderer, 0, sizeof(DeviceRenderer));

  __FAILURE_HANDLE(array_create(&(*renderer)->queue, sizeof(DeviceRendererQueueAction), 16));

  for (uint32_t event_id = 0; event_id < DEVICE_RENDERER_TIMING_EVENTS_COUNT; event_id++) {
    __FAILURE_HANDLE(cuEventCreate(&(*renderer)->time_start[event_id], 0));
    __FAILURE_HANDLE(cuEventCreate(&(*renderer)->time_end[event_id], 0));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
    __FAILURE_HANDLE(
      array_create(&(*renderer)->kernel_times[event_id].kernels, sizeof(DeviceRendererKernelTimer), DEVICE_RENDERER_MAX_TIMED_KERNELS));

    for (uint32_t timer_id = 0; timer_id < DEVICE_RENDERER_MAX_TIMED_KERNELS; timer_id++) {
      __FAILURE_HANDLE(cuEventCreate(&(*renderer)->kernel_times[event_id].kernels[timer_id].time_start, 0));
      __FAILURE_HANDLE(cuEventCreate(&(*renderer)->kernel_times[event_id].kernels[timer_id].time_end, 0));
    }
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */
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

static LuminaryResult _device_renderer_build_main_kernel_queue(DeviceRenderer* renderer, DeviceRendererQueueArgs* args) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(args);

  DeviceRendererQueueAction action;

  for (uint32_t depth = 0; depth <= args->max_depth; depth++) {
    action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_DEPTH;
    action.mem_update = (DeviceRendererQueueActionMemUpdate) {.depth = depth};
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    // TODO: Figure out better ways of doing this.
    if (depth == 0) {
      action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_QUEUE_NEXT_SAMPLE;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

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

    action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
    action.optix_type = OPTIX_KERNEL_TYPE_RAYTRACE;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    if (args->render_volumes) {
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
    action.optix_type = OPTIX_KERNEL_TYPE_SHADING_GEOMETRY_GEO;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
    action.optix_type = OPTIX_KERNEL_TYPE_SHADING_GEOMETRY_SKY;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
    action.cuda_type = CUDA_KERNEL_TYPE_GEOMETRY_PROCESS_TASKS;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    if (args->render_ocean && depth != args->max_depth) {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_OCEAN_PROCESS_TASKS;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

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

    if (args->render_procedural_sky) {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_SKY_PROCESS_TASKS;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_build_debug_kernel_queue(DeviceRenderer* renderer, DeviceRendererQueueArgs* args) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(args);

  DeviceRendererQueueAction action;

  action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_DEPTH;
  action.mem_update = (DeviceRendererQueueActionMemUpdate) {.depth = 0};
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
  action.cuda_type = CUDA_KERNEL_TYPE_GENERATE_TRACE_TASKS;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_QUEUE_NEXT_SAMPLE;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
  action.optix_type = OPTIX_KERNEL_TYPE_RAYTRACE;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  if (args->render_volumes) {
    action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
    action.cuda_type = CUDA_KERNEL_TYPE_VOLUME_PROCESS_EVENTS;
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

  action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
  action.cuda_type = CUDA_KERNEL_TYPE_GEOMETRY_PROCESS_TASKS_DEBUG;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  if (args->render_ocean) {
    action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
    action.cuda_type = CUDA_KERNEL_TYPE_OCEAN_PROCESS_TASKS_DEBUG;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));
  }

  if (args->render_particles) {
    action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
    action.cuda_type = CUDA_KERNEL_TYPE_PARTICLE_PROCESS_TASKS_DEBUG;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));
  }

  action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
  action.cuda_type = CUDA_KERNEL_TYPE_SKY_PROCESS_TASKS_DEBUG;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_build_kernel_queue(DeviceRenderer* renderer, DeviceRendererQueueArgs* args) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(args);

  __FAILURE_HANDLE(array_set_num_elements(&renderer->queue, 0));

  DeviceRendererQueueAction action;

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_CONST_MEM;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  switch (args->shading_mode) {
    case LUMINARY_SHADING_MODE_DEFAULT:
    default:
      __FAILURE_HANDLE(_device_renderer_build_main_kernel_queue(renderer, args));
      break;
    case LUMINARY_SHADING_MODE_ALBEDO:
    case LUMINARY_SHADING_MODE_DEPTH:
    case LUMINARY_SHADING_MODE_NORMAL:
    case LUMINARY_SHADING_MODE_IDENTIFICATION:
    case LUMINARY_SHADING_MODE_LIGHTS:
      __FAILURE_HANDLE(_device_renderer_build_debug_kernel_queue(renderer, args));
      break;
  }

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_END_OF_SAMPLE;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_register_callback(
  DeviceRenderer* renderer, CUhostFn callback_continue_func, CUhostFn callback_finished_func, DeviceCommonCallbackData callback_data) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(callback_continue_func);
  __CHECK_NULL_ARGUMENT(callback_finished_func);

  renderer->registered_callback_continue_func = callback_continue_func;
  renderer->registered_callback_finished_func = callback_finished_func;

  renderer->common_callback_data = callback_data;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_init_new_render(DeviceRenderer* renderer) {
  __CHECK_NULL_ARGUMENT(renderer);

  renderer->render_id++;

  for (uint32_t event_id = 0; event_id < DEVICE_RENDERER_TIMING_EVENTS_COUNT; event_id++) {
    renderer->total_render_time[event_id] = 0.0f;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_queue_cuda_kernel(
  DeviceRenderer* renderer, Device* device, CUDAKernelType type, uint32_t* launch_id) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(launch_id);

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  uint32_t event_id = renderer->event_id & DEVICE_RENDERER_TIMING_EVENTS_MASK;

  if (*launch_id < DEVICE_RENDERER_MAX_TIMED_KERNELS) {
    DeviceRendererKernelTimer* timer = renderer->kernel_times[event_id].kernels + *launch_id;

    __FAILURE_HANDLE(kernel_get_name(device->cuda_kernels[type], &timer->name));

    CUDA_FAILURE_HANDLE(cuEventRecord(timer->time_start, device->stream_main));
  }
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  __FAILURE_HANDLE(kernel_execute(device->cuda_kernels[type], device->stream_main));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  if (*launch_id < DEVICE_RENDERER_MAX_TIMED_KERNELS) {
    DeviceRendererKernelTimer* timer = renderer->kernel_times[event_id].kernels + *launch_id;

    CUDA_FAILURE_HANDLE(cuEventRecord(timer->time_end, device->stream_main));
  }
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  *launch_id = *launch_id + 1;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_queue_optix_kernel(
  DeviceRenderer* renderer, Device* device, OptixKernelType type, uint32_t* launch_id) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(launch_id);

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  uint32_t event_id = renderer->event_id & DEVICE_RENDERER_TIMING_EVENTS_MASK;

  if (*launch_id < DEVICE_RENDERER_MAX_TIMED_KERNELS) {
    DeviceRendererKernelTimer* timer = renderer->kernel_times[event_id].kernels + *launch_id;

    timer->name = "Optix Kernel";

    CUDA_FAILURE_HANDLE(cuEventRecord(timer->time_start, device->stream_main));
  }
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  __FAILURE_HANDLE(optix_kernel_execute(device->optix_kernels[type], device));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  if (*launch_id < DEVICE_RENDERER_MAX_TIMED_KERNELS) {
    DeviceRendererKernelTimer* timer = renderer->kernel_times[event_id].kernels + *launch_id;

    CUDA_FAILURE_HANDLE(cuEventRecord(timer->time_end, device->stream_main));
  }
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  *launch_id = *launch_id + 1;

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

  // Same callback data is used for continue and finished callbacks
  DeviceRenderCallbackData* shared_callback_data = renderer->callback_data + event_id;

  shared_callback_data->common          = renderer->common_callback_data;
  shared_callback_data->render_id       = renderer->render_id;
  shared_callback_data->render_event_id = renderer->event_id;

  CUDA_FAILURE_HANDLE(cuEventRecord(renderer->time_start[event_id], device->stream_main));

  // Query only during the first sample
  bool is_gbuffer_meta_query = (sample_count->current_sample_count == 0);

  if (is_gbuffer_meta_query) {
    const uint32_t undersampling_stage = (device->undersampling_state & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;

    // Query only if enough samples have been computed after this iteration.
    if (undersampling_stage > (1 + device->constant_memory->settings.supersampling)) {
      is_gbuffer_meta_query = false;
    }
  }

  uint32_t launch_id = 0;

  for (uint32_t action_id = renderer->action_ptr; action_id < num_actions; action_id++) {
    const DeviceRendererQueueAction* action = renderer->queue + action_id;

    switch (action->type) {
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL:
        __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, action->cuda_type, &launch_id));
        break;
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL:
        __FAILURE_HANDLE(_device_renderer_queue_optix_kernel(renderer, device, action->optix_type, &launch_id));
        break;
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_CONST_MEM:
        __FAILURE_HANDLE(device_update_dynamic_const_mem(device, sample_count->current_sample_count, 0xFFFF, 0xFFFF));
        break;
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_DEPTH:
        __FAILURE_HANDLE(device_update_depth_const_mem(device, action->mem_update.depth));

        if (is_gbuffer_meta_query && (action->mem_update.depth > 0)) {
          __FAILURE_HANDLE(device_query_gbuffer_meta(device));
          is_gbuffer_meta_query = false;
        }
        break;
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_QUEUE_NEXT_SAMPLE:
        CUDA_FAILURE_HANDLE(cuEventRecord(device->event_queue_render, device->stream_main));
        CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_callbacks, device->event_queue_render, CU_EVENT_WAIT_DEFAULT));
        CUDA_FAILURE_HANDLE(cuLaunchHostFunc(device->stream_callbacks, renderer->registered_callback_continue_func, shared_callback_data));
        break;
      case DEVICE_RENDERER_QUEUE_ACTION_TYPE_END_OF_SAMPLE:
        if (device->constant_memory->settings.shading_mode == LUMINARY_SHADING_MODE_DEFAULT) {
          if (device->sample_count.current_sample_count == 0 || (device->undersampling_state & UNDERSAMPLING_FIRST_SAMPLE_MASK)) {
            __FAILURE_HANDLE(
              _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_FIRST_SAMPLE, &launch_id));
          }
          else {
            __FAILURE_HANDLE(
              _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_UPDATE, &launch_id));

            // TODO: This is only needed when outputting
            if (device->constant_memory->camera.do_firefly_rejection) {
              __FAILURE_HANDLE(
                _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_OUTPUT, &launch_id));
            }
            else {
              __FAILURE_HANDLE(
                _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_OUTPUT_RAW, &launch_id));
            }
          }
        }
        else {
          __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_AOV, &launch_id));
        }

        if ((device->undersampling_state & ~UNDERSAMPLING_FIRST_SAMPLE_MASK) == 0) {
          sample_count->current_sample_count++;
        }

        CUDA_FAILURE_HANDLE(cuEventRecord(renderer->time_end[event_id], device->stream_main));
        CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_callbacks, renderer->time_end[event_id], CU_EVENT_WAIT_DEFAULT));
        CUDA_FAILURE_HANDLE(cuLaunchHostFunc(device->stream_callbacks, renderer->registered_callback_finished_func, shared_callback_data));

        renderer->event_id++;

        if (is_gbuffer_meta_query) {
          __FAILURE_HANDLE(device_query_gbuffer_meta(device));
        }

        break;
    }
  }

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  renderer->kernel_times[event_id].num_kernel_launches = min(launch_id, DEVICE_RENDERER_MAX_TIMED_KERNELS);
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_update_render_time(DeviceRenderer* renderer, uint32_t event_id) {
  __CHECK_NULL_ARGUMENT(renderer);

  if (event_id >= renderer->event_id) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Renderer tried to update time for event %u when the next event is %u", event_id, renderer->event_id);
  }

  event_id = event_id & DEVICE_RENDERER_TIMING_EVENTS_MASK;

  float event_time;
  CUDA_FAILURE_HANDLE(cuEventElapsedTime(&event_time, renderer->time_start[event_id], renderer->time_end[event_id]));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING

  const uint32_t num_kernel_launches = renderer->kernel_times[event_id].num_kernel_launches;

  info_message("[EventIdx: %05u] ------- Kernel Times -------", event_id);

  float total_kernel_time = 0.0f;

  for (uint32_t launch_id = 0; launch_id < num_kernel_launches; launch_id++) {
    float kernel_time;
    CUDA_FAILURE_HANDLE(cuEventElapsedTime(
      &kernel_time, renderer->kernel_times[event_id].kernels[launch_id].time_start,
      renderer->kernel_times[event_id].kernels[launch_id].time_end));

    info_message(
      "[LaunchIdx: %03u] %32s | %07.2fms (%05.2f%%)", launch_id, renderer->kernel_times[event_id].kernels[launch_id].name, kernel_time,
      100.0f * kernel_time / event_time);

    total_kernel_time += kernel_time;
  }

  info_message("Total Time spent in Kernels: %07.2fms (%05.2f%%)", total_kernel_time, 100.0f * total_kernel_time / event_time);
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  // Convert from milliseconds to seconds.
  event_time *= 0.001f;

  renderer->total_render_time[event_id] = renderer->total_render_time[(event_id - 1) & DEVICE_RENDERER_TIMING_EVENTS_MASK] + event_time;

  renderer->last_time = event_time;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_get_render_time(DeviceRenderer* renderer, uint32_t event_id, float* time) {
  __CHECK_NULL_ARGUMENT(renderer);

  if (event_id >= renderer->event_id) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Renderer tried to update time for event %u when the next event is %u", event_id, renderer->event_id);
  }

  if (event_id + DEVICE_RENDERER_TIMING_EVENTS_MASK < renderer->event_id) {
    warn_message("Returned render time is stale.");
  }

  *time = renderer->total_render_time[event_id & DEVICE_RENDERER_TIMING_EVENTS_MASK];

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_destroy(DeviceRenderer** renderer) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(*renderer);

  __FAILURE_HANDLE(array_destroy(&(*renderer)->queue));

  for (uint32_t event_id = 0; event_id < DEVICE_RENDERER_TIMING_EVENTS_COUNT; event_id++) {
    __FAILURE_HANDLE(cuEventDestroy((*renderer)->time_start[event_id]));
    __FAILURE_HANDLE(cuEventDestroy((*renderer)->time_end[event_id]));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
    for (uint32_t timer_id = 0; timer_id < DEVICE_RENDERER_MAX_TIMED_KERNELS; timer_id++) {
      __FAILURE_HANDLE(cuEventDestroy((*renderer)->kernel_times[event_id].kernels[timer_id].time_start));
      __FAILURE_HANDLE(cuEventDestroy((*renderer)->kernel_times[event_id].kernels[timer_id].time_end));
    }

    __FAILURE_HANDLE(array_destroy(&(*renderer)->kernel_times[event_id].kernels));
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */
  }

  __FAILURE_HANDLE(host_free(renderer));

  return LUMINARY_SUCCESS;
}
