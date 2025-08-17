#include "device_renderer.h"

#include "device.h"
#include "internal_error.h"

struct DeviceRendererWork {
  uint32_t event_id;
  uint32_t launch_id;
  SampleCountSlice* sample_count;
  bool allow_gbuffer_meta_query;
  DeviceRenderCallbackData* shared_callback_data;
} typedef DeviceRendererWork;

LuminaryResult device_renderer_create(DeviceRenderer** renderer) {
  __CHECK_NULL_ARGUMENT(renderer);

  __FAILURE_HANDLE(host_malloc(renderer, sizeof(DeviceRenderer)));
  memset(*renderer, 0, sizeof(DeviceRenderer));

  __FAILURE_HANDLE(array_create(&(*renderer)->prepass_queue, sizeof(DeviceRendererQueueAction), 16));
  __FAILURE_HANDLE(array_create(&(*renderer)->queue, sizeof(DeviceRendererQueueAction), 16));
  __FAILURE_HANDLE(array_create(&(*renderer)->postpass_queue, sizeof(DeviceRendererQueueAction), 16));

  for (uint32_t event_id = 0; event_id < DEVICE_RENDERER_TIMING_EVENTS_COUNT; event_id++) {
    __FAILURE_HANDLE(cuEventCreate(&(*renderer)->time_start[event_id], 0));
    __FAILURE_HANDLE(cuEventCreate(&(*renderer)->time_end[event_id], 0));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
    for (uint32_t timer_id = 0; timer_id < DEVICE_RENDERER_MAX_TIMED_KERNELS; timer_id++) {
      memset(&(*renderer)->kernel_times[event_id].kernels[timer_id], 0, sizeof(DeviceRendererKernelTimer));

      for (uint32_t tile_id = 0; tile_id < DEVICE_RENDERER_MAX_TIMED_TILES; tile_id++) {
        __FAILURE_HANDLE(cuEventCreate(&(*renderer)->kernel_times[event_id].kernels[timer_id].time_start[tile_id], 0));
        __FAILURE_HANDLE(cuEventCreate(&(*renderer)->kernel_times[event_id].kernels[timer_id].time_end[tile_id], 0));
      }
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

    if (depth == 0) {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_TASKS_CREATE;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
    action.optix_type = OPTIX_KERNEL_TYPE_RAYTRACE;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    if (args->render_volumes) {
      action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
      action.optix_type = OPTIX_KERNEL_TYPE_SHADING_VOLUME_GEO;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));

      action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
      action.optix_type = OPTIX_KERNEL_TYPE_SHADING_VOLUME_SKY;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
    action.cuda_type = CUDA_KERNEL_TYPE_VOLUME_PROCESS_EVENTS;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

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
    action.cuda_type = CUDA_KERNEL_TYPE_TASKS_SORT;
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

    if (args->render_volumes && depth != args->max_depth) {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_VOLUME_PROCESS_TASKS;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    if (args->render_particles) {
      action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
      action.optix_type = OPTIX_KERNEL_TYPE_SHADING_PARTICLES_GEO;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));

      action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
      action.optix_type = OPTIX_KERNEL_TYPE_SHADING_PARTICLES_SKY;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    if (args->render_particles && depth != args->max_depth) {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_PARTICLE_PROCESS_TASKS;
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
  action.cuda_type = CUDA_KERNEL_TYPE_TASKS_CREATE;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
  action.optix_type = OPTIX_KERNEL_TYPE_RAYTRACE;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
  action.cuda_type = CUDA_KERNEL_TYPE_VOLUME_PROCESS_EVENTS;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  if (args->render_inscattering) {
    action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
    action.cuda_type = CUDA_KERNEL_TYPE_SKY_PROCESS_INSCATTERING_EVENTS;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));
  }

  action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
  action.cuda_type = CUDA_KERNEL_TYPE_TASKS_SORT;
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

  __FAILURE_HANDLE(array_set_num_elements(&renderer->prepass_queue, 0));
  __FAILURE_HANDLE(array_set_num_elements(&renderer->queue, 0));
  __FAILURE_HANDLE(array_set_num_elements(&renderer->postpass_queue, 0));

  DeviceRendererQueueAction action;

  ////////////////////////////////////////////////////////////////////
  // Prepass Queue
  ////////////////////////////////////////////////////////////////////

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_START_OF_SAMPLE;
  __FAILURE_HANDLE(array_push(&renderer->prepass_queue, &action));

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_CONST_MEM;
  __FAILURE_HANDLE(array_push(&renderer->prepass_queue, &action));

  ////////////////////////////////////////////////////////////////////
  // Main Queue
  ////////////////////////////////////////////////////////////////////

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_TILE_ID;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_QUEUE_CONTINUATION;
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

  ////////////////////////////////////////////////////////////////////
  // Postpass Queue
  ////////////////////////////////////////////////////////////////////

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_END_OF_SAMPLE;
  __FAILURE_HANDLE(array_push(&renderer->postpass_queue, &action));

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

  renderer->tile_id = 0;
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

  if (*launch_id < DEVICE_RENDERER_MAX_TIMED_KERNELS && renderer->tile_id < DEVICE_RENDERER_MAX_TIMED_TILES) {
    DeviceRendererKernelTimer* timer = renderer->kernel_times[event_id].kernels + *launch_id;

    __FAILURE_HANDLE(kernel_get_name(device->cuda_kernels[type], &timer->name));

    if (timer->num_queued_tile_executions < DEVICE_RENDERER_MAX_TIMED_TILES) {
      CUDA_FAILURE_HANDLE(cuEventRecord(timer->time_start[timer->num_queued_tile_executions], device->stream_main));
    }
  }
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  __FAILURE_HANDLE(kernel_execute(device->cuda_kernels[type], device->stream_main));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  if (*launch_id < DEVICE_RENDERER_MAX_TIMED_KERNELS && renderer->tile_id < DEVICE_RENDERER_MAX_TIMED_TILES) {
    DeviceRendererKernelTimer* timer = renderer->kernel_times[event_id].kernels + *launch_id;

    if (timer->num_queued_tile_executions < DEVICE_RENDERER_MAX_TIMED_TILES) {
      CUDA_FAILURE_HANDLE(cuEventRecord(timer->time_end[timer->num_queued_tile_executions++], device->stream_main));
    }
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

  if (*launch_id < DEVICE_RENDERER_MAX_TIMED_KERNELS && renderer->tile_id < DEVICE_RENDERER_MAX_TIMED_TILES) {
    DeviceRendererKernelTimer* timer = renderer->kernel_times[event_id].kernels + *launch_id;

    timer->name = "Optix Kernel";

    if (timer->num_queued_tile_executions < DEVICE_RENDERER_MAX_TIMED_TILES) {
      CUDA_FAILURE_HANDLE(cuEventRecord(timer->time_start[timer->num_queued_tile_executions], device->stream_main));
    }
  }
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  __FAILURE_HANDLE(optix_kernel_execute(device->optix_kernels[type], device));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  if (*launch_id < DEVICE_RENDERER_MAX_TIMED_KERNELS && renderer->tile_id < DEVICE_RENDERER_MAX_TIMED_TILES) {
    DeviceRendererKernelTimer* timer = renderer->kernel_times[event_id].kernels + *launch_id;

    if (timer->num_queued_tile_executions < DEVICE_RENDERER_MAX_TIMED_TILES) {
      CUDA_FAILURE_HANDLE(cuEventRecord(timer->time_end[timer->num_queued_tile_executions++], device->stream_main));
    }
  }
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  *launch_id = *launch_id + 1;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_handle_queue_action(
  DeviceRenderer* renderer, Device* device, DeviceRendererWork* work, const DeviceRendererQueueAction* action) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);

  switch (action->type) {
    case DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL:
      __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, action->cuda_type, &work->launch_id));
      break;
    case DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL:
      __FAILURE_HANDLE(_device_renderer_queue_optix_kernel(renderer, device, action->optix_type, &work->launch_id));
      break;
    case DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_CONST_MEM:
      __FAILURE_HANDLE(device_update_dynamic_const_mem(device, work->sample_count->current_sample_count, 0xFFFF, 0xFFFF));
      break;
    case DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_TILE_ID:
      __FAILURE_HANDLE(device_update_tile_id_const_mem(device, renderer->tile_id));
      break;
    case DEVICE_RENDERER_QUEUE_ACTION_TYPE_UPDATE_DEPTH:
      __FAILURE_HANDLE(device_update_depth_const_mem(device, action->mem_update.depth));

      const bool allow_gbuffer_meta_query = work->allow_gbuffer_meta_query && (action->mem_update.depth > 0);
      if (allow_gbuffer_meta_query && (device->gbuffer_meta_state == GBUFFER_META_STATE_NOT_READY)) {
        __FAILURE_HANDLE(device_query_gbuffer_meta(device));
      }
      break;
    case DEVICE_RENDERER_QUEUE_ACTION_TYPE_QUEUE_CONTINUATION:
      CUDA_FAILURE_HANDLE(cuEventRecord(device->event_queue_render, device->stream_main));
      CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_callbacks, device->event_queue_render, CU_EVENT_WAIT_DEFAULT));
      CUDA_FAILURE_HANDLE(
        cuLaunchHostFunc(device->stream_callbacks, renderer->registered_callback_continue_func, work->shared_callback_data));
      break;
    case DEVICE_RENDERER_QUEUE_ACTION_TYPE_START_OF_SAMPLE:
      CUDA_FAILURE_HANDLE(cuEventRecord(renderer->time_start[work->event_id], device->stream_main));
      break;
    case DEVICE_RENDERER_QUEUE_ACTION_TYPE_END_OF_SAMPLE:
      if (device->constant_memory->settings.shading_mode == LUMINARY_SHADING_MODE_DEFAULT) {
        if (device->aggregate_sample_count == 0) {
          __FAILURE_HANDLE(
            _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_FIRST_SAMPLE, &work->launch_id));
        }
        else {
          __FAILURE_HANDLE(
            _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_UPDATE, &work->launch_id));

          // TODO: This is only needed when outputting
          if (device->constant_memory->camera.do_firefly_rejection) {
            __FAILURE_HANDLE(
              _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_OUTPUT, &work->launch_id));
          }
          else {
            __FAILURE_HANDLE(
              _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_OUTPUT_RAW, &work->launch_id));
          }
        }
      }
      else {
        __FAILURE_HANDLE(
          _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TEMPORAL_ACCUMULATION_AOV, &work->launch_id));
      }

      CUDA_FAILURE_HANDLE(cuEventRecord(renderer->time_end[work->event_id], device->stream_main));
      CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_callbacks, renderer->time_end[work->event_id], CU_EVENT_WAIT_DEFAULT));
      CUDA_FAILURE_HANDLE(
        cuLaunchHostFunc(device->stream_callbacks, renderer->registered_callback_finished_func, work->shared_callback_data));

      renderer->event_id++;

      if (work->allow_gbuffer_meta_query && device->gbuffer_meta_state == GBUFFER_META_STATE_NOT_READY) {
        __FAILURE_HANDLE(device_query_gbuffer_meta(device));
      }

      break;
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_execute_queue(
  DeviceRenderer* renderer, Device* device, DeviceRendererWork* work, const ARRAY DeviceRendererQueueAction* queue) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);
  __CHECK_NULL_ARGUMENT(work);
  __CHECK_NULL_ARGUMENT(queue);

  uint32_t num_actions;
  __FAILURE_HANDLE(array_get_num_elements(queue, &num_actions));

  for (uint32_t action_id = 0; action_id < num_actions; action_id++) {
    const DeviceRendererQueueAction* action = queue + action_id;

    __FAILURE_HANDLE(_device_renderer_handle_queue_action(renderer, device, work, action));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_continue(DeviceRenderer* renderer, Device* device, SampleCountSlice* sample_count) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);

  if (sample_count->current_sample_count == sample_count->end_sample_count)
    return LUMINARY_SUCCESS;

  ////////////////////////////////////////////////////////////////////
  // Gather data
  ////////////////////////////////////////////////////////////////////

  uint32_t event_id = renderer->event_id & DEVICE_RENDERER_TIMING_EVENTS_MASK;

  // Same callback data is used for continue and finished callbacks
  DeviceRenderCallbackData* shared_callback_data = renderer->callback_data + event_id;

  shared_callback_data->common          = renderer->common_callback_data;
  shared_callback_data->render_id       = renderer->render_id;
  shared_callback_data->render_event_id = renderer->event_id;

  const uint32_t undersampling_stage = (device->undersampling_state & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;

  uint32_t tile_count;
  __FAILURE_HANDLE(device_renderer_get_tile_count(renderer, device, undersampling_stage, &tile_count));

  // Query only during the first sample and if enough samples have been computed after this iteration.
  const bool allow_gbuffer_meta_query = undersampling_stage <= (1 + device->constant_memory->settings.supersampling);

  renderer->is_rendering_first_sample = (device->undersampling_state & UNDERSAMPLING_FIRST_SAMPLE_MASK) != 0;

  ////////////////////////////////////////////////////////////////////
  // Setup work
  ////////////////////////////////////////////////////////////////////

  DeviceRendererWork work;
  work.event_id                 = event_id;
  work.launch_id                = 0;
  work.sample_count             = sample_count;
  work.allow_gbuffer_meta_query = allow_gbuffer_meta_query;
  work.shared_callback_data     = shared_callback_data;

  ////////////////////////////////////////////////////////////////////
  // Execute
  ////////////////////////////////////////////////////////////////////

  if (renderer->tile_id == 0) {
    __FAILURE_HANDLE(_device_renderer_execute_queue(renderer, device, &work, renderer->prepass_queue));
  }

  __FAILURE_HANDLE(_device_renderer_execute_queue(renderer, device, &work, renderer->queue));
  renderer->tile_id++;

  if (renderer->tile_id == tile_count) {
    __FAILURE_HANDLE(_device_renderer_execute_queue(renderer, device, &work, renderer->postpass_queue));

    renderer->tile_id = 0;
  }

  ////////////////////////////////////////////////////////////////////
  // Exit
  ////////////////////////////////////////////////////////////////////

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  renderer->kernel_times[event_id].num_kernel_launches = min(work.launch_id, DEVICE_RENDERER_MAX_TIMED_KERNELS);
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_update_render_time(DeviceRenderer* renderer, uint32_t target_event_id) {
  __CHECK_NULL_ARGUMENT(renderer);

  if (target_event_id >= renderer->event_id) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Renderer tried to update time for event %u when the next event is %u", target_event_id,
      renderer->event_id);
  }

  while (target_event_id >= renderer->timing_event_id) {
    uint32_t event_id = renderer->timing_event_id & DEVICE_RENDERER_TIMING_EVENTS_MASK;

    float event_time;
    CUDA_FAILURE_HANDLE(cuEventElapsedTime(&event_time, renderer->time_start[event_id], renderer->time_end[event_id]));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
    const uint32_t num_kernel_launches = renderer->kernel_times[event_id].num_kernel_launches;

    info_message("[EventIdx: %05u] ------- Kernel Times -------", event_id);

    float total_render_time = 0.0f;

    for (uint32_t launch_id = 0; launch_id < num_kernel_launches; launch_id++) {
      float total_kernel_time = 0.0f;

      const uint32_t num_tiles = renderer->kernel_times[event_id].kernels[launch_id].num_queued_tile_executions;

      for (uint32_t tile_id = 0; tile_id < num_tiles; tile_id++) {
        float kernel_time;
        CUDA_FAILURE_HANDLE(cuEventElapsedTime(
          &kernel_time, renderer->kernel_times[event_id].kernels[launch_id].time_start[tile_id],
          renderer->kernel_times[event_id].kernels[launch_id].time_end[tile_id]));

        total_kernel_time += kernel_time;
      }

      renderer->kernel_times[event_id].kernels[launch_id].num_queued_tile_executions = 0;

      info_message(
        "[LaunchIdx: %03u] %32s | %07.2fms (%05.2f%%)", launch_id, renderer->kernel_times[event_id].kernels[launch_id].name,
        total_kernel_time, 100.0f * total_kernel_time / event_time);

      total_render_time += total_kernel_time;
    }

    info_message("Total Time spent in Kernels: %07.2fms (%05.2f%%)", total_render_time, 100.0f * total_render_time / event_time);
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

    // Convert from milliseconds to seconds.
    event_time *= 0.001f;

    renderer->total_render_time[event_id] = renderer->total_render_time[(event_id - 1) & DEVICE_RENDERER_TIMING_EVENTS_MASK] + event_time;

    renderer->last_time = event_time;

    renderer->timing_event_id++;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_get_render_time(DeviceRenderer* renderer, uint32_t event_id, float* time) {
  __CHECK_NULL_ARGUMENT(renderer);

  if (event_id >= renderer->event_id) {
    __RETURN_ERROR(
      LUMINARY_ERROR_API_EXCEPTION, "Renderer tried to update time for event %u when the next event is %u", event_id, renderer->event_id);
  }

  __FAILURE_HANDLE(device_renderer_update_render_time(renderer, event_id));

  if (event_id + DEVICE_RENDERER_TIMING_EVENTS_MASK < renderer->event_id) {
    warn_message("Returned render time is stale.");
  }

  *time = renderer->total_render_time[event_id & DEVICE_RENDERER_TIMING_EVENTS_MASK];

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_get_latest_event_id(DeviceRenderer* renderer, uint32_t* event_id) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(event_id);

  if (renderer->event_id == 0) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Renderer has not been executed yet.");
  }

  // Event ID is the ID of the next event so we need to decrement.
  *event_id = renderer->event_id - 1;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_get_status(DeviceRenderer* renderer, uint32_t* status) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(status);

  *status = DEVICE_RENDERER_STATUS_FLAGS_NONE;

  if (renderer->tile_id == 0) {
    *status |= DEVICE_RENDERER_STATUS_FLAGS_READY;
  }
  else {
    *status |= DEVICE_RENDERER_STATUS_FLAGS_IN_PROGRESS;
  }

  if (renderer->is_rendering_first_sample) {
    *status |= DEVICE_RENDERER_STATUS_FLAGS_FIRST_SAMPLE;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_get_tile_count(
  DeviceRenderer* renderer, Device* device, uint32_t undersampling_stage, uint32_t* tile_count) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(tile_count);

  uint32_t width;
  uint32_t height;
  __FAILURE_HANDLE(device_get_internal_resolution(device, &width, &height));

  const uint32_t internal_width_this_sample  = (width + (1 << undersampling_stage) - 1) >> undersampling_stage;
  const uint32_t internal_height_this_sample = (height + (1 << undersampling_stage) - 1) >> undersampling_stage;
  const uint32_t internal_pixels_this_sample = internal_width_this_sample * internal_height_this_sample;

  uint32_t allocated_tasks;
  __FAILURE_HANDLE(device_get_allocated_task_count(device, &allocated_tasks));

  *tile_count = (internal_pixels_this_sample + allocated_tasks - 1) / allocated_tasks;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_destroy(DeviceRenderer** renderer) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(*renderer);

  __FAILURE_HANDLE(array_destroy(&(*renderer)->prepass_queue));
  __FAILURE_HANDLE(array_destroy(&(*renderer)->queue));
  __FAILURE_HANDLE(array_destroy(&(*renderer)->postpass_queue));

  for (uint32_t event_id = 0; event_id < DEVICE_RENDERER_TIMING_EVENTS_COUNT; event_id++) {
    __FAILURE_HANDLE(cuEventDestroy((*renderer)->time_start[event_id]));
    __FAILURE_HANDLE(cuEventDestroy((*renderer)->time_end[event_id]));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
    for (uint32_t timer_id = 0; timer_id < DEVICE_RENDERER_MAX_TIMED_KERNELS; timer_id++) {
      for (uint32_t tile_id = 0; tile_id < DEVICE_RENDERER_MAX_TIMED_TILES; tile_id++) {
        __FAILURE_HANDLE(cuEventDestroy((*renderer)->kernel_times[event_id].kernels[timer_id].time_start[tile_id]));
        __FAILURE_HANDLE(cuEventDestroy((*renderer)->kernel_times[event_id].kernels[timer_id].time_end[tile_id]));
      }
    }
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */
  }

  __FAILURE_HANDLE(host_free(renderer));

  return LUMINARY_SUCCESS;
}
