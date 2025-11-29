#include "device_renderer.h"

#include "device.h"
#include "device_adaptive_sampler.h"
#include "internal_error.h"

struct DeviceRendererWork {
  uint32_t event_id;
  uint32_t launch_id;
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
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_VOLUME_PROCESS_INSCATTERING;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));

      action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
      action.optix_type = OPTIX_KERNEL_TYPE_SHADOW_VOLUME;
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

    action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
    action.cuda_type = CUDA_KERNEL_TYPE_GEOMETRY_PROCESS_TASKS;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    if (args->render_particles) {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_PARTICLE_PROCESS_TASKS;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    if (args->render_ocean) {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_OCEAN_PROCESS_TASKS;
      __FAILURE_HANDLE(array_push(&renderer->queue, &action));
    }

    action.type       = DEVICE_RENDERER_QUEUE_ACTION_TYPE_OPTIX_KERNEL;
    action.optix_type = OPTIX_KERNEL_TYPE_SHADOW;
    __FAILURE_HANDLE(array_push(&renderer->queue, &action));

    if (args->render_volumes && depth != args->max_depth) {
      action.type      = DEVICE_RENDERER_QUEUE_ACTION_TYPE_CUDA_KERNEL;
      action.cuda_type = CUDA_KERNEL_TYPE_VOLUME_PROCESS_TASKS;
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

static LuminaryResult _device_renderer_build_kernel_queue(DeviceRenderer* renderer, DeviceRendererQueueArgs* args) {
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

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_START_OF_TILE;
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

  action.type = DEVICE_RENDERER_QUEUE_ACTION_TYPE_END_OF_TILE;
  __FAILURE_HANDLE(array_push(&renderer->queue, &action));

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

LuminaryResult device_renderer_init_new_render(DeviceRenderer* renderer, DeviceRendererQueueArgs* args) {
  __CHECK_NULL_ARGUMENT(renderer);

  memset(renderer->executed_aggregate_sample_counts, 0, sizeof(renderer->executed_aggregate_sample_counts));

  renderer->tile_id = 0;
  renderer->render_id++;

  for (uint32_t event_id = 0; event_id < DEVICE_RENDERER_TIMING_EVENTS_COUNT; event_id++) {
    renderer->total_render_time[event_id] = 0.0f;
  }

  __FAILURE_HANDLE(_device_renderer_build_kernel_queue(renderer, args));

  renderer->status_flags = DEVICE_RENDERER_STATUS_FLAG_READY;

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

static LuminaryResult _device_renderer_queue_adaptive_sampling_update(DeviceRenderer* renderer, Device* device, DeviceRendererWork* work) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);

  // Only main device can compute adaptive sampling counts
  if (device->is_main_device == false)
    return LUMINARY_SUCCESS;

  // We have reached the last stage, there is nothing left to do.
  if (renderer->sample_allocation.stage_id >= ADAPTIVE_SAMPLER_NUM_STAGES)
    return LUMINARY_SUCCESS;

  if (renderer->executed_aggregate_sample_counts[renderer->sample_allocation.stage_id] < 64)
    return LUMINARY_SUCCESS;

  work->shared_callback_data->adaptive_sampling_build_stage_id = renderer->sample_allocation.stage_id + 1;

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
      __FAILURE_HANDLE(device_update_dynamic_const_mem(device, renderer->sample_allocation));
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
      if ((device->undersampling_state & UNDERSAMPLING_STAGE_MASK) != 0) {
        __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(
          renderer, device, CUDA_KERNEL_TYPE_ACCUMULATION_GENERATE_RESULT_UNDERSAMPLING, &work->launch_id));
      }
      else {
        __FAILURE_HANDLE(
          _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_ACCUMULATION_GENERATE_RESULT, &work->launch_id));
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
    case DEVICE_RENDERER_QUEUE_ACTION_TYPE_START_OF_TILE:
      __FAILURE_HANDLE(device_update_tile_id_const_mem(device, renderer->tile_id));
      break;
    case DEVICE_RENDERER_QUEUE_ACTION_TYPE_END_OF_TILE:
      if (renderer->executed_aggregate_sample_counts[0] == 0) {
        // We assume that every device will always execute the base adaptive sampling stage first.
        __DEBUG_ASSERT(renderer->executed_aggregate_sample_counts[1] == 0);

        __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(
          renderer, device, CUDA_KERNEL_TYPE_ACCUMULATION_COLLECT_RESULTS_FIRST_SAMPLE, &work->launch_id));
      }
      else {
        __FAILURE_HANDLE(
          _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_ACCUMULATION_COLLECT_RESULTS, &work->launch_id));
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

LuminaryResult device_renderer_continue(DeviceRenderer* renderer, Device* device) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);

  if (renderer->shutdown)
    return LUMINARY_SUCCESS;

  // Abort render if we exceeded maximum supported sample count
  if (renderer->sample_allocation.global_sample_id >= MAX_NUM_GLOBAL_SAMPLES)
    return LUMINARY_SUCCESS;

  // Renderer has no samples allocated.
  if (renderer->sample_allocation.num_samples == 0)
    return LUMINARY_SUCCESS;

  if ((renderer->status_flags & DEVICE_RENDERER_STATUS_FLAG_FINISHED) != 0)
    return LUMINARY_SUCCESS;

  if ((renderer->status_flags & DEVICE_RENDERER_STATUS_FLAG_READY) != 0) {
    renderer->status_flags &= ~DEVICE_RENDERER_STATUS_FLAG_READY;
    renderer->status_flags |= DEVICE_RENDERER_STATUS_FLAG_IN_PROGRESS;

    if ((device->undersampling_state & UNDERSAMPLING_FIRST_SAMPLE_MASK) != 0)
      renderer->status_flags |= DEVICE_RENDERER_STATUS_FLAG_FIRST_SAMPLE;
  }

  ////////////////////////////////////////////////////////////////////
  // Gather data
  ////////////////////////////////////////////////////////////////////

  uint32_t event_id = renderer->event_id & DEVICE_RENDERER_TIMING_EVENTS_MASK;

  // Same callback data is used for continue and finished callbacks
  DeviceRenderCallbackData* shared_callback_data = renderer->callback_data + event_id;

  shared_callback_data->common                           = renderer->common_callback_data;
  shared_callback_data->render_id                        = renderer->render_id;
  shared_callback_data->render_event_id                  = renderer->event_id;
  shared_callback_data->adaptive_sampling_build_stage_id = ADAPTIVE_SAMPLING_STAGE_INVALID;

  const uint32_t undersampling_stage = (device->undersampling_state & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;

  uint32_t tile_count;
  __FAILURE_HANDLE(device_renderer_get_tile_count(renderer, device, undersampling_stage, &tile_count));

  // We have already computed all tiles of this iteration.
  if (renderer->tile_id >= tile_count)
    return LUMINARY_SUCCESS;

  // Query only during the first sample and if enough samples have been computed after this iteration.
  const bool allow_gbuffer_meta_query = undersampling_stage <= (1 + device->constant_memory->settings.supersampling);

  ////////////////////////////////////////////////////////////////////
  // Setup work
  ////////////////////////////////////////////////////////////////////

  DeviceRendererWork work;
  work.event_id                 = event_id;
  work.launch_id                = 0;
  work.allow_gbuffer_meta_query = allow_gbuffer_meta_query;
  work.shared_callback_data     = shared_callback_data;

  __FAILURE_HANDLE(_device_renderer_queue_adaptive_sampling_update(renderer, device, &work));

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

    renderer->status_flags &= ~DEVICE_RENDERER_STATUS_FLAG_IN_PROGRESS;
    renderer->status_flags |= DEVICE_RENDERER_STATUS_FLAG_FINISHED;
  }

  ////////////////////////////////////////////////////////////////////
  // Exit
  ////////////////////////////////////////////////////////////////////

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  renderer->kernel_times[event_id].num_kernel_launches = min(work.launch_id, DEVICE_RENDERER_MAX_TIMED_KERNELS);
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_finish_iteration(DeviceRenderer* renderer, bool is_undersampling) {
  __CHECK_NULL_ARGUMENT(renderer);

  renderer->status_flags &= ~(DEVICE_RENDERER_STATUS_FLAG_FINISHED | DEVICE_RENDERER_STATUS_FLAG_FIRST_SAMPLE);
  renderer->status_flags |= DEVICE_RENDERER_STATUS_FLAG_READY;

  renderer->tile_id = 0;

  if (is_undersampling == false) {
    renderer->executed_aggregate_sample_counts[renderer->sample_allocation.stage_id]++;
    __FAILURE_HANDLE(device_sample_allocation_step_next(&renderer->sample_allocation));
  }

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

LuminaryResult device_renderer_allocate_sample(DeviceRenderer* renderer, DeviceAdaptiveSampler* sampler) {
  __CHECK_NULL_ARGUMENT(renderer);

  if (renderer->sample_allocation.num_samples == 0)
    __FAILURE_HANDLE(device_adaptive_sampler_allocate_sample(sampler, &renderer->sample_allocation, 1));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_externalize_samples(
  DeviceRenderer* renderer, uint32_t stage_sample_counts[ADAPTIVE_SAMPLER_NUM_STAGES + 1]) {
  __CHECK_NULL_ARGUMENT(renderer);

  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES + 1; stage_id++) {
    stage_sample_counts[stage_id] = renderer->executed_aggregate_sample_counts[stage_id];
  }

  memset(renderer->executed_aggregate_sample_counts, 0, sizeof(renderer->executed_aggregate_sample_counts));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_register_external_samples(
  DeviceRenderer* renderer, uint32_t stage_sample_counts[ADAPTIVE_SAMPLER_NUM_STAGES + 1]) {
  __CHECK_NULL_ARGUMENT(renderer);

  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES + 1; stage_id++) {
    renderer->executed_aggregate_sample_counts[stage_id] += stage_sample_counts[stage_id];
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

LuminaryResult device_renderer_get_status(DeviceRenderer* renderer, DeviceRendererStatusFlags* status) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(status);

  *status = renderer->status_flags;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_get_tile_count(
  DeviceRenderer* renderer, Device* device, uint32_t undersampling_stage, uint32_t* tile_count) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(tile_count);

  uint32_t upper_bounds_total_tasks;
  if (undersampling_stage) {
    __DEBUG_ASSERT(renderer->sample_allocation.stage_id == 0);

    uint32_t width;
    uint32_t height;
    __FAILURE_HANDLE(device_get_internal_render_resolution(device, &width, &height));

    const uint32_t internal_width_this_sample  = (width + (1 << undersampling_stage) - 1) >> undersampling_stage;
    const uint32_t internal_height_this_sample = (height + (1 << undersampling_stage) - 1) >> undersampling_stage;
    const uint32_t internal_pixels_this_sample = internal_width_this_sample * internal_height_this_sample;

    upper_bounds_total_tasks = internal_pixels_this_sample;
  }
  else {
    upper_bounds_total_tasks = renderer->sample_allocation.upper_bound_tasks_per_sample;
  }

  uint32_t allocated_tasks;
  __FAILURE_HANDLE(device_get_allocated_task_count(device, &allocated_tasks));

  *tile_count = (upper_bounds_total_tasks + allocated_tasks - 1) / allocated_tasks;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_get_total_executed_samples(DeviceRenderer* renderer, uint32_t* aggregate_sample_count) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(aggregate_sample_count);

  uint32_t sample_count = 0;

  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES + 1; stage_id++) {
    sample_count += renderer->executed_aggregate_sample_counts[stage_id];
  }

  *aggregate_sample_count = sample_count;

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_shutdown(DeviceRenderer* renderer) {
  __CHECK_NULL_ARGUMENT(renderer);

  renderer->shutdown = true;

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
