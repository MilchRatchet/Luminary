#include "device_renderer.h"

#include "device.h"
#include "device_adaptive_sampler.h"
#include "internal_error.h"

enum DeviceRendererCommandType {
  DEVICE_RENDERER_COMMAND_START,
  DEVICE_RENDERER_COMMAND_CONTINUE,
  DEVICE_RENDERER_COMMAND_ABORT_RENDER,
  DEVICE_RENDERER_COMMAND_SHUTDOWN
} typedef DeviceRendererCommandType;

struct DeviceRendererCommand {
  DeviceRendererCommandType type;
  DeviceRendererQueueArgs args;
} typedef DeviceRendererCommand;

enum DeviceRendererReplyType {
  DEVICE_RENDERER_REPLY_SUBMITTED,
  DEVICE_RENDERER_REPLY_ABORT_SUBMITTED,
  DEVICE_RENDERER_REPLY_HAS_SHUT_DOWN
} typedef DeviceRendererReplyType;

struct DeviceRendererReply {
  DeviceRendererReplyType type;
  LuminaryResult result;
} typedef DeviceRendererReply;

static LuminaryResult _device_renderer_thread_main(void* args);

LuminaryResult device_renderer_create(DeviceRenderer** renderer, Device* device, uint64_t host_id) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);

  __FAILURE_HANDLE(host_malloc(renderer, sizeof(DeviceRenderer)));
  memset(*renderer, 0, sizeof(DeviceRenderer));

  CUDA_FAILURE_HANDLE(cuStreamCreateWithPriority(&(*renderer)->stream, CU_STREAM_NON_BLOCKING, 2));
  CUDA_FAILURE_HANDLE(cuEventCreate(&(*renderer)->manager_ready, CU_EVENT_DISABLE_TIMING));
  CUDA_FAILURE_HANDLE(cuEventCreate(&(*renderer)->renderer_ready, CU_EVENT_DISABLE_TIMING));
  CUDA_FAILURE_HANDLE(cuEventCreate(&(*renderer)->abort_flag_uploaded, CU_EVENT_DISABLE_TIMING));
  CUDA_FAILURE_HANDLE(cuEventCreate(&(*renderer)->abort_complete, CU_EVENT_DISABLE_TIMING));
  __FAILURE_HANDLE(device_malloc_staging(&(*renderer)->work_pending, sizeof(uint32_t), DEVICE_MEMORY_STAGING_FLAG_NONE));

  for (uint32_t event_id = 0; event_id < DEVICE_RENDERER_TIMING_EVENTS_COUNT; event_id++) {
    __FAILURE_HANDLE(cuEventCreate(&(*renderer)->time_start[event_id], 0));
    __FAILURE_HANDLE(cuEventCreate(&(*renderer)->time_end[event_id], 0));
  }

  (*renderer)->device  = device;
  (*renderer)->host_id = host_id;
  __FAILURE_HANDLE(queue_create(&(*renderer)->commands, sizeof(DeviceRendererCommand), 1));
  __FAILURE_HANDLE(queue_create(&(*renderer)->replies, sizeof(DeviceRendererReply), 1));
  __FAILURE_HANDLE(thread_create(&(*renderer)->thread));
  __FAILURE_HANDLE(thread_start((*renderer)->thread, _device_renderer_thread_main, *renderer));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_handle_callback(DeviceRenderer* renderer, DeviceRenderCallbackData* data, bool* is_valid) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(data);
  __CHECK_NULL_ARGUMENT(is_valid);

  *is_valid = (renderer->shutdown == false) && (renderer->device->state_abort == false) && renderer->render_id == data->render_id;

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

static LuminaryResult _device_renderer_init_new_render(DeviceRenderer* renderer, const DeviceRendererQueueArgs* args) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(args);

  memset(renderer->executed_aggregate_sample_counts, 0, sizeof(renderer->executed_aggregate_sample_counts));

  renderer->pass_id = 0;
  renderer->args    = *args;
  renderer->render_id++;

  for (uint32_t event_id = 0; event_id < DEVICE_RENDERER_TIMING_EVENTS_COUNT; event_id++) {
    renderer->total_render_time[event_id] = 0.0f;
  }

  renderer->status_flags = DEVICE_RENDERER_STATUS_FLAG_READY;

  return LUMINARY_SUCCESS;
}

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
static LuminaryResult _device_renderer_collect_kernel_timing(DeviceRendererKernelTimer* timer, uint32_t slot_id) {
  DeviceRendererKernelTimingSlot* slot = timer->slots + slot_id;
  if (slot->pending) {
    float elapsed_ms;
    CUDA_FAILURE_HANDLE(cuEventElapsedTime(&elapsed_ms, slot->start, slot->end));
    timer->total_ms += elapsed_ms;
    slot->pending = false;
  }
  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_start_kernel_timing(
  DeviceRenderer* renderer, uint32_t kernel_id, const char* name, DeviceRendererKernelTimingSlot** slot) {
  const uint32_t event_id          = renderer->event_id & DEVICE_RENDERER_TIMING_EVENTS_MASK;
  DeviceRendererKernelTimer* timer = renderer->kernel_times[event_id].kernels + kernel_id;
  const uint32_t slot_id           = renderer->pass_id % DEVICE_RENDERER_KERNEL_TIMING_SLOTS;
  *slot                            = timer->slots + slot_id;
  timer->name                      = name;

  __FAILURE_HANDLE(_device_renderer_collect_kernel_timing(timer, slot_id));
  if ((*slot)->start == 0) {
    CUDA_FAILURE_HANDLE(cuEventCreate(&(*slot)->start, 0));
    CUDA_FAILURE_HANDLE(cuEventCreate(&(*slot)->end, 0));
  }
  CUDA_FAILURE_HANDLE(cuEventRecord((*slot)->start, renderer->stream));
  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_end_kernel_timing(DeviceRenderer* renderer, DeviceRendererKernelTimingSlot* slot) {
  CUDA_FAILURE_HANDLE(cuEventRecord(slot->end, renderer->stream));
  slot->pending = true;
  return LUMINARY_SUCCESS;
}
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

static LuminaryResult _device_renderer_queue_cuda_kernel(DeviceRenderer* renderer, Device* device, CUDAKernelType type) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  const char* name;
  DeviceRendererKernelTimingSlot* slot;
  __FAILURE_HANDLE(kernel_get_name(device->cuda_kernels[type], &name));
  __FAILURE_HANDLE(_device_renderer_start_kernel_timing(renderer, type, name, &slot));
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  __FAILURE_HANDLE(kernel_execute(device->cuda_kernels[type], renderer->stream));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  __FAILURE_HANDLE(_device_renderer_end_kernel_timing(renderer, slot));
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */
  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_queue_optix_kernel(DeviceRenderer* renderer, Device* device, OptixKernelType type) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  static const char* const names[OPTIX_KERNEL_TYPE_COUNT] = {
    [OPTIX_KERNEL_TYPE_RAYTRACE]      = "OptiX raytrace",
    [OPTIX_KERNEL_TYPE_SHADOW]        = "OptiX shadow",
    [OPTIX_KERNEL_TYPE_SHADOW_VOLUME] = "OptiX volume shadow",
  };
  DeviceRendererKernelTimingSlot* slot;
  __FAILURE_HANDLE(_device_renderer_start_kernel_timing(renderer, CUDA_KERNEL_TYPE_COUNT + type, names[type], &slot));
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

  __FAILURE_HANDLE(optix_kernel_execute(device->optix_kernels[type], device));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
  __FAILURE_HANDLE(_device_renderer_end_kernel_timing(renderer, slot));
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */
  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_queue_adaptive_sampling_update(
  DeviceRenderer* renderer, Device* device, DeviceRenderCallbackData* callback_data) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);

  // Only main device can compute adaptive sampling counts
  if (device->is_main_device == false)
    return LUMINARY_SUCCESS;

  if (renderer->args.enable_adaptive_sampling == false)
    return LUMINARY_SUCCESS;

  const uint32_t stage_id = renderer->sample_allocation.stage_id;

  // We have reached the last stage, there is nothing left to do.
  if (stage_id >= ADAPTIVE_SAMPLER_NUM_STAGES)
    return LUMINARY_SUCCESS;

  const uint32_t samples_this_stage = renderer->executed_aggregate_sample_counts[stage_id];
  const uint32_t samples_to_update  = renderer->args.adaptive_sampling_update_interval << stage_id;

  if (samples_this_stage < samples_to_update)
    return LUMINARY_SUCCESS;

  callback_data->adaptive_sampling_build_stage_id = stage_id + 1;

  return LUMINARY_SUCCESS;
}

static bool _device_renderer_is_first_sample(const DeviceRenderer* renderer) {
  for (uint32_t stage_id = 0; stage_id <= ADAPTIVE_SAMPLER_NUM_STAGES; stage_id++) {
    if (renderer->executed_aggregate_sample_counts[stage_id] != 0)
      return false;
  }

  return true;
}

static LuminaryResult _device_renderer_queue_sample_start(DeviceRenderer* renderer, Device* device, uint32_t event_id) {
  CUDA_FAILURE_HANDLE(cuEventRecord(renderer->time_start[event_id], renderer->stream));

  const uint32_t thread_count = device->work_buffers->allocated_thread_count;
  __FAILURE_HANDLE(device_memset(device->work_buffers->task_queues, 0, 0, sizeof(DeviceTaskQueue) * thread_count, renderer->stream));
  __FAILURE_HANDLE(device_memset(device->work_buffers->trace_counts, 0, 0, sizeof(uint16_t) * thread_count, renderer->stream));

  if (renderer->sample_allocation.stage_id > 0 && _device_renderer_is_first_sample(renderer)) {
    const size_t moment_buffer_size = sizeof(float) * device->work_buffers->allocated_internal_pixel_count;
    for (uint32_t channel_id = 0; channel_id < FRAME_CHANNEL_COUNT; channel_id++) {
      __FAILURE_HANDLE(device_memset(device->work_buffers->frame_first_moment[channel_id], 0, 0, moment_buffer_size, renderer->stream));
    }

    __FAILURE_HANDLE(device_memset(device->work_buffers->frame_second_moment_luminance, 0, 0, moment_buffer_size, renderer->stream));
  }

  return device_update_dynamic_const_mem(device, renderer->sample_allocation, renderer->executed_aggregate_sample_counts);
}

static LuminaryResult _device_renderer_queue_main_pass(DeviceRenderer* renderer, Device* device) {
  const DeviceRendererQueueArgs* args = &renderer->args;

  __FAILURE_HANDLE(_device_renderer_queue_optix_kernel(renderer, device, OPTIX_KERNEL_TYPE_RAYTRACE));

  if (args->render_volumes) {
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_VOLUME_PROCESS_INSCATTERING));
    __FAILURE_HANDLE(_device_renderer_queue_optix_kernel(renderer, device, OPTIX_KERNEL_TYPE_SHADOW_VOLUME));
  }

  __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_VOLUME_PROCESS_EVENTS));

  if (args->render_clouds) {
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_CLOUD_PROCESS_TASKS));
  }

  if (args->render_inscattering) {
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_SKY_PROCESS_INSCATTERING_EVENTS));
  }

  __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TASKS_SORT));
  __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_GEOMETRY_PROCESS_TASKS));

  if (args->render_particles) {
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_PARTICLE_PROCESS_TASKS));
  }

  if (args->render_ocean) {
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_OCEAN_PROCESS_TASKS));
  }

  __FAILURE_HANDLE(_device_renderer_queue_optix_kernel(renderer, device, OPTIX_KERNEL_TYPE_SHADOW));

  if (args->render_volumes && args->max_depth != 0) {
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_VOLUME_PROCESS_TASKS));
  }

  if (args->render_procedural_sky) {
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_SKY_PROCESS_TASKS));
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_queue_debug_pass(DeviceRenderer* renderer, Device* device) {
  const DeviceRendererQueueArgs* args = &renderer->args;

  __FAILURE_HANDLE(_device_renderer_queue_optix_kernel(renderer, device, OPTIX_KERNEL_TYPE_RAYTRACE));
  __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_VOLUME_PROCESS_EVENTS));

  if (args->render_inscattering) {
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_SKY_PROCESS_INSCATTERING_EVENTS));
  }

  __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_TASKS_SORT));
  __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_GEOMETRY_PROCESS_TASKS_DEBUG));

  if (args->render_ocean) {
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_OCEAN_PROCESS_TASKS_DEBUG));
  }

  if (args->render_particles) {
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_PARTICLE_PROCESS_TASKS_DEBUG));
  }

  return _device_renderer_queue_cuda_kernel(renderer, device, CUDA_KERNEL_TYPE_SKY_PROCESS_TASKS_DEBUG);
}

static LuminaryResult _device_renderer_queue_continue_callback(
  DeviceRenderer* renderer, Device* device, DeviceRenderCallbackData* callback_data) {
  CUDA_FAILURE_HANDLE(cuEventRecord(device->event_queue_render, renderer->stream));
  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_callbacks, device->event_queue_render, CU_EVENT_WAIT_DEFAULT));
  CUDA_FAILURE_HANDLE(cuLaunchHostFunc(device->stream_callbacks, renderer->registered_callback_continue_func, callback_data));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_continue(DeviceRenderer* renderer, Device* device) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(device);

  if (device->state_abort)
    return LUMINARY_SUCCESS;

  // Renderer has no samples allocated.
  if (renderer->sample_allocation.num_samples == 0)
    return LUMINARY_SUCCESS;

  if ((renderer->status_flags & DEVICE_RENDERER_STATUS_FLAG_FINISHED) != 0)
    return LUMINARY_SUCCESS;

  const bool starts_sample = (renderer->status_flags & DEVICE_RENDERER_STATUS_FLAG_READY) != 0;

  if (starts_sample) {
    renderer->status_flags &= ~DEVICE_RENDERER_STATUS_FLAG_READY;
    renderer->status_flags |= DEVICE_RENDERER_STATUS_FLAG_IN_PROGRESS;

    if ((device->undersampling_state & UNDERSAMPLING_FIRST_SAMPLE_MASK) != 0)
      renderer->status_flags |= DEVICE_RENDERER_STATUS_FLAG_FIRST_SAMPLE;

    __FAILURE_HANDLE(device_constant_memory_manager_ensure_synced(device->constant_memory, device, device->stream_main));
  }

  ////////////////////////////////////////////////////////////////////
  // Gather data
  ////////////////////////////////////////////////////////////////////

  uint32_t event_id = renderer->event_id & DEVICE_RENDERER_TIMING_EVENTS_MASK;

  const uint32_t undersampling_stage = (device->undersampling_state & UNDERSAMPLING_STAGE_MASK) >> UNDERSAMPLING_STAGE_SHIFT;

  if (starts_sample)
    renderer->callback_id++;

  DeviceRenderCallbackData* shared_callback_data = renderer->callback_data + (renderer->callback_id & DEVICE_RENDERER_TIMING_EVENTS_MASK);

  if (starts_sample) {
    shared_callback_data->common                           = renderer->common_callback_data;
    shared_callback_data->render_id                        = renderer->render_id;
    shared_callback_data->render_event_id                  = renderer->event_id;
    shared_callback_data->adaptive_sampling_build_stage_id = ADAPTIVE_SAMPLING_STAGE_INVALID;
#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
    for (uint32_t kernel_id = 0; kernel_id < DEVICE_RENDERER_TIMED_KERNEL_TYPES; kernel_id++) {
      DeviceRendererKernelTimer* timer = renderer->kernel_times[event_id].kernels + kernel_id;
      timer->name                      = (const char*) 0;
      timer->total_ms                  = 0.0;
      for (uint32_t slot_id = 0; slot_id < DEVICE_RENDERER_KERNEL_TIMING_SLOTS; slot_id++) {
        timer->slots[slot_id].pending = false;
      }
    }
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */
  }

  const DeviceConstantMemory* constant_memory;
  __FAILURE_HANDLE(device_constant_memory_manager_get_host_buffer(device->constant_memory, &constant_memory));

  // Query only during the first sample and if enough samples have been computed after this iteration.
  const bool allow_gbuffer_meta_query = undersampling_stage <= (1 + constant_memory->settings.supersampling);

  ////////////////////////////////////////////////////////////////////
  // Setup work
  ////////////////////////////////////////////////////////////////////

  bool debug_pass;
  switch (renderer->args.shading_mode) {
    case LUMINARY_SHADING_MODE_ALBEDO:
    case LUMINARY_SHADING_MODE_DEPTH:
    case LUMINARY_SHADING_MODE_NORMAL:
    case LUMINARY_SHADING_MODE_IDENTIFICATION:
    case LUMINARY_SHADING_MODE_LIGHTS:
      debug_pass = true;
      break;
    default:
      debug_pass = false;
      break;
  }

  const uint32_t pass_id      = renderer->pass_id;
  const bool single_pass      = debug_pass || renderer->args.max_depth == 0;
  const bool starts_pass_pair = (pass_id & 1u) == 0;
  const bool check_pending    = single_pass ? pass_id > 0 : (starts_pass_pair == false && pass_id > 1);
  if (check_pending && *renderer->work_pending == 0) {
    __FAILURE_HANDLE(_device_renderer_queue_adaptive_sampling_update(renderer, device, shared_callback_data));

    if (allow_gbuffer_meta_query && device->gbuffer_meta_state == GBUFFER_META_STATE_NOT_READY) {
      __FAILURE_HANDLE(device_query_gbuffer_meta(device));
    }

    CUDA_FAILURE_HANDLE(cuEventRecord(device->event_queue_render, renderer->stream));
    CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_callbacks, device->event_queue_render, CU_EVENT_WAIT_DEFAULT));
    CUDA_FAILURE_HANDLE(cuLaunchHostFunc(device->stream_callbacks, renderer->registered_callback_finished_func, shared_callback_data));
    CUDA_FAILURE_HANDLE(cuLaunchHostFunc(device->stream_callbacks, renderer->registered_callback_continue_func, shared_callback_data));

    renderer->event_id++;
    renderer->status_flags &= ~DEVICE_RENDERER_STATUS_FLAG_IN_PROGRESS;
    renderer->status_flags |= DEVICE_RENDERER_STATUS_FLAG_FINISHED;

    return LUMINARY_SUCCESS;
  }

  const bool creates_tasks    = single_pass || starts_pass_pair;
  const bool collects_results = single_pass || starts_pass_pair == false;

  CUDA_FAILURE_HANDLE(cuEventRecord(renderer->manager_ready, device->stream_main));
  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(renderer->stream, renderer->manager_ready, CU_EVENT_WAIT_DEFAULT));

  if (starts_sample) {
    __FAILURE_HANDLE(_device_renderer_queue_sample_start(renderer, device, event_id));
  }

  if (creates_tasks) {
    const CUDAKernelType create_tasks =
      renderer->sample_allocation.stage_id > 0 ? CUDA_KERNEL_TYPE_TASKS_CREATE_ADAPTIVE_SAMPLING : CUDA_KERNEL_TYPE_TASKS_CREATE;
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, create_tasks));
  }

  if (single_pass == false) {
    __FAILURE_HANDLE(_device_renderer_queue_continue_callback(renderer, device, shared_callback_data));
  }

  if (debug_pass) {
    __FAILURE_HANDLE(_device_renderer_queue_debug_pass(renderer, device));
  }
  else {
    __FAILURE_HANDLE(_device_renderer_queue_main_pass(renderer, device));
  }

  if (collects_results) {
    const CUDAKernelType collect_results = (renderer->sample_allocation.stage_id == 0 && _device_renderer_is_first_sample(renderer))
                                             ? CUDA_KERNEL_TYPE_ACCUMULATION_COLLECT_RESULTS_FIRST_SAMPLE
                                             : CUDA_KERNEL_TYPE_ACCUMULATION_COLLECT_RESULTS;
    __FAILURE_HANDLE(device_memset(device->work_buffers->task_work_pending, 0, 0, sizeof(uint32_t), renderer->stream));
    __FAILURE_HANDLE(_device_renderer_queue_cuda_kernel(renderer, device, collect_results));
    CUDA_FAILURE_HANDLE(cuEventRecord(renderer->time_end[event_id], renderer->stream));
    __FAILURE_HANDLE(
      device_download(renderer->work_pending, device->work_buffers->task_work_pending, 0, sizeof(uint32_t), renderer->stream));
  }

  if (single_pass) {
    __FAILURE_HANDLE(_device_renderer_queue_continue_callback(renderer, device, shared_callback_data));
  }

  renderer->pass_id++;

  ////////////////////////////////////////////////////////////////////
  // Exit
  ////////////////////////////////////////////////////////////////////

  CUDA_FAILURE_HANDLE(cuEventRecord(renderer->renderer_ready, renderer->stream));
  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_main, renderer->renderer_ready, CU_EVENT_WAIT_DEFAULT));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_finish_iteration(DeviceRenderer* renderer, bool is_undersampling) {
  __CHECK_NULL_ARGUMENT(renderer);

  renderer->status_flags &= ~(DEVICE_RENDERER_STATUS_FLAG_FINISHED | DEVICE_RENDERER_STATUS_FLAG_FIRST_SAMPLE);
  renderer->status_flags |= DEVICE_RENDERER_STATUS_FLAG_READY;

  renderer->pass_id = 0;

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
    info_message("[EventIdx: %05u] ------- Kernel Times -------", event_id);

    double total_render_time = 0.0;

    for (uint32_t kernel_id = 0; kernel_id < DEVICE_RENDERER_TIMED_KERNEL_TYPES; kernel_id++) {
      DeviceRendererKernelTimer* timer = renderer->kernel_times[event_id].kernels + kernel_id;
      if (timer->name == (const char*) 0)
        continue;

      for (uint32_t slot_id = 0; slot_id < DEVICE_RENDERER_KERNEL_TIMING_SLOTS; slot_id++) {
        __FAILURE_HANDLE(_device_renderer_collect_kernel_timing(timer, slot_id));
      }

      info_message("%32s | %07.2fms (%05.2f%%)", timer->name, timer->total_ms, 100.0 * timer->total_ms / event_time);
      total_render_time += timer->total_ms;
    }

    info_message("Total Time spent in Kernels: %07.2fms (%05.2f%%)", total_render_time, 100.0 * total_render_time / event_time);
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */

    // Convert from milliseconds to seconds.
    event_time *= 0.001f;

    renderer->total_render_time[event_id] = renderer->total_render_time[(event_id - 1) & DEVICE_RENDERER_TIMING_EVENTS_MASK] + event_time;

    renderer->last_time = event_time;

    renderer->timing_event_id++;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_allocate_sample(DeviceRenderer* renderer, AdaptiveSampler* sampler) {
  __CHECK_NULL_ARGUMENT(renderer);

  if (renderer->sample_allocation.num_samples == 0)
    __FAILURE_HANDLE(adaptive_sampler_allocate_sample(sampler, &renderer->sample_allocation, 4));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_externalize_samples(
  DeviceRenderer* renderer, uint32_t num_stage_executions[ADAPTIVE_SAMPLER_NUM_STAGES + 1]) {
  __CHECK_NULL_ARGUMENT(renderer);

  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES + 1; stage_id++) {
    num_stage_executions[stage_id] = renderer->executed_aggregate_sample_counts[stage_id];
  }

  memset(renderer->executed_aggregate_sample_counts, 0, sizeof(renderer->executed_aggregate_sample_counts));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_register_external_samples(
  DeviceRenderer* renderer, uint32_t num_stage_executions[ADAPTIVE_SAMPLER_NUM_STAGES + 1]) {
  __CHECK_NULL_ARGUMENT(renderer);

  for (uint32_t stage_id = 0; stage_id < ADAPTIVE_SAMPLER_NUM_STAGES + 1; stage_id++) {
    renderer->executed_aggregate_sample_counts[stage_id] += num_stage_executions[stage_id];
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
  if (undersampling_stage || renderer->sample_allocation.stage_id == 0) {
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

static LuminaryResult _device_renderer_abort(DeviceRenderer* renderer, bool shutdown) {
  Device* device = renderer->device;

  if (device->state != DEVICE_STATE_UNAVAILABLE && device->state_abort == false) {
    __FAILURE_HANDLE(device_abort_set(device->abort, device, true));
    device->state_abort = true;
  }

  CUDA_FAILURE_HANDLE(cuEventRecord(renderer->abort_flag_uploaded, device->stream_abort));
  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(renderer->stream, renderer->abort_flag_uploaded, CU_EVENT_WAIT_DEFAULT));
  CUDA_FAILURE_HANDLE(cuEventRecord(renderer->abort_complete, renderer->stream));

  if (shutdown) {
    CUDA_FAILURE_HANDLE(cuEventSynchronize(renderer->abort_complete));
    __FAILURE_HANDLE(device_output_wait_for_completion(device->output, device->stream_main));
    CUDA_FAILURE_HANDLE(cuStreamSynchronize(device->stream_callbacks));
  }

  renderer->status_flags = DEVICE_RENDERER_STATUS_FLAG_NONE;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _device_renderer_execute_command(DeviceRenderer* renderer, const DeviceRendererCommand* command) {
  Device* device = renderer->device;
  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));

  LuminaryResult result = LUMINARY_SUCCESS;
  switch (command->type) {
    case DEVICE_RENDERER_COMMAND_START:
      result = _device_renderer_init_new_render(renderer, &command->args);
      if (result == LUMINARY_SUCCESS)
        result = _device_renderer_continue(renderer, device);
      break;
    case DEVICE_RENDERER_COMMAND_CONTINUE:
      result = _device_renderer_continue(renderer, device);
      break;
    case DEVICE_RENDERER_COMMAND_ABORT_RENDER:
    case DEVICE_RENDERER_COMMAND_SHUTDOWN:
      result = _device_renderer_abort(renderer, command->type == DEVICE_RENDERER_COMMAND_SHUTDOWN);
      break;
  }

  CUcontext context;
  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&context));

  return result;
}

static LuminaryResult _device_renderer_process_commands(DeviceRenderer* renderer) {
  for (;;) {
    DeviceRendererCommand command;
    bool received;
    __FAILURE_HANDLE(queue_pop_blocking(renderer->commands, &command, &received));
    if (received == false)
      return LUMINARY_SUCCESS;

    DeviceRendererReply reply;
    reply.type = DEVICE_RENDERER_REPLY_SUBMITTED;
    if (command.type == DEVICE_RENDERER_COMMAND_ABORT_RENDER)
      reply.type = DEVICE_RENDERER_REPLY_ABORT_SUBMITTED;
    if (command.type == DEVICE_RENDERER_COMMAND_SHUTDOWN)
      reply.type = DEVICE_RENDERER_REPLY_HAS_SHUT_DOWN;

    reply.result = _device_renderer_execute_command(renderer, &command);
    __FAILURE_HANDLE(queue_push(renderer->replies, &reply));

    if (command.type == DEVICE_RENDERER_COMMAND_SHUTDOWN)
      return LUMINARY_SUCCESS;
  }
}

static LuminaryResult _device_renderer_thread_main(void* args) {
  DeviceRenderer* renderer = args;
  error_registry_make_host_current(renderer->host_id);

  const LuminaryResult result       = _device_renderer_process_commands(renderer);
  const LuminaryResult close_result = queue_set_is_blocking(renderer->replies, false);

  return (result == LUMINARY_SUCCESS) ? close_result : result;
}

static LuminaryResult _device_renderer_wait_for_reply(DeviceRenderer* renderer, DeviceRendererReplyType expected) {
  DeviceRendererReply reply;
  bool received;
  __FAILURE_HANDLE(queue_pop_blocking(renderer->replies, &reply, &received));

  if (received == false || reply.type != expected) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Unexpected device renderer reply.");
  }

  return reply.result;
}

LuminaryResult device_renderer_start(DeviceRenderer* renderer, const DeviceRendererQueueArgs* args) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(args);

  if (renderer->shutdown)
    return LUMINARY_SUCCESS;

  DeviceRendererCommand command = {.type = DEVICE_RENDERER_COMMAND_START, .args = *args};
  __FAILURE_HANDLE(queue_push(renderer->commands, &command));

  return _device_renderer_wait_for_reply(renderer, DEVICE_RENDERER_REPLY_SUBMITTED);
}

LuminaryResult device_renderer_continue(DeviceRenderer* renderer) {
  __CHECK_NULL_ARGUMENT(renderer);

  if (renderer->shutdown)
    return LUMINARY_SUCCESS;

  DeviceRendererCommand command = {.type = DEVICE_RENDERER_COMMAND_CONTINUE};
  __FAILURE_HANDLE(queue_push(renderer->commands, &command));

  return _device_renderer_wait_for_reply(renderer, DEVICE_RENDERER_REPLY_SUBMITTED);
}

LuminaryResult device_renderer_request_abort(DeviceRenderer* renderer) {
  __CHECK_NULL_ARGUMENT(renderer);

  if (renderer->shutdown)
    return LUMINARY_SUCCESS;

  DeviceRendererCommand command = {.type = DEVICE_RENDERER_COMMAND_ABORT_RENDER};

  return queue_push(renderer->commands, &command);
}

LuminaryResult device_renderer_wait_for_abort_submission(DeviceRenderer* renderer) {
  __CHECK_NULL_ARGUMENT(renderer);

  if (renderer->shutdown)
    return LUMINARY_SUCCESS;

  __FAILURE_HANDLE(_device_renderer_wait_for_reply(renderer, DEVICE_RENDERER_REPLY_ABORT_SUBMITTED));

  Device* device = renderer->device;
  CUDA_FAILURE_HANDLE(cuCtxPushCurrent(device->cuda_ctx));
  CUDA_FAILURE_HANDLE(cuStreamWaitEvent(device->stream_main, renderer->abort_complete, CU_EVENT_WAIT_DEFAULT));
  __FAILURE_HANDLE(device_output_wait_for_completion(device->output, device->stream_main));

  CUcontext context;
  CUDA_FAILURE_HANDLE(cuCtxPopCurrent(&context));

  return LUMINARY_SUCCESS;
}

LuminaryResult device_renderer_shutdown(DeviceRenderer* renderer) {
  __CHECK_NULL_ARGUMENT(renderer);

  if (renderer->shutdown)
    return LUMINARY_SUCCESS;

  DeviceRendererCommand command = {.type = DEVICE_RENDERER_COMMAND_SHUTDOWN};
  __FAILURE_HANDLE(queue_push(renderer->commands, &command));
  const LuminaryResult result = _device_renderer_wait_for_reply(renderer, DEVICE_RENDERER_REPLY_HAS_SHUT_DOWN);
  __FAILURE_HANDLE(thread_join(renderer->thread));
  renderer->shutdown = true;
  __FAILURE_HANDLE(thread_get_last_result(renderer->thread));

  return result;
}

LuminaryResult device_renderer_destroy(DeviceRenderer** renderer) {
  __CHECK_NULL_ARGUMENT(renderer);
  __CHECK_NULL_ARGUMENT(*renderer);

  __FAILURE_HANDLE(device_renderer_shutdown(*renderer));
  __FAILURE_HANDLE(thread_destroy(&(*renderer)->thread));
  __FAILURE_HANDLE(queue_destroy(&(*renderer)->commands));
  __FAILURE_HANDLE(queue_destroy(&(*renderer)->replies));

  CUDA_FAILURE_HANDLE(cuStreamDestroy((*renderer)->stream));
  CUDA_FAILURE_HANDLE(cuEventDestroy((*renderer)->manager_ready));
  CUDA_FAILURE_HANDLE(cuEventDestroy((*renderer)->renderer_ready));
  CUDA_FAILURE_HANDLE(cuEventDestroy((*renderer)->abort_flag_uploaded));
  CUDA_FAILURE_HANDLE(cuEventDestroy((*renderer)->abort_complete));
  __FAILURE_HANDLE(device_free_staging(&(*renderer)->work_pending));

  for (uint32_t event_id = 0; event_id < DEVICE_RENDERER_TIMING_EVENTS_COUNT; event_id++) {
    __FAILURE_HANDLE(cuEventDestroy((*renderer)->time_start[event_id]));
    __FAILURE_HANDLE(cuEventDestroy((*renderer)->time_end[event_id]));

#ifdef DEVICE_RENDERER_DO_PER_KERNEL_TIMING
    for (uint32_t kernel_id = 0; kernel_id < DEVICE_RENDERER_TIMED_KERNEL_TYPES; kernel_id++) {
      DeviceRendererKernelTimer* timer = (*renderer)->kernel_times[event_id].kernels + kernel_id;
      for (uint32_t slot_id = 0; slot_id < DEVICE_RENDERER_KERNEL_TIMING_SLOTS; slot_id++) {
        DeviceRendererKernelTimingSlot* slot = timer->slots + slot_id;
        if (slot->start != 0) {
          CUDA_FAILURE_HANDLE(cuEventDestroy(slot->start));
          CUDA_FAILURE_HANDLE(cuEventDestroy(slot->end));
        }
      }
    }
#endif /* DEVICE_RENDERER_DO_PER_KERNEL_TIMING */
  }

  __FAILURE_HANDLE(host_free(renderer));

  return LUMINARY_SUCCESS;
}
