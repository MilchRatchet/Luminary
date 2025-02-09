// Functions work differently when executed from this kernel
// This emulates the old device.iteration_type == TYPE_LIGHT checks.
#define SHADING_KERNEL
#define OPTIX_KERNEL

#include "bsdf.cuh"
#include "directives.cuh"
#include "geometry_utils.cuh"
#include "ior_stack.cuh"
#include "math.cuh"
#include "memory.cuh"
#include "shading_kernel.cuh"
#include "utils.cuh"

extern "C" __global__ void __raygen__optix() {
  HANDLE_DEVICE_ABORT();

  const int task_count  = device.ptrs.task_counts[THREAD_ID * TASK_ADDRESS_COUNT_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  const int task_offset = device.ptrs.task_offsets[THREAD_ID * TASK_ADDRESS_OFFSET_STRIDE + TASK_ADDRESS_OFFSET_GEOMETRY];
  int trace_count       = device.ptrs.trace_counts[THREAD_ID];

  for (int i = 0; i < task_count; i++) {
    const uint32_t offset                = get_task_address(task_offset + i);
    DeviceTask task                      = task_load(offset);
    const TriangleHandle triangle_handle = triangle_handle_load(offset);
    const float depth                    = trace_depth_load(offset);
    const uint32_t pixel                 = get_pixel_id(task.index);

    task.origin = add_vector(task.origin, scale_vector(task.ray, depth));

    GBufferData data;
    if (triangle_handle.instance_id == HIT_TYPE_OCEAN) {
      data = ocean_generate_g_buffer(task, pixel);
    }
    else {
      data = geometry_generate_g_buffer(task, triangle_handle, pixel);
    }

    ////////////////////////////////////////////////////////////////////
    // Bounce Ray Sampling
    ////////////////////////////////////////////////////////////////////

    BSDFSampleInfo bounce_info;
    vec3 bounce_ray = bsdf_sample(data, task.index, bounce_info);

    ////////////////////////////////////////////////////////////////////
    // Update delta path state
    ////////////////////////////////////////////////////////////////////

    bool is_delta_distribution;
    if (bounce_info.is_transparent_pass) {
      const float refraction_scale = (data.ior_in > data.ior_out) ? data.ior_in / data.ior_out : data.ior_out / data.ior_in;
      is_delta_distribution        = data.roughness * fminf(refraction_scale - 1.0f, 1.0f) <= GEOMETRY_DELTA_PATH_CUTOFF;
    }
    else {
      is_delta_distribution = bounce_info.is_microfacet_based && (data.roughness <= GEOMETRY_DELTA_PATH_CUTOFF);
    }

    const bool is_pass_through = bsdf_is_pass_through_ray(bounce_info.is_transparent_pass, data.ior_in, data.ior_out);

    ////////////////////////////////////////////////////////////////////
    // Light Ray Sampling
    ////////////////////////////////////////////////////////////////////

    RGBF accumulated_light = data.emission;

    if (data.flags & G_BUFFER_FLAG_USE_LIGHT_RAYS) {
      accumulated_light = add_color(accumulated_light, optix_compute_light_ray_sun(data, task.index));
      accumulated_light = add_color(accumulated_light, optix_compute_light_ray_geo(data, task.index));
    }

    accumulated_light = add_color(
      accumulated_light,
      optix_compute_light_ray_ambient_sky(data, bounce_ray, bounce_info.weight, bounce_info.is_transparent_pass, task.index));

    const RGBF record = load_RGBF(device.ptrs.records + pixel);

    accumulated_light = mul_color(accumulated_light, record);

    write_beauty_buffer(accumulated_light, pixel, task.state);

    if (bounce_info.is_transparent_pass) {
      const IORStackMethod ior_stack_method =
        (data.flags & G_BUFFER_FLAG_REFRACTION_IS_INSIDE) ? IOR_STACK_METHOD_PULL : IOR_STACK_METHOD_PUSH;
      ior_stack_interact(data.ior_out, pixel, ior_stack_method);
    }

    data.position = shift_origin_vector(data.position, data.V, bounce_ray, bounce_info.is_transparent_pass);

    uint16_t new_state = task.state;

    if (!is_delta_distribution) {
      new_state &= ~STATE_FLAG_DELTA_PATH;
    }

    if (!is_pass_through) {
      new_state &= ~STATE_FLAG_CAMERA_DIRECTION;
    }

    DeviceTask bounce_task;
    bounce_task.state  = new_state;
    bounce_task.origin = data.position;
    bounce_task.ray    = bounce_ray;
    bounce_task.index  = task.index;

    RGBF bounce_record = mul_color(record, bounce_info.weight);

    // This must be done after the trace rays due to some optimization in the compiler.
    // The compiler reloads these values at some point for some reason and if we overwrite
    // the values we will get garbage. I am not sure if this is a compiler bug or some undefined
    // behaviour on my side.
    if (task_russian_roulette(bounce_task, task.state, bounce_record)) {
      task_store(bounce_task, get_task_address(trace_count++));
      store_RGBF(device.ptrs.records, pixel, bounce_record);
    }
  }

  device.ptrs.trace_counts[THREAD_ID] = trace_count;
}
