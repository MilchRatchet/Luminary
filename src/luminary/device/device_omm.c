#include "device_omm.h"

#include "device.h"
#include "internal_error.h"
#include "kernel_args.h"

// OMMs should not occupy too much memory
#define MAX_MEMORY_USAGE 100000000ul

#define OMM_STATE_SIZE(__level__, __format__) \
  (((1u << (__level__ * 2u)) * ((__format__ == OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE) ? 1u : 2u) + 7u) / 8u)

LuminaryResult omm_create(OpacityMicromap** omm) {
  __CHECK_NULL_ARGUMENT(omm);

  __FAILURE_HANDLE(host_malloc(omm, sizeof(OpacityMicromap)));
  memset(*omm, 0, sizeof(OpacityMicromap));

  return LUMINARY_SUCCESS;
}

static size_t _omm_array_size(const uint32_t count, const uint32_t level, const OptixOpacityMicromapFormat format) {
  if (format != OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE && format != OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE) {
    return 0;
  }

  // OMMs are byte aligned, hence even the low subdivision levels are at least 1 byte in size
  const uint32_t state_size = OMM_STATE_SIZE(level, format);

  return state_size * count;
}

LuminaryResult omm_build(OpacityMicromap* omm, const Mesh* mesh, Device* device) {
  __CHECK_NULL_ARGUMENT(omm);
  __CHECK_NULL_ARGUMENT(mesh);
  __CHECK_NULL_ARGUMENT(device);

  const OptixOpacityMicromapFormat format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
  const uint32_t total_tri_count          = mesh->data.triangle_count;

  // Highest allowed level is 12 according to OptiX Docs
  const uint8_t max_num_levels = 6;
  uint8_t num_levels           = 0;

  ////////////////////////////////////////////////////////////////////
  // OMM construction
  ////////////////////////////////////////////////////////////////////

  uint32_t* triangles_per_level;
  __FAILURE_HANDLE(host_malloc(&triangles_per_level, sizeof(uint32_t) * max_num_levels));
  memset(triangles_per_level, 0, sizeof(uint32_t) * max_num_levels);

  void** data;
  __FAILURE_HANDLE(host_malloc(&data, sizeof(DEVICE void*) * max_num_levels));

  // For each triangle, we store the final level, 0xFF specifies that the triangle has not reached its final level yet
  uint8_t* triangle_level;
  __FAILURE_HANDLE(host_malloc(&triangle_level, total_tri_count));

  for (uint32_t i = 0; i < total_tri_count; i++) {
    triangle_level[i] = 0xFF;
  }

  DEVICE void* triangle_level_buffer;
  __FAILURE_HANDLE(device_malloc(&triangle_level_buffer, total_tri_count));

  __FAILURE_HANDLE(device_upload(triangle_level_buffer, triangle_level, 0, total_tri_count, device->stream_main));

  size_t memory_usage = 0;

  uint32_t remaining_triangles = 0;
  for (; num_levels < max_num_levels;) {
    const size_t data_size = _omm_array_size(total_tri_count, num_levels, format);
    __FAILURE_HANDLE(device_malloc(data + num_levels, data_size));

    if (num_levels) {
      KernelArgsOMMRefineFormat4 args = {
        .mesh_id        = mesh->id,
        .triangle_count = total_tri_count,
        .dst            = (uint8_t*) DEVICE_PTR(data[num_levels]),
        .src            = (const uint8_t*) DEVICE_PTR(data[num_levels - 1]),
        .level_record   = (uint8_t*) DEVICE_PTR(triangle_level_buffer),
        .src_level      = num_levels - 1,
      };

      __FAILURE_HANDLE(kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_OMM_REFINE_FORMAT_4], &args, device->stream_main));
    }
    else {
      // Make sure that all the data is actually present
      __FAILURE_HANDLE(device_sync_constant_memory(device));
      __FAILURE_HANDLE(device_staging_manager_execute(device->staging_manager));

      KernelArgsOMMLevel0Format4 args = {
        .mesh_id        = mesh->id,
        .triangle_count = total_tri_count,
        .dst            = (uint8_t*) DEVICE_PTR(data[0]),
        .level_record   = (uint8_t*) DEVICE_PTR(triangle_level_buffer),
      };

      __FAILURE_HANDLE(kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_OMM_LEVEL_0_FORMAT_4], &args, device->stream_main));
    }

    __FAILURE_HANDLE(device_download(triangle_level, triangle_level_buffer, 0, total_tri_count, device->stream_main));

    remaining_triangles         = 0;
    uint32_t triangles_at_level = 0;

    for (uint32_t i = 0; i < total_tri_count; i++) {
      if (triangle_level[i] == 0xFF)
        remaining_triangles++;

      if (triangle_level[i] == num_levels)
        triangles_at_level++;
    }

    log_message("[OptiX OMM] Remaining triangles after %u iterations: %u.", num_levels, remaining_triangles);

    triangles_per_level[num_levels] = triangles_at_level;

    memory_usage += triangles_at_level * OMM_STATE_SIZE(num_levels, format);

    num_levels++;

    if (!remaining_triangles) {
      break;
    }

    if (memory_usage + remaining_triangles * OMM_STATE_SIZE(num_levels, format) > MAX_MEMORY_USAGE) {
      log_message("[OptiX OMM] Exceeded memory budget at subdivision level %u.", num_levels);
      break;
    }
  }

  // Some triangles needed more refinement but max level was reached.
  if (remaining_triangles) {
    triangles_per_level[num_levels - 1] += remaining_triangles;

    for (uint32_t i = 0; i < total_tri_count; i++) {
      if (triangle_level[i] == 0xFF) {
        triangle_level[i] = num_levels - 1;
      }
    }

    __FAILURE_HANDLE(device_upload(triangle_level_buffer, triangle_level, 0, total_tri_count, device->stream_main));
  }

  size_t final_array_size = 0;
  size_t* array_offset_per_level;
  __FAILURE_HANDLE(host_malloc(&array_offset_per_level, sizeof(size_t) * num_levels));

  size_t* array_size_per_level;
  __FAILURE_HANDLE(host_malloc(&array_size_per_level, sizeof(size_t) * num_levels));

  for (uint32_t i = 0; i < num_levels; i++) {
    const size_t state_size = OMM_STATE_SIZE(i, format);

    log_message("[OptiX OMM] Total triangles at subdivision level %u: %u.", i, triangles_per_level[i]);

    array_size_per_level[i]   = state_size * triangles_per_level[i];
    array_offset_per_level[i] = (i) ? array_offset_per_level[i - 1] + array_size_per_level[i - 1] : 0;

    final_array_size += array_size_per_level[i];
  }

  DEVICE void* omm_array;
  __FAILURE_HANDLE(device_malloc(&omm_array, final_array_size));

  ////////////////////////////////////////////////////////////////////
  // Description setup
  ////////////////////////////////////////////////////////////////////

  OptixOpacityMicromapDesc* desc;
  __FAILURE_HANDLE(host_malloc(&desc, sizeof(OptixOpacityMicromapDesc) * total_tri_count));

  for (uint32_t i = 0; i < total_tri_count; i++) {
    const uint32_t level = (triangle_level[i] == 0xFF) ? max_num_levels - 1 : triangle_level[i];

    desc[i].byteOffset       = (unsigned int) array_offset_per_level[level];
    desc[i].subdivisionLevel = (unsigned short) level;
    desc[i].format           = format;

    const size_t state_size = OMM_STATE_SIZE(level, format);

    array_offset_per_level[level] += state_size;
  }

  __FAILURE_HANDLE(host_free(&array_offset_per_level));
  __FAILURE_HANDLE(host_free(&array_size_per_level));

  DEVICE void* desc_buffer;
  __FAILURE_HANDLE(device_malloc(&desc_buffer, sizeof(OptixOpacityMicromapDesc) * total_tri_count));
  __FAILURE_HANDLE(device_upload(desc_buffer, desc, 0, sizeof(OptixOpacityMicromapDesc) * total_tri_count, device->stream_main));

  for (uint32_t i = 0; i < num_levels; i++) {
    KernelArgsOMMGatherArrayFormat4 args = {
      .triangle_count = total_tri_count,
      .dst            = (uint8_t*) DEVICE_PTR(omm_array),
      .src            = (const uint8_t*) DEVICE_PTR(data[i]),
      .level          = i,
      .level_record   = (const uint8_t*) DEVICE_PTR(triangle_level_buffer),
      .desc           = (const OptixOpacityMicromapDesc*) DEVICE_PTR(desc_buffer),
    };

    __FAILURE_HANDLE(
      kernel_execute_with_args(device->cuda_kernels[CUDA_KERNEL_TYPE_OMM_GATHER_ARRAY_FORMAT_4], &args, device->stream_main));
  }

  __FAILURE_HANDLE(host_free(&desc));

  __FAILURE_HANDLE(host_free(&triangle_level));
  __FAILURE_HANDLE(device_free(&triangle_level_buffer));

  for (uint32_t i = 0; i < num_levels; i++) {
    __FAILURE_HANDLE(device_free(&data[i]));
  }

  __FAILURE_HANDLE(host_free(&data));

  ////////////////////////////////////////////////////////////////////
  // Histogram setup
  ////////////////////////////////////////////////////////////////////

  OptixOpacityMicromapHistogramEntry* histogram;
  __FAILURE_HANDLE(host_malloc(&histogram, sizeof(OptixOpacityMicromapHistogramEntry) * num_levels));

  for (uint32_t i = 0; i < num_levels; i++) {
    histogram[i].count            = triangles_per_level[i];
    histogram[i].subdivisionLevel = i;
    histogram[i].format           = format;
  }

  ////////////////////////////////////////////////////////////////////
  // Usage count setup
  ////////////////////////////////////////////////////////////////////

  OptixOpacityMicromapUsageCount* usage;
  __FAILURE_HANDLE(host_malloc(&usage, sizeof(OptixOpacityMicromapUsageCount) * num_levels));

  for (uint32_t i = 0; i < num_levels; i++) {
    usage[i].count            = triangles_per_level[i];
    usage[i].subdivisionLevel = i;
    usage[i].format           = format;
  }

  __FAILURE_HANDLE(host_free(&triangles_per_level));

  ////////////////////////////////////////////////////////////////////
  // OMM array construction
  ////////////////////////////////////////////////////////////////////

  OptixOpacityMicromapArrayBuildInput array_build_input;
  memset(&array_build_input, 0, sizeof(OptixOpacityMicromapArrayBuildInput));

  array_build_input.flags                        = OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE;
  array_build_input.inputBuffer                  = DEVICE_CUPTR(omm_array);
  array_build_input.numMicromapHistogramEntries  = num_levels;
  array_build_input.micromapHistogramEntries     = histogram;
  array_build_input.perMicromapDescBuffer        = DEVICE_CUPTR(desc_buffer);
  array_build_input.perMicromapDescStrideInBytes = sizeof(OptixOpacityMicromapDesc);

  OptixMicromapBufferSizes buffer_sizes;
  memset(&buffer_sizes, 0, sizeof(OptixMicromapBufferSizes));

  OPTIX_FAILURE_HANDLE(optixOpacityMicromapArrayComputeMemoryUsage(device->optix_ctx, &array_build_input, &buffer_sizes));

  __FAILURE_HANDLE(device_malloc(&omm->buffer, buffer_sizes.outputSizeInBytes));
  DEVICE void* temp_buffer;
  __FAILURE_HANDLE(device_malloc(&temp_buffer, buffer_sizes.tempSizeInBytes));

  OptixMicromapBuffers buffers;
  memset(&buffers, 0, sizeof(OptixMicromapBuffers));

  buffers.output            = DEVICE_CUPTR(omm->buffer);
  buffers.outputSizeInBytes = buffer_sizes.outputSizeInBytes;
  buffers.temp              = DEVICE_CUPTR(temp_buffer);
  buffers.tempSizeInBytes   = buffer_sizes.tempSizeInBytes;

  OPTIX_FAILURE_HANDLE(optixOpacityMicromapArrayBuild(device->optix_ctx, 0, &array_build_input, &buffers));

  __FAILURE_HANDLE(device_free(&desc_buffer));
  __FAILURE_HANDLE(device_free(&temp_buffer));
  __FAILURE_HANDLE(device_free(&omm_array));
  __FAILURE_HANDLE(host_free(&histogram));

  ////////////////////////////////////////////////////////////////////
  // BVH input construction
  ////////////////////////////////////////////////////////////////////

  OptixBuildInputOpacityMicromap bvh_input;
  memset(&bvh_input, 0, sizeof(OptixBuildInputOpacityMicromap));

  bvh_input.indexingMode           = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR;
  bvh_input.opacityMicromapArray   = DEVICE_CUPTR(omm->buffer);
  bvh_input.numMicromapUsageCounts = num_levels;
  bvh_input.micromapUsageCounts    = usage;

  omm->optix_build_input = bvh_input;

  return LUMINARY_SUCCESS;
}

LuminaryResult omm_destroy(OpacityMicromap** omm) {
  __CHECK_NULL_ARGUMENT(omm);

  if ((*omm)->buffer) {
    __FAILURE_HANDLE(device_free(&(*omm)->buffer));
  }

  if ((*omm)->optix_build_input.micromapUsageCounts) {
    __FAILURE_HANDLE(host_free(&(*omm)->optix_build_input.micromapUsageCounts));
  }

  __FAILURE_HANDLE(host_free(omm));

  return LUMINARY_SUCCESS;
}
