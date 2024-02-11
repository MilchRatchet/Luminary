#ifndef CU_MICROMAP_H
#define CU_MICROMAP_H

#include "buffer.h"
#include "device.h"
#include "memory.cuh"
#include "micromap_utils.cuh"
#include "utils.cuh"

static size_t _omm_array_size(const uint32_t count, const uint32_t level, const OptixOpacityMicromapFormat format) {
  if (format != OPTIX_OPACITY_MICROMAP_FORMAT_2_STATE && format != OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE) {
    return 0;
  }

  // OMMs are byte aligned, hence even the low subdivision levels are at least 1 byte in size
  const uint32_t state_size = OMM_STATE_SIZE(level, format);

  return state_size * count;
}

OptixBuildInputOpacityMicromap micromap_opacity_build(RaytraceInstance* instance) {
  const OptixOpacityMicromapFormat format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
  const uint32_t total_tri_count          = instance->scene.triangle_data.triangle_count;

  // Highest allowed level is 12 according to OptiX Docs
  const uint8_t max_num_levels = 6;
  uint8_t num_levels           = 0;

  ////////////////////////////////////////////////////////////////////
  // OMM construction
  ////////////////////////////////////////////////////////////////////

  uint32_t* triangles_per_level = (uint32_t*) calloc(1, sizeof(uint32_t) * max_num_levels);
  void** data                   = (void**) malloc(sizeof(void*) * max_num_levels);

  // For each triangle, we store the final level, 0xFF specifies that the triangle has not reached its final level yet
  uint8_t* triangle_level = (uint8_t*) malloc(total_tri_count);
  void* triangle_level_buffer;
  device_malloc(&triangle_level_buffer, total_tri_count);

  for (uint32_t i = 0; i < total_tri_count; i++) {
    triangle_level[i] = 0xFF;
  }
  device_upload(triangle_level_buffer, triangle_level, total_tri_count);

  size_t memory_usage = 0;

  uint32_t remaining_triangles = 0;
  for (; num_levels < max_num_levels;) {
    const size_t data_size = _omm_array_size(total_tri_count, num_levels, format);
    device_malloc(data + num_levels, data_size);
    gpuErrchk(cudaDeviceSynchronize());

    if (num_levels) {
      omm_refine_format_4<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
        (uint8_t*) data[num_levels], (uint8_t*) data[num_levels - 1], (uint8_t*) triangle_level_buffer, num_levels - 1);
    }
    else {
      omm_level_0_format_4<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((uint8_t*) data[0], (uint8_t*) triangle_level_buffer);
    }
    gpuErrchk(cudaDeviceSynchronize());

    device_download(triangle_level, triangle_level_buffer, total_tri_count);
    gpuErrchk(cudaDeviceSynchronize());

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

    device_upload(triangle_level_buffer, triangle_level, total_tri_count);
  }

  size_t final_array_size        = 0;
  size_t* array_offset_per_level = (size_t*) malloc(sizeof(size_t) * num_levels);
  size_t* array_size_per_level   = (size_t*) malloc(sizeof(size_t) * num_levels);
  for (uint32_t i = 0; i < num_levels; i++) {
    const size_t state_size = OMM_STATE_SIZE(i, format);

    log_message("[OptiX OMM] Total triangles at subdivision level %u: %u.", i, triangles_per_level[i]);

    array_size_per_level[i]   = state_size * triangles_per_level[i];
    array_offset_per_level[i] = (i) ? array_offset_per_level[i - 1] + array_size_per_level[i - 1] : 0;

    final_array_size += array_size_per_level[i];
  }

  void* omm_array;
  device_malloc(&omm_array, final_array_size);

  ////////////////////////////////////////////////////////////////////
  // Description setup
  ////////////////////////////////////////////////////////////////////

  OptixOpacityMicromapDesc* desc = (OptixOpacityMicromapDesc*) malloc(sizeof(OptixOpacityMicromapDesc) * total_tri_count);

  for (uint32_t i = 0; i < total_tri_count; i++) {
    const uint32_t level = (triangle_level[i] == 0xFF) ? max_num_levels - 1 : triangle_level[i];

    desc[i].byteOffset       = array_offset_per_level[level];
    desc[i].subdivisionLevel = level;
    desc[i].format           = format;

    const size_t state_size = OMM_STATE_SIZE(level, format);

    array_offset_per_level[level] += state_size;
  }

  free(array_offset_per_level);
  free(array_size_per_level);

  void* desc_buffer;
  device_malloc(&desc_buffer, sizeof(OptixOpacityMicromapDesc) * total_tri_count);
  device_upload(desc_buffer, desc, sizeof(OptixOpacityMicromapDesc) * total_tri_count);

  for (uint32_t i = 0; i < num_levels; i++) {
    omm_gather_array_format_4<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
      (uint8_t*) omm_array, (uint8_t*) data[i], i, (uint8_t*) triangle_level_buffer, (OptixOpacityMicromapDesc*) desc_buffer);
  }
  gpuErrchk(cudaDeviceSynchronize());

  free(triangle_level);
  device_free(triangle_level_buffer, total_tri_count);

  for (uint32_t i = 0; i < num_levels; i++) {
    const size_t data_size = _omm_array_size(total_tri_count, i, format);
    device_free(data[i], data_size);
  }

  free(data);

  ////////////////////////////////////////////////////////////////////
  // Histogram setup
  ////////////////////////////////////////////////////////////////////

  OptixOpacityMicromapHistogramEntry* histogram =
    (OptixOpacityMicromapHistogramEntry*) malloc(sizeof(OptixOpacityMicromapHistogramEntry) * num_levels);

  for (uint32_t i = 0; i < num_levels; i++) {
    histogram[i].count            = triangles_per_level[i];
    histogram[i].subdivisionLevel = i;
    histogram[i].format           = format;
  }

  ////////////////////////////////////////////////////////////////////
  // Usage count setup
  ////////////////////////////////////////////////////////////////////

  OptixOpacityMicromapUsageCount* usage = (OptixOpacityMicromapUsageCount*) malloc(sizeof(OptixOpacityMicromapUsageCount) * num_levels);
  for (uint32_t i = 0; i < num_levels; i++) {
    usage[i].count            = triangles_per_level[i];
    usage[i].subdivisionLevel = i;
    usage[i].format           = format;
  }

  free(triangles_per_level);

  ////////////////////////////////////////////////////////////////////
  // DMM array construction
  ////////////////////////////////////////////////////////////////////

  OptixOpacityMicromapArrayBuildInput array_build_input;
  memset(&array_build_input, 0, sizeof(OptixOpacityMicromapArrayBuildInput));

  array_build_input.flags                        = OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE;
  array_build_input.inputBuffer                  = (CUdeviceptr) omm_array;
  array_build_input.numMicromapHistogramEntries  = num_levels;
  array_build_input.micromapHistogramEntries     = histogram;
  array_build_input.perMicromapDescBuffer        = (CUdeviceptr) desc_buffer;
  array_build_input.perMicromapDescStrideInBytes = sizeof(OptixOpacityMicromapDesc);

  OptixMicromapBufferSizes buffer_sizes;
  memset(&buffer_sizes, 0, sizeof(OptixMicromapBufferSizes));

  OPTIX_CHECK(optixOpacityMicromapArrayComputeMemoryUsage(instance->optix_ctx, &array_build_input, &buffer_sizes));

  void* output_buffer;
  device_malloc(&output_buffer, buffer_sizes.outputSizeInBytes);
  void* temp_buffer;
  device_malloc(&temp_buffer, buffer_sizes.tempSizeInBytes);

  OptixMicromapBuffers buffers;
  memset(&buffers, 0, sizeof(OptixMicromapBuffers));

  buffers.output            = (CUdeviceptr) output_buffer;
  buffers.outputSizeInBytes = buffer_sizes.outputSizeInBytes;
  buffers.temp              = (CUdeviceptr) temp_buffer;
  buffers.tempSizeInBytes   = buffer_sizes.tempSizeInBytes;

  OPTIX_CHECK(optixOpacityMicromapArrayBuild(instance->optix_ctx, 0, &array_build_input, &buffers));

  device_free(desc_buffer, sizeof(OptixOpacityMicromapDesc) * num_levels);
  device_free(temp_buffer, buffer_sizes.tempSizeInBytes);
  device_free(omm_array, final_array_size);

  ////////////////////////////////////////////////////////////////////
  // BVH input construction
  ////////////////////////////////////////////////////////////////////

  OptixBuildInputOpacityMicromap bvh_input;
  memset(&bvh_input, 0, sizeof(OptixBuildInputOpacityMicromap));

  bvh_input.indexingMode           = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR;
  bvh_input.opacityMicromapArray   = (CUdeviceptr) output_buffer;
  bvh_input.numMicromapUsageCounts = num_levels;
  bvh_input.micromapUsageCounts    = usage;

  return bvh_input;
}

void micromap_opacity_free(OptixBuildInputOpacityMicromap data) {
  free((void*) data.micromapUsageCounts);
}

OptixBuildInputDisplacementMicromap micromap_displacement_build(RaytraceInstance* instance) {
  // Initial implementation only supports the basic one uncompressed block as a DMM
  // It is planned to at least support uncompressed blocks that sum to 1024 microtriangles in the future
  // level  = 3
  // format = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES

  const uint32_t total_tri_count = instance->scene.triangle_data.triangle_count;

  ////////////////////////////////////////////////////////////////////
  // Index computation
  ////////////////////////////////////////////////////////////////////

  void* indices_buffer;
  device_malloc(&indices_buffer, total_tri_count * sizeof(uint32_t));

  dmm_precompute_indices<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((uint32_t*) indices_buffer);
  gpuErrchk(cudaDeviceSynchronize());

  uint32_t* indices = (uint32_t*) malloc(sizeof(uint32_t) * total_tri_count);
  uint32_t* mapping = (uint32_t*) malloc(sizeof(uint32_t) * total_tri_count);

  device_download(indices, indices_buffer, total_tri_count * sizeof(uint32_t));
  gpuErrchk(cudaDeviceSynchronize());

  mapping[0] = DMM_NONE;

  uint32_t dmm_count = 1;
  for (uint32_t i = 0; i < total_tri_count; i++) {
    if (indices[i]) {
      indices[i]         = dmm_count;
      mapping[dmm_count] = i;
      dmm_count++;
    }
  }

  if (dmm_count == 1) {
    log_message("[Optix DMM] No displacement maps exist. No DMM was built.");

    OptixBuildInputDisplacementMicromap empty_result;
    memset(&empty_result, 0, sizeof(OptixBuildInputDisplacementMicromap));

    free(indices);
    free(mapping);
    device_free(indices_buffer, total_tri_count * sizeof(uint32_t));

    return empty_result;
  }

  device_upload(indices_buffer, indices, total_tri_count * sizeof(uint32_t));
  gpuErrchk(cudaDeviceSynchronize());
  free(indices);

  void* mapping_buffer;
  device_malloc(&mapping_buffer, dmm_count * sizeof(uint32_t));
  device_upload(mapping_buffer, mapping, dmm_count * sizeof(uint32_t));
  gpuErrchk(cudaDeviceSynchronize());
  free(mapping);

  ////////////////////////////////////////////////////////////////////
  // Vertex direction computation
  ////////////////////////////////////////////////////////////////////

  void* vertex_direction_buffer;
  device_malloc(&vertex_direction_buffer, total_tri_count * sizeof(half) * 9);

  dmm_setup_vertex_directions<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((half*) vertex_direction_buffer);
  gpuErrchk(cudaDeviceSynchronize());

  ////////////////////////////////////////////////////////////////////
  // Usage count setup
  ////////////////////////////////////////////////////////////////////

  OptixDisplacementMicromapUsageCount* usage =
    (OptixDisplacementMicromapUsageCount*) malloc(sizeof(OptixDisplacementMicromapUsageCount) * 2);

  usage[0].count            = total_tri_count - (dmm_count - 1);
  usage[0].format           = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
  usage[0].subdivisionLevel = 0;

  usage[1].count            = dmm_count - 1;
  usage[1].format           = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
  usage[1].subdivisionLevel = 3;

  ////////////////////////////////////////////////////////////////////
  // Description setup
  ////////////////////////////////////////////////////////////////////

  OptixDisplacementMicromapDesc* desc = (OptixDisplacementMicromapDesc*) malloc(sizeof(OptixDisplacementMicromapDesc) * dmm_count);
  for (uint32_t i = 0; i < dmm_count; i++) {
    desc[i].byteOffset       = i * 64;
    desc[i].format           = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
    desc[i].subdivisionLevel = (i) ? 3 : 0;
  }

  void* desc_buffer;
  device_malloc(&desc_buffer, sizeof(OptixDisplacementMicromapDesc) * dmm_count);
  device_upload(desc_buffer, desc, sizeof(OptixDisplacementMicromapDesc) * dmm_count);
  gpuErrchk(cudaDeviceSynchronize());

  free(desc);

  ////////////////////////////////////////////////////////////////////
  // Histogram setup
  ////////////////////////////////////////////////////////////////////

  OptixDisplacementMicromapHistogramEntry* histogram =
    (OptixDisplacementMicromapHistogramEntry*) malloc(sizeof(OptixDisplacementMicromapHistogramEntry) * 2);

  histogram[0].count            = 1;
  histogram[0].format           = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
  histogram[0].subdivisionLevel = 0;

  histogram[1].count            = dmm_count - 1;
  histogram[1].format           = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
  histogram[1].subdivisionLevel = 3;

  ////////////////////////////////////////////////////////////////////
  // DMM construction
  ////////////////////////////////////////////////////////////////////

  const size_t dmm_data_size = 64 * dmm_count;

  void* data;
  device_malloc(&data, dmm_data_size);

  dmm_build_level_3_format_64<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>((uint8_t*) data, (uint32_t*) mapping_buffer, dmm_count);
  gpuErrchk(cudaDeviceSynchronize());

  device_free(mapping_buffer, dmm_count * sizeof(uint32_t));

  ////////////////////////////////////////////////////////////////////
  // DMM array construction
  ////////////////////////////////////////////////////////////////////

  OptixDisplacementMicromapArrayBuildInput array_build_input;
  memset(&array_build_input, 0, sizeof(OptixDisplacementMicromapArrayBuildInput));

  array_build_input.flags = OPTIX_DISPLACEMENT_MICROMAP_FLAG_PREFER_FAST_TRACE;

  array_build_input.displacementMicromapHistogramEntries    = histogram;
  array_build_input.numDisplacementMicromapHistogramEntries = 2;

  array_build_input.perDisplacementMicromapDescBuffer        = (CUdeviceptr) desc_buffer;
  array_build_input.perDisplacementMicromapDescStrideInBytes = 0;

  array_build_input.displacementValuesBuffer = (CUdeviceptr) data;

  OptixMicromapBufferSizes buffer_sizes;
  memset(&buffer_sizes, 0, sizeof(OptixMicromapBufferSizes));

  OPTIX_CHECK(optixDisplacementMicromapArrayComputeMemoryUsage(instance->optix_ctx, &array_build_input, &buffer_sizes));
  gpuErrchk(cudaDeviceSynchronize());

  void* output_buffer;
  device_malloc(&output_buffer, buffer_sizes.outputSizeInBytes);
  void* temp_buffer;
  device_malloc(&temp_buffer, buffer_sizes.tempSizeInBytes);

  OptixMicromapBuffers buffers;
  memset(&buffers, 0, sizeof(OptixMicromapBuffers));

  buffers.output            = (CUdeviceptr) output_buffer;
  buffers.outputSizeInBytes = buffer_sizes.outputSizeInBytes;
  buffers.temp              = (CUdeviceptr) temp_buffer;
  buffers.tempSizeInBytes   = buffer_sizes.tempSizeInBytes;

  OPTIX_CHECK(optixDisplacementMicromapArrayBuild(instance->optix_ctx, 0, &array_build_input, &buffers));
  gpuErrchk(cudaDeviceSynchronize());

  device_free(temp_buffer, buffer_sizes.tempSizeInBytes);
  device_free(data, dmm_data_size);
  free(histogram);

  ////////////////////////////////////////////////////////////////////
  // BVH input construction
  ////////////////////////////////////////////////////////////////////

  OptixBuildInputDisplacementMicromap bvh_input;
  memset(&bvh_input, 0, sizeof(OptixBuildInputDisplacementMicromap));

  bvh_input.indexingMode = OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;

  bvh_input.displacementMicromapUsageCounts    = usage;
  bvh_input.numDisplacementMicromapUsageCounts = 2;

  bvh_input.displacementMicromapArray = (CUdeviceptr) output_buffer;

  bvh_input.displacementMicromapIndexBuffer        = (CUdeviceptr) indices_buffer;
  bvh_input.displacementMicromapIndexSizeInBytes   = 4;
  bvh_input.displacementMicromapIndexStrideInBytes = 0;
  bvh_input.displacementMicromapIndexOffset        = 0;

  bvh_input.vertexDirectionsBuffer       = (CUdeviceptr) vertex_direction_buffer;
  bvh_input.vertexDirectionFormat        = OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_HALF3;
  bvh_input.vertexDirectionStrideInBytes = 0;

  return bvh_input;
}

#endif /* CU_MICROMAP_H */
