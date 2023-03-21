#ifndef DEVICE_H
#define DEVICE_H

#include <cuda_runtime_api.h>

#include "stddef.h"
#include "structs.h"
#include "utils.h"

#if __cplusplus
extern "C" {
#endif

////////////////////////////////////////////////////////////////////
// device.cu
////////////////////////////////////////////////////////////////////
#define device_update_symbol(symbol, data) _device_update_symbol(offsetof(DeviceConstantMemory, symbol), &(data), sizeof(data))
#define device_gather_symbol(symbol, data) _device_gather_symbol(&(data), offsetof(DeviceConstantMemory, symbol), sizeof(data))
void _device_update_symbol(const size_t offset, const void* src, const size_t size);
void _device_gather_symbol(void* dst, const size_t offset, const size_t size);
void device_initialize_random_generators();
unsigned int device_get_thread_count();
void device_init();
void device_generate_tasks();
void device_execute_main_kernels(RaytraceInstance* instance, int type);
void device_execute_debug_kernels(RaytraceInstance* instance, int type);
void device_handle_accumulation(RaytraceInstance* instance);
void device_copy_framebuffer_to_8bit(
  RGBAhalf* gpu_source, XRGB8* gpu_scratch, XRGB8* cpu_dest, const int width, const int height, const int ld);

////////////////////////////////////////////////////////////////////
// bloom.cuh
////////////////////////////////////////////////////////////////////
void device_bloom_apply(RaytraceInstance* instance, RGBAhalf* src, RGBAhalf* dst);
void device_bloom_allocate_mips(RaytraceInstance* instance);
void device_bloom_free_mips(RaytraceInstance* instance);

////////////////////////////////////////////////////////////////////
// brdf_unittest.cuh
////////////////////////////////////////////////////////////////////
int device_brdf_unittest(const float tolerance);

////////////////////////////////////////////////////////////////////
// cloud_noise.cuh
////////////////////////////////////////////////////////////////////
void device_cloud_noise_generate(RaytraceInstance* instance);

////////////////////////////////////////////////////////////////////
// mipmap.cuh
////////////////////////////////////////////////////////////////////
void device_mipmap_generate(cudaMipmappedArray_t mipmap_array, TextureRGBA* tex);
unsigned int device_mipmap_compute_max_level(TextureRGBA* tex);

////////////////////////////////////////////////////////////////////
// sky.cuh
////////////////////////////////////////////////////////////////////
void device_sky_generate_LUTs(RaytraceInstance* instance);

#if __cplusplus
}
#endif

#endif /* DEVICE_H */
