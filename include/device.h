#ifndef DEVICE_H
#define DEVICE_H

#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stubs.h>

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
void device_gather_device_table(void* dst, enum cudaMemcpyKind kind);
unsigned int device_get_thread_count();
void device_init();
void device_generate_tasks();
void device_execute_main_kernels(RaytraceInstance* instance, int depth);
void device_execute_debug_kernels(RaytraceInstance* instance);
void device_handle_accumulation(RaytraceInstance* instance);
void device_copy_framebuffer_to_8bit(
  RGBF* src, XRGB8* dst, const int width, const int height, const int ld, const OutputVariable output_variable);

////////////////////////////////////////////////////////////////////
// brdf_unittest.cuh
////////////////////////////////////////////////////////////////////
int device_brdf_unittest(const float tolerance);

////////////////////////////////////////////////////////////////////
// bsdf_lut.cuh
////////////////////////////////////////////////////////////////////
void bsdf_compute_energy_lut(RaytraceInstance* instance);

////////////////////////////////////////////////////////////////////
// camera_post.cuh
////////////////////////////////////////////////////////////////////
void device_camera_post_init(RaytraceInstance* instance);
void device_camera_post_apply(RaytraceInstance* instance, const RGBF* src, RGBF* dst);
void device_camera_post_clear(RaytraceInstance* instance);

////////////////////////////////////////////////////////////////////
// cloud_noise.cuh
////////////////////////////////////////////////////////////////////
void device_cloud_noise_generate(RaytraceInstance* instance);

////////////////////////////////////////////////////////////////////
// micromap.cuh
////////////////////////////////////////////////////////////////////
OptixBuildInputOpacityMicromap micromap_opacity_build(RaytraceInstance* instance);
void micromap_opacity_free(OptixBuildInputOpacityMicromap data);
OptixBuildInputDisplacementMicromap micromap_displacement_build(RaytraceInstance* instance);

////////////////////////////////////////////////////////////////////
// mipmap.cuh
////////////////////////////////////////////////////////////////////
void device_mipmap_generate(cudaMipmappedArray_t mipmap_array, TextureRGBA* tex);
unsigned int device_mipmap_compute_max_level(TextureRGBA* tex);

////////////////////////////////////////////////////////////////////
// particle.cuh
////////////////////////////////////////////////////////////////////
void device_particle_generate(RaytraceInstance* instance);

////////////////////////////////////////////////////////////////////
// random_unittest.cuh
////////////////////////////////////////////////////////////////////
int device_random_unittest();

////////////////////////////////////////////////////////////////////
// sky.cuh
////////////////////////////////////////////////////////////////////
void device_sky_generate_LUTs(RaytraceInstance* instance);

////////////////////////////////////////////////////////////////////
// sky_hdri.cuh
////////////////////////////////////////////////////////////////////
void sky_hdri_generate_LUT(RaytraceInstance* instance);
void sky_hdri_set_pos_to_cam(RaytraceInstance* instance);

#if __cplusplus
}
#endif

#endif /* DEVICE_H */
