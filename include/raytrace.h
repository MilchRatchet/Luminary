#ifndef RAYTRACE_H
#define RAYTRACE_H

#include <stdint.h>

#include "scene.h"
#include "structs.h"
#include "utils.h"

#if __cplusplus
extern "C" {
#endif

void initialize_device();
void raytracing_init(RaytraceInstance** _instance, General general, TextureAtlas tex_atlas, Scene* scene);
void reset_raytracing(RaytraceInstance* instance);
void allocate_buffers(RaytraceInstance* instance);
void update_jitter(RaytraceInstance* instance);
void update_device_scene(RaytraceInstance* instance);
void prepare_trace(RaytraceInstance* instance);
void update_temporal_matrix(RaytraceInstance* instance);
void center_toy_at_camera(RaytraceInstance* instance);
void trace_scene(RaytraceInstance* instance);
void apply_bloom(RaytraceInstance* instance, RGBAhalf* src, RGBAhalf* dst);
void free_inputs(RaytraceInstance* instance);
void free_outputs(RaytraceInstance* instance);
void clouds_generate(RaytraceInstance* instance);
void sky_generate_LUTs(RaytraceInstance* instance);
void cudatexture_create_atlas(DeviceBuffer** buffer, TextureRGBA* textures, const int textures_length, const uint32_t flags);
void cudatexture_free_buffer(DeviceBuffer* texture_atlas, const int textures_length);
void initialize_8bit_frame(RaytraceInstance* instance, const unsigned int width, const unsigned int height);
void free_8bit_frame(RaytraceInstance* instance);
void copy_framebuffer_to_8bit(RGBAhalf* gpu_source, XRGB8* gpu_scratch, XRGB8* cpu_dest, const int width, const int height, const int ld);
void* memcpy_gpu_to_cpu(void* gpu_ptr, size_t size);
void* memcpy_texture_to_cpu(void* textures_ptr, uint64_t* count);
void update_device_pointers(RaytraceInstance* instance);
void free_host_memory(void* ptr);
int brdf_unittest(const float tolerance);

/*
 * OptiX Denoiser Interface
 */
void optix_denoise_create(RaytraceInstance* instance);
DeviceBuffer* optix_denoise_apply(RaytraceInstance* instance, RGBF* src);
float optix_denoise_auto_exposure(RaytraceInstance* instance);
void optix_denoise_free(RaytraceInstance* instance);

#if __cplusplus
}
#endif

#endif /* RAYTRACE_H */
