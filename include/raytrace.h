#ifndef RAYTRACE_H
#define RAYTRACE_H

#include <stdint.h>

#include "image.h"
#include "primitives.h"
#include "scene.h"
#include "texture.h"
#include "utils.h"

#if __cplusplus
extern "C" {
#endif

void initialize_device();
RaytraceInstance* init_raytracing(
  General general, DeviceBuffer* albedo_atlas, int albedo_atlas_length, DeviceBuffer* illuminance_atlas, int illuminance_atlas_length,
  DeviceBuffer* material_atlas, int material_atlas_length, Scene scene);
void reset_raytracing(RaytraceInstance* instance);
void allocate_buffers(RaytraceInstance* instance);
void update_jitter(RaytraceInstance* instance);
void update_device_scene(RaytraceInstance* instance);
void prepare_trace(RaytraceInstance* instance);
void update_temporal_matrix(RaytraceInstance* instance);
void center_toy_at_camera(RaytraceInstance* instance);
void trace_scene(RaytraceInstance* instance);
void apply_bloom(RaytraceInstance* instance, RGBF* src, RGBF* dst);
void free_inputs(RaytraceInstance* instance);
void free_outputs(RaytraceInstance* instance);
void generate_clouds(RaytraceInstance* instance);
DeviceBuffer* initialize_textures(TextureRGBA* textures, const int textures_length);
void free_textures_atlas(DeviceBuffer* texture_atlas, const int textures_length);
void initialize_8bit_frame(RaytraceInstance* instance, const unsigned int width, const unsigned int height);
void free_8bit_frame(RaytraceInstance* instance);
void copy_framebuffer_to_8bit(RGBF* gpu_source, XRGB8* gpu_scratch, XRGB8* cpu_dest, const int width, const int height, const int ld);
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
