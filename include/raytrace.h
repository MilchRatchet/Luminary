#ifndef RAYTRACE_H
#define RAYTRACE_H

#include <stdint.h>

#include "scene.h"
#include "structs.h"
#include "utils.h"

#if __cplusplus
extern "C" {
#endif

void raytrace_execute(RaytraceInstance* instance);
void raytrace_init(RaytraceInstance** _instance, General general, TextureAtlas tex_atlas, Scene* scene);
void raytrace_reset(RaytraceInstance* instance);
void raytrace_prepare(RaytraceInstance* instance);
void raytrace_update_device_pointers(RaytraceInstance* instance);
void raytrace_allocate_buffers(RaytraceInstance* instance);
void raytrace_free_work_buffers(RaytraceInstance* instance);
void raytrace_free_output_buffers(RaytraceInstance* instance);
void raytrace_init_8bit_frame(RaytraceInstance* instance, const unsigned int width, const unsigned int height);
void raytrace_free_8bit_frame(RaytraceInstance* instance);
void raytrace_update_temporal_matrix(RaytraceInstance* instance);
void raytrace_update_jitter(RaytraceInstance* instance);
void raytrace_update_camera_transform(const Scene scene, const unsigned int width, const unsigned int height);
void raytrace_update_light_resampling_active(RaytraceInstance* instance);

////////////////////////////////////////////////////////////////////
// not yet ported
////////////////////////////////////////////////////////////////////

void raytrace_update_device_scene(RaytraceInstance* instance);
void raytrace_center_toy_at_camera(RaytraceInstance* instance);
void apply_bloom(RaytraceInstance* instance, RGBAhalf* src, RGBAhalf* dst);
void clouds_generate(RaytraceInstance* instance);
void sky_generate_LUTs(RaytraceInstance* instance);
void device_copy_framebuffer_to_8bit(
  RGBAhalf* gpu_source, XRGB8* gpu_scratch, XRGB8* cpu_dest, const int width, const int height, const int ld);
void* memcpy_gpu_to_cpu(void* gpu_ptr, size_t size);
void* memcpy_texture_to_cpu(void* textures_ptr, uint64_t* count);
void free_host_memory(void* ptr);
int brdf_unittest(const float tolerance);

#if __cplusplus
}
#endif

#endif /* RAYTRACE_H */
