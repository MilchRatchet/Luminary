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
void raytrace_init(RaytraceInstance** _instance, General general, TextureAtlas tex_atlas, Scene* scene, CommandlineOptions options);
void raytrace_reset(RaytraceInstance* instance);
void raytrace_prepare(RaytraceInstance* instance);
void raytrace_increment(RaytraceInstance* instance);
void raytrace_build_structures(RaytraceInstance* instance);
void raytrace_update_device_pointers(RaytraceInstance* instance);
void raytrace_allocate_buffers(RaytraceInstance* instance);
void raytrace_free_work_buffers(RaytraceInstance* instance);
void raytrace_free_output_buffers(RaytraceInstance* instance);
void raytrace_init_8bit_frame(RaytraceInstance* instance, const unsigned int width, const unsigned int height);
void raytrace_free_8bit_frame(RaytraceInstance* instance);
void raytrace_update_ray_emitter(RaytraceInstance* instance);
void raytrace_update_toy_rotation(RaytraceInstance* instance);
void raytrace_update_device_scene(RaytraceInstance* instance);
void raytrace_center_toy_at_camera(RaytraceInstance* instance);
DeviceBuffer* raytrace_get_accumulate_buffer(RaytraceInstance* instance, OutputVariable output_variable);
const char* raytrace_get_output_variable_name(OutputVariable output_variable);

#if __cplusplus
}
#endif

#endif /* RAYTRACE_H */
