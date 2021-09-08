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
  General general, void* albedo_atlas, int albedo_atlas_length, void* illuminance_atlas, int illuminance_atlas_length, void* material_atlas,
  int material_atlas_length, Scene scene);
void update_scene(RaytraceInstance* instance);
void trace_scene(RaytraceInstance* instance, const int temporal_frames);
void apply_bloom(RaytraceInstance* instance, RGBF* image);
void free_inputs(RaytraceInstance* instance);
void free_outputs(RaytraceInstance* instance);
void* initialize_textures(TextureRGBA* textures, const int textures_length);
void free_textures(void* texture_atlas, const int textures_length);
void initialize_8bit_frame(RaytraceInstance* instance, const int width, const int height);
void free_8bit_frame(RaytraceInstance* instance);
void copy_framebuffer_to_8bit(XRGB8* buffer, const int width, const int height, RGBF* source, RaytraceInstance* instance);
void* initialize_optix_denoise_for_realtime(RaytraceInstance* instance);
float get_auto_exposure_from_optix(void* input, RaytraceInstance* instance);
RGBF* denoise_with_optix_realtime(void* input);
void free_realtime_denoise(void* input);

#if __cplusplus
}
#endif

#endif /* RAYTRACE_H */
