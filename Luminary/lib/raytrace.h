#ifndef RAYTRACE_H
#define RAYTRACE_H

#include <stdint.h>
#include "scene.h"
#include "image.h"
#include "primitives.h"
#include "texture.h"
#include "utils.h"

#if __cplusplus
extern "C" {
#endif

void initialize_device();
raytrace_instance* init_raytracing(
  const unsigned int width, const unsigned int height, const int reflection_depth,
  const int diffuse_samples, void* albedo_atlas, int albedo_atlas_length, void* illuminance_atlas,
  int illuminance_atlas_length, void* material_atlas, int material_atlas_length, Scene scene,
  int denoiser);
void copy_framebuffer_to_cpu(raytrace_instance* instance);
void trace_scene(
  raytrace_instance* instance, const int progress, const int temporal_frames,
  const unsigned int update_mask);
void free_inputs(raytrace_instance* instance);
void free_outputs(raytrace_instance* instance);
void* initialize_textures(TextureRGBA* textures, const int textures_length);
void free_textures(void* texture_atlas, const int textures_length);
void initiliaze_8bit_frame(raytrace_instance* instance);
void free_8bit_frame(raytrace_instance* instance);
void copy_framebuffer_to_8bit(RGB8* buffer, RGBF* source, raytrace_instance* instance);
void* initialize_optix_denoise_for_realtime(raytrace_instance* instance);
float get_auto_exposure_from_optix(void* input, const float exposure);
RGBF* denoise_with_optix_realtime(void* input);
void free_realtime_denoise(void* input);

#if __cplusplus
}
#endif

#endif /* RAYTRACE_H */
