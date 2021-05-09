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
void trace_scene(Scene scene, raytrace_instance* instance, const int progress);
void free_inputs(raytrace_instance* instance);
void free_outputs(raytrace_instance* instance);
void* initialize_textures(TextureRGBA* textures, const int textures_length);
void free_textures(void* texture_atlas, const int textures_length);
void initiliaze_realtime(raytrace_instance* instance);
void free_realtime(raytrace_instance* instance);
void copy_framebuffer_to_8bit(RGB8* buffer, raytrace_instance* instance);

#if __cplusplus
}
#endif

#endif /* RAYTRACE_H */
