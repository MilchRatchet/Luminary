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
  int illuminance_atlas_length, void* material_atlas, int material_atlas_length);
void trace_scene(Scene scene, raytrace_instance* instance);
void free_raytracing(raytrace_instance* instance);
void* initialize_textures(TextureRGBA* textures, const int textures_length);
void free_textures(void* texture_atlas, const int textures_length);
#if __cplusplus
}
#endif

#endif /* RAYTRACE_H */
