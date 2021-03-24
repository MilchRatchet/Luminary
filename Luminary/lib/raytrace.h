#ifndef RAYTRACE_H
#define RAYTRACE_H

#include <stdint.h>
#include "scene.h"
#include "image.h"
#include "primitives.h"
#include "texture.h"

#if __cplusplus
extern "C" {
#endif

struct raytrace_instance {
  unsigned int width;
  unsigned int height;
  RGBF* frame_buffer;
  RGBF* frame_buffer_gpu;
  int reflection_depth;
  int diffuse_samples;
} typedef raytrace_instance;

void initialize_device();
raytrace_instance* init_raytracing(
  const unsigned int width, const unsigned int height, const int reflection_depth,
  const int diffuse_samples);
void trace_scene(
  Scene scene, raytrace_instance* instance, void* albedo_atlas, void* illuminance_atlas,
  void* material_atlas, texture_assignment* texture_assignments, int meshes_count);
void free_raytracing(raytrace_instance* instance);
void* initialize_textures(TextureRGBA* textures, const int textures_length);
void free_textures(void* texture_atlas, const int textures_length);
#if __cplusplus
}
#endif

#endif /* RAYTRACE_H */
