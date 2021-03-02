/*
 * raytrace.h - Routines for raytracing
 */
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

raytrace_instance* init_raytracing(
  const unsigned int width, const unsigned int height, const int reflection_depth,
  const int diffuse_samples);
void trace_scene(Scene scene, raytrace_instance* instance);
void frame_buffer_to_image(Camera camera, raytrace_instance* instance, RGB8* image);
void free_raytracing(raytrace_instance* instance);
void* initialize_textures(TextureRGBA* textures, const int textures_length);
void free_textures(void* texture_atlas, const int textures_length);
#if __cplusplus
}
#endif
