/*
 * raytrace.h - Routines for raytracing
 */
#include <stdint.h>
#include "scene.h"
#include "image.h"
#include "primitives.h"

#if __cplusplus
extern "C" {
#endif

struct raytrace_instance {
  unsigned int width;
  unsigned int height;
  RGBF* frame_buffer;
  RGBF* frame_buffer_gpu;
  unsigned int reflection_depth;
} typedef raytrace_instance;

raytrace_instance* init_raytracing(
  const unsigned int width, const unsigned int height, const unsigned int reflection_depth);
void trace_scene(Scene scene, raytrace_instance* instance);
void frame_buffer_to_image(Camera camera, raytrace_instance* instance, RGB8* image);
void free_raytracing(raytrace_instance* instance);
#if __cplusplus
}
#endif
