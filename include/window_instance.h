#ifndef WINDOW_INSTANCE_H
#define WINDOW_INSTANCE_H

#include "SDL.h"
#include "SDL_opengl.h"
#include "SDL_opengl_glext.h"
#include "utils.h"

struct WindowInstance {
  unsigned int width;
  unsigned int height;
  unsigned int ld;
  SDL_Window* window;
  DeviceBuffer* buffer;
  unsigned int max_width;
  unsigned int max_height;
  GLuint tex;
  cudaGraphicsResource_t resource;
  GLuint framebuffer;
} typedef WindowInstance;

WindowInstance* window_instance_init(RaytraceInstance* instance);
void window_instance_resize_buffer(WindowInstance* window);
int window_instance_is_visible(const WindowInstance* window);
void window_instance_update(WindowInstance* window);
void window_instance_free(WindowInstance* window);

#endif /* WINDOW_INSTANCE_H */
