#ifndef REALTIME_H
#define REALTIME_H

#include "SDL.h"
#include "utils.h"

struct WindowInstance {
  unsigned int width;
  unsigned int height;
  unsigned int ld;
  SDL_Window* window;
  SDL_Surface* window_surface;
  XRGB8* buffer;
  DeviceBuffer* gpu_buffer;
  unsigned int max_width;
  unsigned int max_height;
} typedef WindowInstance;

WindowInstance* window_instance_init(RaytraceInstance* instance);
void window_instance_update_pointer(WindowInstance* window);
void window_instance_free(WindowInstance* window);

#endif /* REALTIME_H */
