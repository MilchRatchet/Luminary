#ifndef REALTIME_H
#define REALTIME_H

#include "SDL.h"
#include "utils.h"

struct RealtimeInstance {
  unsigned int width;
  unsigned int height;
  unsigned int ld;
  SDL_Window* window;
  SDL_Surface* window_surface;
  XRGB8* buffer;
  unsigned int gpu_buffer_size;
} typedef RealtimeInstance;

RealtimeInstance* init_realtime_instance(RaytraceInstance* instance);
void update_8bit_frame(RealtimeInstance* realtime, RaytraceInstance* instance);
void free_realtime_instance(RealtimeInstance* realtime);

#endif /* REALTIME_H */
