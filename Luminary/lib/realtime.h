#ifndef REALTIME_H
#define REALTIME_H

#include "utils.h"
#include "SDL/SDL.h"

struct RealtimeInstance {
  int width;
  int height;
  SDL_Window* window;
  SDL_Surface* window_surface;
  SDL_Surface* surface;
  RGB8* buffer;
} typedef RealtimeInstance;

RealtimeInstance* init_realtime_instance(RaytraceInstance* instance);
void free_realtime_instance(RealtimeInstance* realtime);

#endif /* REALTIME_H */