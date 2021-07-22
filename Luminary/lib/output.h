#ifndef OUTPUT_H
#define OUTPUT_H

#include "utils.h"
#include "SDL/SDL.h"
#include <time.h>

struct RealtimeInstance {
  int width;
  int height;
  SDL_Window* window;
  SDL_Surface* window_surface;
  SDL_Surface* surface;
  RGB8* buffer;
} typedef RealtimeInstance;

void offline_output(
  Scene scene, RaytraceInstance* instance, char* output_name, int progress, clock_t time);
void realtime_output(Scene scene, RaytraceInstance* instance, const int filters);

#endif /* OUTPUT_H */
