#include "realtime.h"

#include <stdlib.h>

#include "raytrace.h"

RealtimeInstance* init_realtime_instance(RaytraceInstance* instance) {
  RealtimeInstance* realtime = (RealtimeInstance*) malloc(sizeof(RealtimeInstance));

  SDL_Init(SDL_INIT_VIDEO);

  SDL_Rect rect;
  SDL_Rect screen_size;

  SDL_GetDisplayUsableBounds(0, &rect);
  SDL_GetDisplayBounds(0, &screen_size);

  rect.y += screen_size.h - rect.h;
  rect.h -= screen_size.h - rect.h;

  // It seems SDL window dimensions must be a multiple of 4
  rect.h = rect.h & 0xfffffffc;

  rect.w = rect.h * ((float) instance->width) / instance->height;

  rect.w = rect.w & 0xfffffffc;

  if (rect.w > screen_size.w) {
    rect.w = screen_size.w;
    rect.h = rect.w * ((float) instance->height) / instance->width;
  }

  realtime->width  = rect.w;
  realtime->height = rect.h;

  realtime->window = SDL_CreateWindow("Luminary", SDL_WINDOWPOS_CENTERED, rect.y, rect.w, rect.h, SDL_WINDOW_SHOWN);

  realtime->window_surface = SDL_GetWindowSurface(realtime->window);

  Uint32 rmask, gmask, bmask, amask;

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
  rmask = 0xff000000;
  gmask = 0x00ff0000;
  bmask = 0x0000ff00;
  amask = 0x000000ff;
#else
  rmask = 0x000000ff;
  gmask = 0x0000ff00;
  bmask = 0x00ff0000;
  amask = 0xff000000;
#endif

  realtime->buffer = (XRGB8*) realtime->window_surface->pixels;
  realtime->ld     = realtime->window_surface->pitch;
  initialize_8bit_frame(instance, max(rect.w, instance->width), max(rect.h, instance->height));

  return realtime;
}

void free_realtime_instance(RealtimeInstance* realtime) {
  SDL_DestroyWindow(realtime->window);
  SDL_Quit();
  free(realtime);
}
