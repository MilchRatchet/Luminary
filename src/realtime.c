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

  realtime->width  = (unsigned int) rect.w;
  realtime->height = (unsigned int) rect.h;

  realtime->window = SDL_CreateWindow("Luminary", SDL_WINDOWPOS_CENTERED, rect.y, rect.w, rect.h, SDL_WINDOW_SHOWN);

  realtime->window_surface = SDL_GetWindowSurface(realtime->window);

  realtime->buffer = (XRGB8*) realtime->window_surface->pixels;
  realtime->ld     = (unsigned int) realtime->window_surface->pitch;

  realtime->gpu_buffer_size = max((unsigned int) rect.w, instance->width) * max((unsigned int) rect.h, instance->height);
  initialize_8bit_frame(instance, max((unsigned int) rect.w, instance->width), max((unsigned int) rect.h, instance->height));

  return realtime;
}

void update_8bit_frame(RealtimeInstance* realtime, RaytraceInstance* instance) {
  const unsigned int required_buffer_size = max(realtime->width, instance->width) * max(realtime->height, instance->height);

  if (required_buffer_size > realtime->gpu_buffer_size) {
    free_8bit_frame(instance);
    initialize_8bit_frame(instance, max(realtime->width, instance->width), max(realtime->height, instance->height));
    realtime->gpu_buffer_size = required_buffer_size;
  }
}

void free_realtime_instance(RealtimeInstance* realtime) {
  SDL_DestroyWindow(realtime->window);
  SDL_Quit();
  free(realtime);
}
