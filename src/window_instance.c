#include "window_instance.h"

#include <stdlib.h>

#include "buffer.h"
#include "raytrace.h"

// TODO: If window is not maximum height then position the window in the middle rather then the top.
WindowInstance* window_instance_init(RaytraceInstance* instance) {
  WindowInstance* window = (WindowInstance*) calloc(1, sizeof(WindowInstance));

  SDL_Init(SDL_INIT_VIDEO);

  SDL_Rect rect;
  SDL_Rect screen_size;

  SDL_GetDisplayUsableBounds(0, &rect);
  SDL_GetDisplayBounds(0, &screen_size);

  // max_width also acts as leading dimension so it must be slightly larger
  window->max_width  = screen_size.w + 32;
  window->max_height = screen_size.h;

  const int taskbar_margin = screen_size.h - rect.h;

  rect.h -= 4 * taskbar_margin;

  rect.w = min(rect.w, (int) instance->width);
  rect.h = min(rect.h, (int) instance->height);

  const float aspect_ratio = ((float) instance->width) / instance->height;

  // It seems SDL window dimensions must be a multiple of 4
  rect.h = rect.h & 0xfffffffc;

  rect.w = rect.h * aspect_ratio;

  rect.w = rect.w & 0xfffffffc;

  if (rect.w > screen_size.w) {
    rect.w = screen_size.w;
    rect.h = rect.w / aspect_ratio;
  }

  window->width  = (unsigned int) rect.w;
  window->height = (unsigned int) rect.h;

  window->window =
    SDL_CreateWindow("Luminary", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, rect.w, rect.h, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

  window_instance_update_pointer(window);

  device_buffer_init(&window->gpu_buffer);
  device_buffer_malloc(window->gpu_buffer, sizeof(XRGB8), window->max_width * window->max_height);

  return window;
}

void window_instance_update_pointer(WindowInstance* window) {
  window->window_surface = SDL_GetWindowSurface(window->window);
  window->buffer         = (XRGB8*) window->window_surface->pixels;
  SDL_GetWindowSize(window->window, (int*) &(window->width), (int*) &(window->height));
  window->ld = ((unsigned int) window->window_surface->pitch) / sizeof(XRGB8);
}

void window_instance_free(WindowInstance* window) {
  SDL_DestroyWindow(window->window);
  SDL_Quit();
  device_buffer_destroy(&window->gpu_buffer);
  free(window);
}
