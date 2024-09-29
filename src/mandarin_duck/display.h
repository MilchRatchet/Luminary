#ifndef MANDARIN_DUCK_DISPLAY_H
#define MANDARIN_DUCK_DISPLAY_H

#include <SDL3/SDL.h>

#include "utils.h"

struct Display {
  SDL_Window* sdl_window;
  uint32_t width;
  uint32_t height;
} typedef Display;

void display_create(Display** _display, uint32_t width, uint32_t height);
void display_query_events(Display* display, bool* exit_requested);
void display_update(Display* display);
void display_destroy(Display** display);

#endif /* MANDARIN_DUCK_DISPLAY_H */
