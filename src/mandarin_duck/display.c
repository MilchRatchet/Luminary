#include "display.h"

static uint32_t __num_displays = 0;

void display_create(Display** _display, uint32_t width, uint32_t height) {
  MD_CHECK_NULL_ARGUMENT(_display);

  if (__num_displays == 0) {
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
  }

  __num_displays++;

  Display* display;
  LUM_FAILURE_HANDLE(host_malloc(&display, sizeof(Display)));

  SDL_Rect rect;
  SDL_Rect screen_size;

  int display_count;
  SDL_DisplayID* displays = SDL_GetDisplays(&display_count);

  if (display_count == 0) {
    crash_message("No displays available.");
  }

  SDL_GetDisplayUsableBounds(displays[0], &rect);
  SDL_GetDisplayBounds(displays[0], &screen_size);

  rect.w = min(rect.w, (int) width);
  rect.h = min(rect.h, (int) height);

  info_message("Size: %d %d", rect.w, rect.h);

  SDL_PropertiesID sdl_properties = SDL_CreateProperties();
  SDL_SetStringProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_TITLE_STRING, "MandarinDuck - Using Luminary");
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_X_NUMBER, SDL_WINDOWPOS_CENTERED);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_Y_NUMBER, SDL_WINDOWPOS_CENTERED);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, rect.w);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, rect.h);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_WINDOW_CREATE_BORDERLESS_BOOLEAN, 1);

  display->sdl_window = SDL_CreateWindowWithProperties(sdl_properties);

  SDL_DestroyProperties(sdl_properties);

  if (!display->sdl_window) {
    crash_message("Failed to create SDL_Window.");
  }

  *_display = display;
}

void display_query_events(Display* display, bool* exit_requested) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(exit_requested);

  SDL_Event event;

  while (SDL_PollEvent(&event)) {
    switch (event.type) {
      case SDL_EVENT_QUIT:
        *exit_requested = true;
        break;
      default:
        warn_message("Unhandled SDL event type: %u.", event.type);
        break;
    }
  }
}

void display_update(Display* display) {
  SDL_UpdateWindowSurface(display->sdl_window);
}

void display_destroy(Display** display) {
  MD_CHECK_NULL_ARGUMENT(display);

  SDL_DestroyWindow((*display)->sdl_window);
  LUM_FAILURE_HANDLE(host_free(display));

  __num_displays--;

  if (__num_displays == 0) {
    SDL_Quit();
  }
}
