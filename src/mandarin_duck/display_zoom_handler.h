#ifndef MANDARIN_DUCK_DISPLAY_ZOOM_HANDLER
#define MANDARIN_DUCK_DISPLAY_ZOOM_HANDLER

#include "mouse_state.h"
#include "utils.h"

struct DisplayZoomHandler {
  uint32_t display_width;
  uint32_t display_height;
  uint32_t offset_x;
  uint32_t offset_y;
  uint32_t scale;
  float mouse_wheel_accumulate;
} typedef DisplayZoomHandler;

void display_zoom_handler_create(DisplayZoomHandler** zoom);
void display_zoom_handler_set_display_size(DisplayZoomHandler* zoom, uint32_t width, uint32_t height);
void display_zoom_handler_update(DisplayZoomHandler* zoom, MouseState* mouse);
void display_zoom_handler_destroy(DisplayZoomHandler** zoom);

#endif /* MANDARIN_DUCK_DISPLAY_ZOOM_HANDLER */
