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
  float offset_x_internal;
  float offset_y_internal;
  float mouse_wheel_accumulate;
  uint32_t max_scale;
} typedef DisplayZoomHandler;

void display_zoom_handler_create(DisplayZoomHandler** zoom);
void display_zoom_handler_set_display_size(DisplayZoomHandler* zoom, uint32_t width, uint32_t height);
void display_zoom_handler_update(DisplayZoomHandler* zoom, MouseState* mouse);
void display_zoom_handler_image_to_screen(
  const DisplayZoomHandler* zoom, uint32_t x, uint32_t y, uint32_t* restrict out_x, uint32_t* restrict out_y);
void display_zoom_handler_screen_to_image(
  const DisplayZoomHandler* zoom, uint32_t x, uint32_t y, uint32_t* restrict out_x, uint32_t* restrict out_y);
void display_zoom_handler_destroy(DisplayZoomHandler** zoom);

#endif /* MANDARIN_DUCK_DISPLAY_ZOOM_HANDLER */
