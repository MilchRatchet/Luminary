#include "display_zoom_handler.h"

void display_zoom_handler_create(DisplayZoomHandler** zoom) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  LUM_FAILURE_HANDLE(host_malloc(zoom, sizeof(DisplayZoomHandler)));
  memset(*zoom, 0, sizeof(DisplayZoomHandler));
}

void display_zoom_handler_set_display_size(DisplayZoomHandler* zoom, uint32_t width, uint32_t height) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  zoom->display_width  = width;
  zoom->display_height = height;
}

void display_zoom_handler_update(DisplayZoomHandler* zoom, MouseState* mouse) {
  MD_CHECK_NULL_ARGUMENT(zoom);
  MD_CHECK_NULL_ARGUMENT(mouse);

  if (mouse->wheel_motion == 0.0f)
    return;

  zoom->mouse_wheel_accumulate += mouse->wheel_motion;
  zoom->mouse_wheel_accumulate = fmaxf(zoom->mouse_wheel_accumulate, 0.0f);

  const uint32_t old_scale = zoom->scale;
  const uint32_t new_scale = (uint32_t) zoom->mouse_wheel_accumulate;

  if (new_scale != old_scale) {
    if (new_scale) {
      const uint32_t screen_pos_x = mouse->x;
      const uint32_t screen_pos_y = mouse->y;

      const uint32_t image_pos_x = zoom->offset_x + (screen_pos_x >> old_scale);
      const uint32_t image_pos_y = zoom->offset_y + (screen_pos_y >> old_scale);

      zoom->offset_x = image_pos_x - min(screen_pos_x >> new_scale, image_pos_x);
      zoom->offset_y = image_pos_y - min(screen_pos_y >> new_scale, image_pos_y);

      const uint32_t effective_display_width  = zoom->display_width >> new_scale;
      const uint32_t effective_display_height = zoom->display_height >> new_scale;

      zoom->offset_x = min(zoom->offset_x, zoom->display_width - effective_display_width);
      zoom->offset_y = min(zoom->offset_y, zoom->display_height - effective_display_height);
    }
    else {
      zoom->offset_x = 0;
      zoom->offset_y = 0;
    }

    zoom->scale = new_scale;
  }
}

void display_zoom_handler_destroy(DisplayZoomHandler** zoom) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  LUM_FAILURE_HANDLE(host_free(zoom));
}
