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
  zoom->max_scale      = (uint32_t) fmaxf(log2f(max(width, height)), 0.0f);
}

static void _display_zoom_handler_handle_scroll(DisplayZoomHandler* zoom, MouseState* mouse) {
  MD_CHECK_NULL_ARGUMENT(zoom);
  MD_CHECK_NULL_ARGUMENT(mouse);

  if (mouse->wheel_motion == 0.0f)
    return;

  zoom->mouse_wheel_accumulate += mouse->wheel_motion;
  zoom->mouse_wheel_accumulate = fmaxf(zoom->mouse_wheel_accumulate, 0.0f);
  zoom->mouse_wheel_accumulate = fminf(zoom->mouse_wheel_accumulate, zoom->max_scale);

  const uint32_t old_scale = zoom->scale;
  const uint32_t new_scale = (uint32_t) zoom->mouse_wheel_accumulate;

  if (new_scale != old_scale) {
    if (new_scale) {
      const uint32_t screen_pos_x = mouse->x;
      const uint32_t screen_pos_y = mouse->y;

      const float image_pos_x = zoom->offset_x_internal + (screen_pos_x >> old_scale);
      const float image_pos_y = zoom->offset_y_internal + (screen_pos_y >> old_scale);

      zoom->offset_x_internal = image_pos_x - (screen_pos_x >> new_scale);
      zoom->offset_y_internal = image_pos_y - (screen_pos_y >> new_scale);
    }
    else {
      zoom->offset_x_internal = 0.0f;
      zoom->offset_y_internal = 0.0f;
    }

    zoom->scale = new_scale;
  }
}

static void _display_zoom_handler_handle_swipe(DisplayZoomHandler* zoom, MouseState* mouse) {
  MD_CHECK_NULL_ARGUMENT(zoom);
  MD_CHECK_NULL_ARGUMENT(mouse);

  if (mouse->right_down == false)
    return;

  if (mouse->x_motion == 0.0f && mouse->y_motion == 0.0f)
    return;

  zoom->offset_x_internal -= mouse->x_motion / (1u << zoom->scale);
  zoom->offset_y_internal -= mouse->y_motion / (1u << zoom->scale);
}

static void _display_zoom_handler_update_offset(DisplayZoomHandler* zoom) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  const uint32_t effective_display_width  = zoom->display_width >> zoom->scale;
  const uint32_t effective_display_height = zoom->display_height >> zoom->scale;

  zoom->offset_x_internal = fminf(zoom->offset_x_internal, zoom->display_width - effective_display_width);
  zoom->offset_y_internal = fminf(zoom->offset_y_internal, zoom->display_height - effective_display_height);
  zoom->offset_x_internal = fmaxf(zoom->offset_x_internal, 0.0f);
  zoom->offset_y_internal = fmaxf(zoom->offset_y_internal, 0.0f);

  zoom->offset_x = (uint32_t) zoom->offset_x_internal;
  zoom->offset_y = (uint32_t) zoom->offset_y_internal;
}

void display_zoom_handler_update(DisplayZoomHandler* zoom, MouseState* mouse) {
  MD_CHECK_NULL_ARGUMENT(zoom);
  MD_CHECK_NULL_ARGUMENT(mouse);

  _display_zoom_handler_handle_swipe(zoom, mouse);
  _display_zoom_handler_handle_scroll(zoom, mouse);
  _display_zoom_handler_update_offset(zoom);
}

void display_zoom_handler_destroy(DisplayZoomHandler** zoom) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  LUM_FAILURE_HANDLE(host_free(zoom));
}
