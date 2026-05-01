#include "display_zoom_handler.h"

// Helper functions for directional shifting based on scale sign
static inline uint32_t _map_screen_to_image(uint32_t val, int32_t scale) {
  return (scale > 0) ? (val >> scale) : (val << -scale);
}

static inline uint32_t _map_image_to_screen(uint32_t val, int32_t scale) {
  return (scale > 0) ? (val << scale) : (val >> -scale);
}

static void _display_zoom_handler_recalculate_scale_bounds(DisplayZoomHandler* zoom) {
  zoom->max_scale = (int32_t) fmaxf(log2f(max(zoom->display_width, zoom->display_height)), 0.0f);

  if (zoom->display_width > 0 && zoom->display_height > 0) {
    float width_ratio  = (float) zoom->image_width / zoom->display_width;
    float height_ratio = (float) zoom->image_height / zoom->display_height;
    float max_ratio    = fmaxf(width_ratio, height_ratio);

    if (max_ratio > 1.0f) {
      zoom->min_scale = -(int32_t) ceilf(log2f(max_ratio));
    }
    else {
      zoom->min_scale = 0;
    }
  }
  else {
    zoom->min_scale = 0;
  }
}

void display_zoom_handler_create(DisplayZoomHandler** zoom) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  LUM_FAILURE_HANDLE(host_malloc(zoom, sizeof(DisplayZoomHandler)));
  memset(*zoom, 0, sizeof(DisplayZoomHandler));
}

void display_zoom_handler_set_display_size(DisplayZoomHandler* zoom, uint32_t width, uint32_t height) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  zoom->display_width  = width;
  zoom->display_height = height;
  _display_zoom_handler_recalculate_scale_bounds(zoom);
}

void display_zoom_handler_set_image_size(DisplayZoomHandler* zoom, uint32_t width, uint32_t height) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  zoom->image_width  = width;
  zoom->image_height = height;
  _display_zoom_handler_recalculate_scale_bounds(zoom);
}

static void _display_zoom_handler_handle_scroll(DisplayZoomHandler* zoom, MouseState* mouse) {
  MD_CHECK_NULL_ARGUMENT(zoom);
  MD_CHECK_NULL_ARGUMENT(mouse);

  if (mouse->wheel_motion == 0.0f)
    return;

  zoom->mouse_wheel_accumulate += mouse->wheel_motion;
  zoom->mouse_wheel_accumulate = fmaxf(zoom->mouse_wheel_accumulate, (float) zoom->min_scale);
  zoom->mouse_wheel_accumulate = fminf(zoom->mouse_wheel_accumulate, (float) zoom->max_scale);

  const int32_t old_scale = zoom->scale;
  const int32_t new_scale = (int32_t) zoom->mouse_wheel_accumulate;

  if (new_scale != old_scale) {
    const uint32_t screen_pos_x = mouse->x;
    const uint32_t screen_pos_y = mouse->y;

    const float image_pos_x = zoom->offset_x_internal + _map_screen_to_image(screen_pos_x, old_scale);
    const float image_pos_y = zoom->offset_y_internal + _map_screen_to_image(screen_pos_y, old_scale);

    zoom->offset_x_internal = image_pos_x - _map_screen_to_image(screen_pos_x, new_scale);
    zoom->offset_y_internal = image_pos_y - _map_screen_to_image(screen_pos_y, new_scale);

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

  float scale_factor = (zoom->scale > 0) ? (1.0f / (1u << zoom->scale)) : (float) (1u << -zoom->scale);

  zoom->offset_x_internal -= mouse->x_motion * scale_factor;
  zoom->offset_y_internal -= mouse->y_motion * scale_factor;
}

static void _display_zoom_handler_update_offset(DisplayZoomHandler* zoom) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  const uint32_t effective_display_width  = _map_screen_to_image(zoom->display_width, zoom->scale);
  const uint32_t effective_display_height = _map_screen_to_image(zoom->display_height, zoom->scale);

  float min_x = fminf(0.0f, (float) zoom->image_width - (float) effective_display_width);
  float max_x = fmaxf(0.0f, (float) zoom->image_width - (float) effective_display_width);

  float min_y = fminf(0.0f, (float) zoom->image_height - (float) effective_display_height);
  float max_y = fmaxf(0.0f, (float) zoom->image_height - (float) effective_display_height);

  zoom->offset_x_internal = fmaxf(min_x, fminf(zoom->offset_x_internal, max_x));
  zoom->offset_y_internal = fmaxf(min_y, fminf(zoom->offset_y_internal, max_y));

  zoom->offset_x = (int32_t) zoom->offset_x_internal;
  zoom->offset_y = (int32_t) zoom->offset_y_internal;
}

void display_zoom_handler_update(DisplayZoomHandler* zoom, MouseState* mouse) {
  MD_CHECK_NULL_ARGUMENT(zoom);
  MD_CHECK_NULL_ARGUMENT(mouse);

  _display_zoom_handler_handle_swipe(zoom, mouse);
  _display_zoom_handler_handle_scroll(zoom, mouse);
  _display_zoom_handler_update_offset(zoom);
}

void display_zoom_handler_image_to_screen(
  const DisplayZoomHandler* zoom, uint32_t x, uint32_t y, uint32_t* restrict out_x, uint32_t* restrict out_y) {
  MD_CHECK_NULL_ARGUMENT(zoom);
  MD_CHECK_NULL_ARGUMENT(out_x);
  MD_CHECK_NULL_ARGUMENT(out_y);

  int32_t image_x_offset  = (int32_t) x - zoom->offset_x;
  const uint32_t screen_x = (image_x_offset >= 0) ? _map_image_to_screen((uint32_t) image_x_offset, zoom->scale) : 0;

  int32_t image_y_offset  = (int32_t) y - zoom->offset_y;
  const uint32_t screen_y = (image_y_offset >= 0) ? _map_image_to_screen((uint32_t) image_y_offset, zoom->scale) : 0;

  *out_x = min(screen_x, zoom->display_width);
  *out_y = min(screen_y, zoom->display_height);
}

void display_zoom_handler_screen_to_image(
  const DisplayZoomHandler* zoom, uint32_t x, uint32_t y, uint32_t* restrict out_x, uint32_t* restrict out_y) {
  MD_CHECK_NULL_ARGUMENT(zoom);
  MD_CHECK_NULL_ARGUMENT(out_x);
  MD_CHECK_NULL_ARGUMENT(out_y);

  int32_t image_x = (int32_t) _map_screen_to_image(x, zoom->scale) + zoom->offset_x;
  *out_x          = (uint32_t) max(0, image_x);

  int32_t image_y = (int32_t) _map_screen_to_image(y, zoom->scale) + zoom->offset_y;
  *out_y          = (uint32_t) max(0, image_y);
}

void display_zoom_handler_destroy(DisplayZoomHandler** zoom) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  LUM_FAILURE_HANDLE(host_free(zoom));
}
