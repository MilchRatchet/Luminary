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

static float _display_zoom_handler_clamp_axis_offset(float offset, uint32_t display_size, uint32_t image_size, int32_t scale) {
  if (display_size == 0 || image_size == 0)
    return 0.0f;

  const uint32_t last_sample = _map_screen_to_image(display_size - 1, scale);
  const float max_offset     = (float) (image_size - 1);
  const float min_offset     = -(float) last_sample;

  if (scale >= 0 || (1u << -scale) <= image_size)
    return fmaxf(min_offset, fminf(offset, max_offset));

  const float sample_stride    = (float) (1u << -scale);
  const float requested_offset = (float) (int32_t) offset;
  float sample_index           = roundf(((max_offset * 0.5f) - requested_offset) / sample_stride);
  sample_index                 = fmaxf(0.0f, fminf(sample_index, (float) (display_size - 1)));

  const float interval_min = -sample_index * sample_stride;
  const float interval_max = interval_min + max_offset;

  return fmaxf(interval_min, fminf(offset, interval_max));
}

static void _display_zoom_handler_update_offset(DisplayZoomHandler* zoom) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  zoom->offset_x_internal =
    _display_zoom_handler_clamp_axis_offset(zoom->offset_x_internal, zoom->display_width, zoom->image_width, zoom->scale);
  zoom->offset_y_internal =
    _display_zoom_handler_clamp_axis_offset(zoom->offset_y_internal, zoom->display_height, zoom->image_height, zoom->scale);

  zoom->offset_x = (int32_t) zoom->offset_x_internal;
  zoom->offset_y = (int32_t) zoom->offset_y_internal;
}

static void _display_zoom_handler_center(DisplayZoomHandler* zoom) {
  MD_CHECK_NULL_ARGUMENT(zoom);

  if (zoom->display_width == 0 || zoom->display_height == 0 || zoom->image_width == 0 || zoom->image_height == 0)
    return;

  zoom->offset_x_internal = ((float) (zoom->image_width - 1) - (float) _map_screen_to_image(zoom->display_width - 1, zoom->scale)) * 0.5f;
  zoom->offset_y_internal = ((float) (zoom->image_height - 1) - (float) _map_screen_to_image(zoom->display_height - 1, zoom->scale)) * 0.5f;
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

  const bool image_size_changed = (zoom->image_width != width) || (zoom->image_height != height);

  zoom->image_width  = width;
  zoom->image_height = height;
  _display_zoom_handler_recalculate_scale_bounds(zoom);

  if (image_size_changed) {
    zoom->scale                  = zoom->min_scale;
    zoom->mouse_wheel_accumulate = (float) zoom->scale;
    _display_zoom_handler_center(zoom);
  }

  _display_zoom_handler_update_offset(zoom);
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

  if (mouse->right_down == false || mouse->right_phase == MOUSE_PHASE_PRESSED)
    return;

  if (mouse->x_motion == 0.0f && mouse->y_motion == 0.0f)
    return;

  float scale_factor = (zoom->scale > 0) ? (1.0f / (1u << zoom->scale)) : (float) (1u << -zoom->scale);

  zoom->offset_x_internal -= mouse->x_motion * scale_factor;
  zoom->offset_y_internal -= mouse->y_motion * scale_factor;
}

void display_zoom_handler_update(DisplayZoomHandler* zoom, MouseState* mouse, bool ctrl_down) {
  MD_CHECK_NULL_ARGUMENT(zoom);
  MD_CHECK_NULL_ARGUMENT(mouse);

  if (mouse->right_phase == MOUSE_PHASE_PRESSED && ctrl_down) {
    _display_zoom_handler_center(zoom);
  }
  else if (ctrl_down == false) {
    _display_zoom_handler_handle_swipe(zoom, mouse);
  }

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
