#include "render_region.h"

#include "display.h"
#include "ui_renderer.h"

#define RENDER_REGION_VERTEX_SIZE 8

void render_region_create(RenderRegion** region) {
  MD_CHECK_NULL_ARGUMENT(region);

  LUM_FAILURE_HANDLE(host_malloc(region, sizeof(RenderRegion)));
  memset(*region, 0, sizeof(RenderRegion));
}

void render_region_handler_set_display_size(RenderRegion* region, uint32_t width, uint32_t height) {
  MD_CHECK_NULL_ARGUMENT(region);

  region->display_width  = width;
  region->display_height = height;
}

static void _render_region_compute_size(RenderRegion* region) {
  MD_CHECK_NULL_ARGUMENT(region);

  region->x      = fminf(region->x_internal, region->x_internal + region->width_internal);
  region->y      = fminf(region->y_internal, region->y_internal + region->height_internal);
  region->width  = fabsf(region->width_internal);
  region->height = fabsf(region->height_internal);

  region->width  = fminf(region->width, region->display_width);
  region->height = fminf(region->height, region->display_height);

  region->x = fminf(region->x, region->display_width - region->width);
  region->y = fminf(region->y, region->display_height - region->height);
  region->x = fmaxf(region->x, 0.0f);
  region->y = fmaxf(region->y, 0.0f);
}

static void _render_region_commit(RenderRegion* region, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(region);
  MD_CHECK_NULL_ARGUMENT(host);

  if (region->is_selecting) {
    _render_region_compute_size(region);

    region->is_active = (region->width >= 1.0f) && (region->height >= 1.0f);

    LuminaryRendererSettings settings;
    LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

    if (region->is_active) {
      settings.region_x      = region->x / region->display_width;
      settings.region_y      = region->y / region->display_height;
      settings.region_width  = ceilf(region->width) / region->display_width;
      settings.region_height = ceilf(region->height) / region->display_height;
    }
    else {
      settings.region_x      = 0.0f;
      settings.region_y      = 0.0f;
      settings.region_width  = 1.0f;
      settings.region_height = 1.0f;
    }

    LUM_FAILURE_HANDLE(luminary_host_set_settings(host, &settings));
  }

  region->is_selecting = false;
}

void render_region_handle_inputs(
  RenderRegion* region, Display* display, LuminaryHost* host, MouseState* mouse_state, KeyboardState* keyboard_state) {
  MD_CHECK_NULL_ARGUMENT(region);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);
  MD_CHECK_NULL_ARGUMENT(mouse_state);
  MD_CHECK_NULL_ARGUMENT(keyboard_state);

  if (keyboard_state->keys[SDL_SCANCODE_LCTRL].down == false)
    return;

  if (mouse_state->down == false) {
    _render_region_commit(region, host);
    return;
  }

  const DisplayZoomHandler* zoom = display->zoom_handler;

  if (region->is_selecting == false) {
    uint32_t x, y;
    display_zoom_handler_screen_to_image(zoom, mouse_state->x, mouse_state->y, &x, &y);

    region->x_internal      = x;
    region->y_internal      = y;
    region->width_internal  = 0.0f;
    region->height_internal = 0.0f;

    region->is_selecting = true;
  }

  const float scale = 1.0f / (1u << zoom->scale);

  region->width_internal += mouse_state->x_motion * scale;
  region->height_internal += mouse_state->y_motion * scale;

  _render_region_compute_size(region);
}

#define RENDER_REGION_BORDER_LENGTH 32

void render_region_render(RenderRegion* region, Display* display, UIRenderer* renderer) {
  MD_CHECK_NULL_ARGUMENT(region);
  MD_CHECK_NULL_ARGUMENT(renderer);

  if (region->is_active == false && region->is_selecting == false)
    return;

  const DisplayZoomHandler* zoom = display->zoom_handler;

  const uint32_t color = (region->is_selecting) ? MD_COLOR_ACCENT_LIGHT_1 : MD_COLOR_WHITE;

  uint32_t x, y;
  display_zoom_handler_image_to_screen(zoom, region->x, region->y, &x, &y);

  const float region_x1 = ceilf(region->x + region->width);
  const float region_y1 = ceilf(region->y + region->height);

  uint32_t x1, y1;
  display_zoom_handler_image_to_screen(zoom, region_x1, region_y1, &x1, &y1);

  const uint32_t width  = x1 - x;
  const uint32_t height = y1 - y;

  ui_renderer_render_rounded_box(renderer, display, width, height, x, y, 0, color, 0, UI_RENDERER_BACKGROUND_MODE_TRANSPARENT);
}

void render_region_destroy(RenderRegion** region) {
  MD_CHECK_NULL_ARGUMENT(region);
  MD_CHECK_NULL_ARGUMENT(*region);

  LUM_FAILURE_HANDLE(host_free(region));
}
