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

static void _render_region_get_vertex(RenderRegion* region, uint32_t vertex, uint32_t* x, uint32_t* y) {
  switch (vertex) {
    case RENDER_REGION_VERTEX_TOP_LEFT:
      *x = region->x;
      *y = region->y;
      break;
    case RENDER_REGION_VERTEX_TOP_RIGHT:
      *x = region->x + region->width;
      *y = region->y;
      break;
    case RENDER_REGION_VERTEX_BOTTOM_LEFT:
      *x = region->x;
      *y = region->y + region->height;
      break;
    case RENDER_REGION_VERTEX_BOTTOM_RIGHT:
      *x = region->x + region->width;
      *y = region->y + region->height;
      break;
  }
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
      settings.region_width  = region->width / region->display_width;
      settings.region_height = region->height / region->display_height;
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

void render_region_handle_inputs(RenderRegion* region, LuminaryHost* host, MouseState* mouse_state, KeyboardState* keyboard_state) {
  MD_CHECK_NULL_ARGUMENT(region);
  MD_CHECK_NULL_ARGUMENT(host);
  MD_CHECK_NULL_ARGUMENT(mouse_state);
  MD_CHECK_NULL_ARGUMENT(keyboard_state);

  if (keyboard_state->keys[SDL_SCANCODE_LCTRL].down == false)
    return;

  if (mouse_state->down == false) {
    _render_region_commit(region, host);
    return;
  }

  if (region->is_selecting == false) {
    region->x_internal      = mouse_state->x;
    region->y_internal      = mouse_state->y;
    region->width_internal  = 0.0f;
    region->height_internal = 0.0f;

    region->is_selecting = true;
  }

  region->width_internal += mouse_state->x_motion;
  region->height_internal += mouse_state->y_motion;

  _render_region_compute_size(region);
}

#define RENDER_REGION_BORDER_LENGTH 32

void render_region_render(RenderRegion* region, Display* display, UIRenderer* renderer) {
  MD_CHECK_NULL_ARGUMENT(region);
  MD_CHECK_NULL_ARGUMENT(renderer);

  if (region->is_active == false && region->is_selecting == false)
    return;

  const uint32_t color = (region->is_selecting) ? MD_COLOR_ACCENT_LIGHT_1 : MD_COLOR_WHITE;

  uint32_t vertex_x;
  uint32_t vertex_y;
  _render_region_get_vertex(region, RENDER_REGION_VERTEX_TOP_LEFT, &vertex_x, &vertex_y);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_BORDER_LENGTH, RENDER_REGION_VERTEX_SIZE, vertex_x, vertex_y, 0, 0, color,
    UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_VERTEX_SIZE, RENDER_REGION_BORDER_LENGTH, vertex_x, vertex_y, 0, 0, color,
    UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  _render_region_get_vertex(region, RENDER_REGION_VERTEX_TOP_RIGHT, &vertex_x, &vertex_y);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_BORDER_LENGTH, RENDER_REGION_VERTEX_SIZE, vertex_x - RENDER_REGION_BORDER_LENGTH, vertex_y, 0, 0,
    color, UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_VERTEX_SIZE, RENDER_REGION_BORDER_LENGTH, vertex_x - RENDER_REGION_VERTEX_SIZE, vertex_y, 0, 0, color,
    UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  _render_region_get_vertex(region, RENDER_REGION_VERTEX_BOTTOM_LEFT, &vertex_x, &vertex_y);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_BORDER_LENGTH, RENDER_REGION_VERTEX_SIZE, vertex_x, vertex_y - RENDER_REGION_VERTEX_SIZE, 0, 0, color,
    UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_VERTEX_SIZE, RENDER_REGION_BORDER_LENGTH, vertex_x, vertex_y - RENDER_REGION_BORDER_LENGTH, 0, 0,
    color, UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  _render_region_get_vertex(region, RENDER_REGION_VERTEX_BOTTOM_RIGHT, &vertex_x, &vertex_y);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_BORDER_LENGTH, RENDER_REGION_VERTEX_SIZE, vertex_x - RENDER_REGION_BORDER_LENGTH,
    vertex_y - RENDER_REGION_VERTEX_SIZE, 0, 0, color, UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_VERTEX_SIZE, RENDER_REGION_BORDER_LENGTH, vertex_x - RENDER_REGION_VERTEX_SIZE,
    vertex_y - RENDER_REGION_BORDER_LENGTH, 0, 0, color, UI_RENDERER_BACKGROUND_MODE_OPAQUE);
}

void render_region_destroy(RenderRegion** region) {
  MD_CHECK_NULL_ARGUMENT(region);
  MD_CHECK_NULL_ARGUMENT(*region);

  LUM_FAILURE_HANDLE(host_free(region));
}
