#include "render_region.h"

#include "display.h"
#include "ui_renderer.h"

#define RENDER_REGION_VERTEX_SIZE 8

void render_region_create(RenderRegion** region) {
  MD_CHECK_NULL_ARGUMENT(region);

  LUM_FAILURE_HANDLE(host_malloc(region, sizeof(RenderRegion)));
  memset(*region, 0, sizeof(RenderRegion));

  (*region)->x      = 0.0f;
  (*region)->y      = 0.0f;
  (*region)->width  = 1.0f;
  (*region)->height = 1.0f;
}

static void _render_region_get_vertex(RenderRegion* region, uint32_t vertex, uint32_t width, uint32_t height, uint32_t* x, uint32_t* y) {
  switch (vertex) {
    case RENDER_REGION_VERTEX_TOP_LEFT:
      *x = region->x * width;
      *y = region->y * height;
      break;
    case RENDER_REGION_VERTEX_TOP_RIGHT:
      *x = (region->x + region->width) * width;
      *y = region->y * height;
      break;
    case RENDER_REGION_VERTEX_BOTTOM_LEFT:
      *x = region->x * width;
      *y = (region->y + region->height) * height;
      break;
    case RENDER_REGION_VERTEX_BOTTOM_RIGHT:
      *x = (region->x + region->width) * width;
      *y = (region->y + region->height) * height;
      break;
  }
}

void render_region_handle_inputs(RenderRegion* region, Display* display, LuminaryHost* host, MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(region);
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  if (mouse_state->down == false) {
    region->state = RENDER_REGION_STATE_DEFAULT;
    return;
  }

  uint32_t vertex_x[RENDER_REGION_VERTEX_COUNT];
  uint32_t vertex_y[RENDER_REGION_VERTEX_COUNT];
  for (uint32_t vertex_id = 0; vertex_id < RENDER_REGION_VERTEX_COUNT; vertex_id++) {
    _render_region_get_vertex(region, vertex_id, display->width, display->height, vertex_x + vertex_id, vertex_y + vertex_id);
  }

  switch (region->state) {
    case RENDER_REGION_STATE_DEFAULT: {
      if (mouse_state->phase != MOUSE_PHASE_PRESSED)
        break;

      if (mouse_state->x < region->x * display->width)
        break;

      if (mouse_state->x > (region->x + region->width) * display->width)
        break;

      if (mouse_state->y < region->y * display->height)
        break;

      if (mouse_state->y > (region->y + region->height) * display->height)
        break;

      region->state = RENDER_REGION_STATE_MOVE;

      for (uint32_t vertex_id = 0; vertex_id < RENDER_REGION_VERTEX_COUNT; vertex_id++) {
        if (mouse_state->x + RENDER_REGION_VERTEX_SIZE < vertex_x[vertex_id])
          continue;

        if (mouse_state->x > vertex_x[vertex_id] + RENDER_REGION_VERTEX_SIZE)
          continue;

        if (mouse_state->y + RENDER_REGION_VERTEX_SIZE < vertex_y[vertex_id])
          continue;

        if (mouse_state->y > vertex_y[vertex_id] + RENDER_REGION_VERTEX_SIZE)
          continue;

        region->state           = RENDER_REGION_STATE_VERTEX_MOVE;
        region->selected_vertex = vertex_id;
        break;
      }
    } break;
    case RENDER_REGION_STATE_MOVE: {
      region->x += mouse_state->x_motion / display->width;
      region->y += mouse_state->y_motion / display->height;

      region->x = fminf(1.0f - region->width, fmaxf(0.0f, region->x));
      region->y = fminf(1.0f - region->height, fmaxf(0.0f, region->y));
    } break;
    case RENDER_REGION_STATE_VERTEX_MOVE: {
      const float move_x = mouse_state->x_motion / display->width;
      const float move_y = mouse_state->y_motion / display->height;

      switch (region->selected_vertex) {
        case RENDER_REGION_VERTEX_TOP_LEFT:
          region->x += move_x;
          region->y += move_y;
          region->width -= move_x;
          region->height -= move_y;
          break;
        case RENDER_REGION_VERTEX_TOP_RIGHT:
          region->y += move_y;
          region->width += move_x;
          region->height -= move_y;
          region->width = fminf(1.0f - region->x, fmaxf(0.0f, region->width));
          break;
        case RENDER_REGION_VERTEX_BOTTOM_LEFT:
          region->x += move_x;
          region->width -= move_x;
          region->height += move_y;
          region->height = fminf(1.0f - region->y, fmaxf(0.0f, region->height));
          break;
        case RENDER_REGION_VERTEX_BOTTOM_RIGHT:
          region->width += move_x;
          region->height += move_y;
          region->width  = fminf(1.0f - region->x, fmaxf(0.0f, region->width));
          region->height = fminf(1.0f - region->y, fmaxf(0.0f, region->height));
          break;
        default:
          break;
      }

      region->x      = fminf(1.0f - region->width, fmaxf(0.0f, region->x));
      region->y      = fminf(1.0f - region->height, fmaxf(0.0f, region->y));
      region->width  = fminf(1.0f - region->x, fmaxf(0.0f, region->width));
      region->height = fminf(1.0f - region->y, fmaxf(0.0f, region->height));
    } break;
    default:
      break;
  }

  LuminaryRendererSettings settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

  settings.region_x      = region->x;
  settings.region_y      = region->y;
  settings.region_width  = region->width;
  settings.region_height = region->height;

  LUM_FAILURE_HANDLE(luminary_host_set_settings(host, &settings));
}

#define RENDER_REGION_BORDER_LENGTH 32

void render_region_render(RenderRegion* region, Display* display, UIRenderer* renderer) {
  MD_CHECK_NULL_ARGUMENT(region);
  MD_CHECK_NULL_ARGUMENT(renderer);

  uint32_t vertex_x;
  uint32_t vertex_y;
  _render_region_get_vertex(region, RENDER_REGION_VERTEX_TOP_LEFT, display->width, display->height, &vertex_x, &vertex_y);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_BORDER_LENGTH, RENDER_REGION_VERTEX_SIZE, vertex_x, vertex_y, 0, 0, MD_COLOR_WHITE,
    UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_VERTEX_SIZE, RENDER_REGION_BORDER_LENGTH, vertex_x, vertex_y, 0, 0, MD_COLOR_WHITE,
    UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  _render_region_get_vertex(region, RENDER_REGION_VERTEX_TOP_RIGHT, display->width, display->height, &vertex_x, &vertex_y);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_BORDER_LENGTH, RENDER_REGION_VERTEX_SIZE, vertex_x - RENDER_REGION_BORDER_LENGTH, vertex_y, 0, 0,
    MD_COLOR_WHITE, UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_VERTEX_SIZE, RENDER_REGION_BORDER_LENGTH, vertex_x - RENDER_REGION_VERTEX_SIZE, vertex_y, 0, 0,
    MD_COLOR_WHITE, UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  _render_region_get_vertex(region, RENDER_REGION_VERTEX_BOTTOM_LEFT, display->width, display->height, &vertex_x, &vertex_y);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_BORDER_LENGTH, RENDER_REGION_VERTEX_SIZE, vertex_x, vertex_y - RENDER_REGION_VERTEX_SIZE, 0, 0,
    MD_COLOR_WHITE, UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_VERTEX_SIZE, RENDER_REGION_BORDER_LENGTH, vertex_x, vertex_y - RENDER_REGION_BORDER_LENGTH, 0, 0,
    MD_COLOR_WHITE, UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  _render_region_get_vertex(region, RENDER_REGION_VERTEX_BOTTOM_RIGHT, display->width, display->height, &vertex_x, &vertex_y);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_BORDER_LENGTH, RENDER_REGION_VERTEX_SIZE, vertex_x - RENDER_REGION_BORDER_LENGTH,
    vertex_y - RENDER_REGION_VERTEX_SIZE, 0, 0, MD_COLOR_WHITE, UI_RENDERER_BACKGROUND_MODE_OPAQUE);

  ui_renderer_render_rounded_box(
    renderer, display, RENDER_REGION_VERTEX_SIZE, RENDER_REGION_BORDER_LENGTH, vertex_x - RENDER_REGION_VERTEX_SIZE,
    vertex_y - RENDER_REGION_BORDER_LENGTH, 0, 0, MD_COLOR_WHITE, UI_RENDERER_BACKGROUND_MODE_OPAQUE);
}

void render_region_destroy(RenderRegion** region) {
  MD_CHECK_NULL_ARGUMENT(region);
  MD_CHECK_NULL_ARGUMENT(*region);

  LUM_FAILURE_HANDLE(host_free(region));
}
