#ifndef MANDARIN_DUCK_RENDER_REGION_H
#define MANDARIN_DUCK_RENDER_REGION_H

#include "keyboard_state.h"
#include "mouse_state.h"
#include "utils.h"

struct Display typedef Display;
struct UIRenderer typedef UIRenderer;

struct RenderRegion {
  uint32_t display_width;
  uint32_t display_height;
  bool is_selecting;
  bool is_active;
  float x_internal;
  float y_internal;
  float width_internal;
  float height_internal;
  float x;
  float y;
  float width;
  float height;
} typedef RenderRegion;

void render_region_create(RenderRegion** region);
void render_region_handler_set_display_size(RenderRegion* region, uint32_t width, uint32_t height);
void render_region_handle_inputs(
  RenderRegion* region, Display* display, LuminaryHost* host, MouseState* mouse_state, KeyboardState* keyboard_state);
void render_region_render(RenderRegion* region, Display* display, UIRenderer* renderer);
void render_region_destroy(RenderRegion** region);

#endif /* MANDARIN_DUCK_RENDER_REGION_H */
