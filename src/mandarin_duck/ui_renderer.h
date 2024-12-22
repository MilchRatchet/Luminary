#ifndef MANDARIN_DUCK_UI_RENDERER_H
#define MANDARIN_DUCK_UI_RENDERER_H

#include "utils.h"
#include "window.h"

#define UI_RENDERER_STRIDE 8
#define UI_RENDERER_STRIDE_LOG 3
#define UI_RENDERER_STRIDE_BYTES (UI_RENDERER_STRIDE * 4)

#define SHAPE_MASK_COUNT 3

struct UIRenderer {
  uint8_t* block_mask;
  uint8_t* block_mask_border;
  uint32_t block_mask_size;
  uint8_t* disk_mask[SHAPE_MASK_COUNT];
  uint8_t* circle_mask[SHAPE_MASK_COUNT];
  uint32_t shape_mask_size[SHAPE_MASK_COUNT];
} typedef UIRenderer;

void ui_renderer_create(UIRenderer** renderer);
void ui_renderer_create_window_background(UIRenderer* renderer, Display* display, Window* window);
void ui_renderer_render_window(UIRenderer* renderer, Display* display, Window* window);
void ui_renderer_render_rounded_box(
  UIRenderer* renderer, Display* display, uint32_t width, uint32_t height, uint32_t x, uint32_t y, uint32_t rounding_size);
void ui_renderer_destroy(UIRenderer** renderer);

#endif /* MANDARIN_DUCK_UI_RENDERER_H */
