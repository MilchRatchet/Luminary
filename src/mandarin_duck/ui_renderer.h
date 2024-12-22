#ifndef MANDARIN_DUCK_UI_RENDERER_H
#define MANDARIN_DUCK_UI_RENDERER_H

#include "utils.h"
#include "window.h"

#define UI_RENDERER_STRIDE 8
#define UI_RENDERER_STRIDE_LOG 3
#define UI_RENDERER_STRIDE_BYTES (UI_RENDERER_STRIDE * 4)

#define SHAPE_MASK_COUNT 4

struct UIRenderer {
  uint8_t* disk_mask[SHAPE_MASK_COUNT];
  uint8_t* circle_mask[SHAPE_MASK_COUNT];
  uint32_t shape_mask_size[SHAPE_MASK_COUNT];
} typedef UIRenderer;

void ui_renderer_create(UIRenderer** renderer);
void ui_renderer_create_window_background(UIRenderer* renderer, Display* display, Window* window);
void ui_renderer_render_window(UIRenderer* renderer, Display* display, Window* window);
void ui_renderer_destroy(UIRenderer** renderer);

#endif /* MANDARIN_DUCK_UI_RENDERER_H */
