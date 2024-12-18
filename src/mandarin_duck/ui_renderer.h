#ifndef MANDARIN_DUCK_UI_RENDERER_H
#define MANDARIN_DUCK_UI_RENDERER_H

#include "utils.h"
#include "window.h"

#define UI_UNIT_SIZE 16
#define UI_RENDERER_STRIDE 8
#define UI_RENDERER_STRIDE_BYTES (UI_RENDERER_STRIDE * 4)

struct UIRenderer {
  uint8_t* disk_mask;
  uint8_t* circle_mask;
} typedef UIRenderer;

void ui_renderer_create(UIRenderer** renderer);
void ui_renderer_create_window_background(UIRenderer* renderer, Display* display, Window* window);
void ui_renderer_render_window(UIRenderer* renderer, Display* display, Window* window);
void ui_renderer_destroy(UIRenderer** renderer);

#endif /* MANDARIN_DUCK_UI_RENDERER_H */
