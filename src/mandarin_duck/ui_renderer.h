#ifndef MANDARIN_DUCK_UI_RENDERER_H
#define MANDARIN_DUCK_UI_RENDERER_H

#include "utils.h"

#define UI_UNIT_SIZE 16

struct UIRenderer {
  uint8_t* circle_mask;
} typedef UIRenderer;

void ui_renderer_create(UIRenderer** renderer);
void ui_renderer_destroy(UIRenderer** renderer);

#endif /* MANDARIN_DUCK_UI_RENDERER_H */
