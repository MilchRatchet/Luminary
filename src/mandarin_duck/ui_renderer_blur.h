#ifndef MANDARIN_DUCK_UI_RENDERER_BLUR_H
#define MANDARIN_DUCK_UI_RENDERER_BLUR_H

#include "ui_renderer.h"
#include "utils.h"

#define BLUR_MAX_SUPPORTED_DIM (7680)

struct UIRendererBlurRows {
  const uint8_t* reduced;
  uint32_t cols;
  uint32_t rows;
  size_t row_bytes;
  uint8_t expanded[2][BLUR_MAX_SUPPORTED_DIM * sizeof(LuminaryARGB8)];
  uint8_t blended[BLUR_MAX_SUPPORTED_DIM * sizeof(LuminaryARGB8)];
} typedef UIRendererBlurRows;

size_t ui_renderer_blur_buffer_size(uint32_t width, uint32_t height);
bool ui_renderer_blur_rows_init(UIRendererBlurRows* blur, Display* display, Window* window, uint32_t height);

/* Read consecutive rows starting at zero. Consume each row before requesting the next. */
const uint8_t* ui_renderer_blur_rows_get(UIRendererBlurRows* blur, uint32_t row);

#endif /* MANDARIN_DUCK_UI_RENDERER_BLUR_H */
