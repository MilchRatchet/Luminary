#ifndef MANDARIN_DUCK_UI_RENDERER_H
#define MANDARIN_DUCK_UI_RENDERER_H

#include "utils.h"
#include "window.h"

#define UI_RENDERER_STRIDE 8
#define UI_RENDERER_STRIDE_LOG 3
#define UI_RENDERER_STRIDE_BYTES (UI_RENDERER_STRIDE * 4)

enum UIRendererWorkSize {
  UI_RENDERER_WORK_SIZE_32BIT,
  UI_RENDERER_WORK_SIZE_128BIT,
  UI_RENDERER_WORK_SIZE_256BIT,
  UI_RENDERER_WORK_SIZE_COUNT
} typedef UIRendererWorkSize;

static const uint32_t UIRendererWorkSizeStrideLog[UI_RENDERER_WORK_SIZE_COUNT] =
  {[UI_RENDERER_WORK_SIZE_32BIT] = 0, [UI_RENDERER_WORK_SIZE_128BIT] = 2, [UI_RENDERER_WORK_SIZE_256BIT] = 3};

static const uint32_t UIRendererWorkSizeStride[UI_RENDERER_WORK_SIZE_COUNT] =
  {[UI_RENDERER_WORK_SIZE_32BIT] = 1 << 0, [UI_RENDERER_WORK_SIZE_128BIT] = 1 << 2, [UI_RENDERER_WORK_SIZE_256BIT] = 1 << 3};

static const uint32_t UIRendererWorkSizeStrideBytes[UI_RENDERER_WORK_SIZE_COUNT] =
  {[UI_RENDERER_WORK_SIZE_32BIT] = 4 << 0, [UI_RENDERER_WORK_SIZE_128BIT] = 4 << 2, [UI_RENDERER_WORK_SIZE_256BIT] = 4 << 3};

enum UIRendererBackgroundMode {
  UI_RENDERER_BACKGROUND_MODE_OPAQUE          = 0,
  UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT = 1,
  UI_RENDERER_BACKGROUND_MODE_TRANSPARENT     = 2
} typedef UIRendererBackgroundMode;

#define SHAPE_MASK_COUNT 3

struct UIRenderer {
  uint8_t* block_mask[UI_RENDERER_WORK_SIZE_COUNT];
  uint8_t* block_mask_border[UI_RENDERER_WORK_SIZE_COUNT];
  uint32_t block_mask_size[UI_RENDERER_WORK_SIZE_COUNT];
  uint8_t* disk_mask[SHAPE_MASK_COUNT];
  uint8_t* circle_mask[SHAPE_MASK_COUNT];
  uint32_t shape_mask_size[SHAPE_MASK_COUNT];
} typedef UIRenderer;

void ui_renderer_create(UIRenderer** renderer);
void ui_renderer_create_window_background(UIRenderer* renderer, Display* display, Window* window);
void ui_renderer_render_window(UIRenderer* renderer, Display* display, Window* window);
void ui_renderer_render_rounded_box(
  UIRenderer* renderer, Display* display, uint32_t width, uint32_t height, uint32_t x, uint32_t y, uint32_t rounding_size,
  uint32_t border_color, uint32_t background_color, UIRendererBackgroundMode background_mode);
void ui_renderer_render_display_corners(UIRenderer* renderer, Display* display);
void ui_renderer_render_crosshair(UIRenderer* renderer, Display* display, uint32_t x, uint32_t y);
void ui_renderer_destroy(UIRenderer** renderer);

#endif /* MANDARIN_DUCK_UI_RENDERER_H */
