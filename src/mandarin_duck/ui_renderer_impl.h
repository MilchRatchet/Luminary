#ifndef MANDARIN_DUCK_UI_RENDERER_IMPL_H
#define MANDARIN_DUCK_UI_RENDERER_IMPL_H

#include "ui_renderer.h"
#include "ui_renderer_utils.h"
#include "utils.h"

inline Color256 color256_load_generic(const uint8_t* ptr, const UIRendererWorkSize work_size) {
  switch (work_size) {
    case UI_RENDERER_WORK_SIZE_32BIT:
    default:
      return color256_load_si32(ptr);
    case UI_RENDERER_WORK_SIZE_128BIT:
      return color256_load_si128(ptr);
    case UI_RENDERER_WORK_SIZE_256BIT:
      return color256_load(ptr);
  }
}

inline void color256_store_generic(uint8_t* ptr, const Color256 a, const UIRendererWorkSize work_size) {
  switch (work_size) {
    case UI_RENDERER_WORK_SIZE_32BIT:
    default:
      color256_store_si32(ptr, a);
      break;
    case UI_RENDERER_WORK_SIZE_128BIT:
      color256_store_si128(ptr, a);
      break;
    case UI_RENDERER_WORK_SIZE_256BIT:
      color256_store(ptr, a);
      break;
  }
}

struct UIRendererRoundedBoxArgs {
  uint8_t* dst;
  const uint8_t* src;

  uint32_t dst_ld;
  uint32_t src_ld;

  uint32_t cols;
  uint32_t rows;
  uint32_t height_clip;

  uint32_t row;
  uint32_t shape_row;

  bool use_background;
  bool opaque_background;
  uint32_t background_color;

  bool use_border;
  uint32_t border_color;

  const uint8_t* disk_mask;
  const uint8_t* circle_mask;
  uint32_t shape_mask_ld;
  uint32_t shape_mask_half_cols;
  uint32_t shape_mask_half_size;
} typedef UIRendererRoundedBoxArgs;

////////////////////////////////////////////////////////////////////
// Rounded Box Implementation
////////////////////////////////////////////////////////////////////

inline void _ui_renderer_render_rounded_box_corner(
  UIRendererRoundedBoxArgs* args, uint32_t col, uint32_t shape_col, const UIRendererWorkSize work_size) {
  Color256 border          = color256_set_1(args->border_color);
  Color256 background      = color256_set_1(args->background_color);
  Color256 mask_low16      = color256_set_1(0x00FF00FF);
  Color256 mask_high16     = color256_set_1(0xFF00FF00);
  Color256 mask_add        = color256_set_1(0x00800080);
  Color256 mask_full_alpha = color256_set_1(0x000000FF);

  const uint32_t col_offset       = col * UIRendererWorkSizeStrideBytes[work_size];
  const uint32_t shape_col_offset = shape_col * UIRendererWorkSizeStrideBytes[work_size];

  Color256 disk_left = color256_load_generic(args->disk_mask + args->shape_row * args->shape_mask_ld + shape_col_offset, work_size);
  Color256 base_dst  = color256_load_generic(args->dst + col_offset, work_size);
  Color256 base_src  = color256_load_generic(args->src + col_offset, work_size);

  if (args->use_background) {
    base_src = (args->opaque_background) ? background : color256_avg8(base_src, background);
  }

  Color256 base = color256_alpha_blend(base_src, base_dst, disk_left, mask_low16, mask_high16, mask_add, mask_full_alpha);

  if (args->use_border) {
    Color256 circle_left = color256_load_generic(args->circle_mask + args->shape_row * args->shape_mask_ld + shape_col_offset, work_size);
    base                 = color256_alpha_blend(border, base, circle_left, mask_low16, mask_high16, mask_add, mask_full_alpha);
  }

  color256_store_generic(args->dst + col_offset, base, work_size);
}

inline void _ui_renderer_render_rounded_box_top_center(UIRendererRoundedBoxArgs* args, uint32_t col, const UIRendererWorkSize work_size) {
  Color256 border     = color256_set_1(args->border_color);
  Color256 background = color256_set_1(args->background_color);

  const uint32_t col_offset = col * UIRendererWorkSizeStrideBytes[work_size];

  Color256 base;
  if (args->use_border) {
    base = border;
  }
  else {
    base = color256_load_generic(args->src + col_offset, work_size);

    if (args->use_background) {
      base = (args->opaque_background) ? background : color256_avg8(base, background);
    }
  }

  color256_store_generic(args->dst + col_offset, base, work_size);
}

inline void _ui_renderer_render_rounded_box_center(UIRendererRoundedBoxArgs* args, uint32_t col, const UIRendererWorkSize work_size) {
  Color256 background = color256_set_1(args->background_color);

  const uint32_t col_offset = col * UIRendererWorkSizeStrideBytes[work_size];

  Color256 base_src = color256_load_generic(args->src + col_offset, work_size);

  if (args->use_background) {
    base_src = (args->opaque_background) ? background : color256_avg8(base_src, background);
  }

  color256_store_generic(args->dst + col_offset, base_src, work_size);
}

inline void _ui_renderer_render_rounded_box_left_right(
  UIRendererRoundedBoxArgs* args, uint32_t col, bool is_left, const UIRendererWorkSize work_size) {
  Color256 border          = color256_set_1(args->border_color);
  Color256 background      = color256_set_1(args->background_color);
  Color256 mask_low16      = color256_set_1(0x00FF00FF);
  Color256 mask_high16     = color256_set_1(0xFF00FF00);
  Color256 mask_add        = color256_set_1(0x00800080);
  Color256 mask_full_alpha = color256_set_1(0x000000FF);

  const uint32_t col_offset = col * UIRendererWorkSizeStrideBytes[work_size];

  Color256 base_src = color256_load_generic(args->src + col_offset, work_size);

  if (args->use_background) {
    base_src = (args->opaque_background) ? background : color256_avg8(base_src, background);
  }

  if (args->use_border) {
    Color256 border_mask = color256_set(0, 0, 0, 0, 0, 0, 0, 0xFF);
    if (!is_left) {
      switch (work_size) {
        case UI_RENDERER_WORK_SIZE_32BIT:
        default:
          break;
        case UI_RENDERER_WORK_SIZE_128BIT:
          border_mask = color256_set(0, 0, 0, 0, 0xFF, 0, 0, 0);
          break;
        case UI_RENDERER_WORK_SIZE_256BIT:
          border_mask = color256_set(0xFF, 0, 0, 0, 0, 0, 0, 0);
          break;
      }
    }

    base_src = color256_alpha_blend(border, base_src, border_mask, mask_low16, mask_high16, mask_add, mask_full_alpha);
  }

  color256_store_generic(args->dst + col_offset, base_src, work_size);
}

inline void _ui_renderer_render_rounded_box_row_kind_1(UIRendererRoundedBoxArgs* args, const UIRendererWorkSize work_size) {
  if (args->row >= args->height_clip)
    return;

  uint32_t col       = 0;
  uint32_t shape_col = 0;

  for (; col < args->shape_mask_half_cols; col++) {
    _ui_renderer_render_rounded_box_corner(args, col, shape_col, work_size);

    shape_col++;
  }

  for (; col < args->cols - args->shape_mask_half_cols; col++) {
    _ui_renderer_render_rounded_box_top_center(args, col, work_size);
  }

  for (; col < args->cols; col++) {
    _ui_renderer_render_rounded_box_corner(args, col, shape_col, work_size);

    shape_col++;
  }

  args->row++;
  args->shape_row++;
  args->dst = args->dst + args->dst_ld;
  args->src = args->src + args->src_ld;
}

inline void _ui_renderer_render_rounded_box_row_kind_2(UIRendererRoundedBoxArgs* args, const UIRendererWorkSize work_size) {
  const uint32_t end_row = args->row + args->shape_mask_half_size - 1;

  for (; args->row < end_row && args->row < args->height_clip; args->row++) {
    uint32_t col       = 0;
    uint32_t shape_col = 0;

    for (; col < args->shape_mask_half_cols; col++) {
      _ui_renderer_render_rounded_box_corner(args, col, shape_col, work_size);

      shape_col++;
    }

    for (; col < args->cols - args->shape_mask_half_cols; col++) {
      _ui_renderer_render_rounded_box_center(args, col, work_size);
    }

    for (; col < args->cols; col++) {
      _ui_renderer_render_rounded_box_corner(args, col, shape_col, work_size);

      shape_col++;
    }

    args->shape_row++;
    args->dst = args->dst + args->dst_ld;
    args->src = args->src + args->src_ld;
  }
}

inline void _ui_renderer_render_rounded_box_row_kind_3(UIRendererRoundedBoxArgs* args, const UIRendererWorkSize work_size) {
  for (; args->row < args->rows - args->shape_mask_half_size && args->row < args->height_clip; args->row++) {
    uint32_t col = 0;

    for (; col < 1; col++) {
      _ui_renderer_render_rounded_box_left_right(args, col, true, work_size);
    }

    for (; col < args->cols - 1; col++) {
      _ui_renderer_render_rounded_box_center(args, col, work_size);
    }

    for (; col < args->cols; col++) {
      _ui_renderer_render_rounded_box_left_right(args, col, false, work_size);
    }

    args->dst = args->dst + args->dst_ld;
    args->src = args->src + args->src_ld;
  }
}

static void _ui_renderer_render_rounded_box(
  UIRenderer* renderer, uint32_t width, uint32_t height, const uint8_t* src, uint32_t src_x, uint32_t src_y, uint32_t src_ld, uint8_t* dst,
  uint32_t dst_x, uint32_t dst_y, uint32_t dst_ld, uint32_t rounding_size, uint32_t height_clip, uint32_t border_color,
  uint32_t background_color, UIRendererBackgroundMode background_mode) {
  UIRendererWorkSize work_size = UI_RENDERER_WORK_SIZE_32BIT;
  if ((width & (UIRendererWorkSizeStride[UI_RENDERER_WORK_SIZE_256BIT] - 1)) == 0) {
    work_size = UI_RENDERER_WORK_SIZE_256BIT;
  }
  else if ((width & (UIRendererWorkSizeStride[UI_RENDERER_WORK_SIZE_128BIT] - 1)) == 0) {
    work_size = UI_RENDERER_WORK_SIZE_128BIT;
  }

  const uint32_t cols = width >> UIRendererWorkSizeStrideLog[work_size];
  const uint32_t rows = height;

  uint32_t shape_mask_size = rounding_size;

  uint32_t shape_mask_size_id = 0xFFFFFFFF;
  for (uint32_t size_id = 0; size_id < SHAPE_MASK_COUNT; size_id++) {
    const uint32_t size = renderer->shape_mask_size[size_id];

    if (size == shape_mask_size) {
      shape_mask_size_id = size_id;
      break;
    }
  }

  shape_mask_size =
    (shape_mask_size_id != 0xFFFFFFFF) ? renderer->shape_mask_size[shape_mask_size_id] : renderer->block_mask_size[work_size];

  const uint8_t* disk_mask = (shape_mask_size_id != 0xFFFFFFFF) ? renderer->disk_mask[shape_mask_size_id] : renderer->block_mask[work_size];
  const uint8_t* circle_mask =
    (shape_mask_size_id != 0xFFFFFFFF) ? renderer->circle_mask[shape_mask_size_id] : renderer->block_mask_border[work_size];

  const uint32_t shape_mask_half_size = shape_mask_size >> 1;
  const uint32_t shape_mask_ld        = shape_mask_size * sizeof(LuminaryARGB8);

  const uint32_t shape_mask_half_cols = shape_mask_half_size >> UIRendererWorkSizeStrideLog[work_size];

  if (cols < shape_mask_half_cols)
    return;

  UIRendererRoundedBoxArgs args = {
    .dst                  = dst + sizeof(LuminaryARGB8) * dst_x + dst_y * dst_ld,
    .src                  = src + sizeof(LuminaryARGB8) * src_x + src_y * src_ld,
    .dst_ld               = dst_ld,
    .src_ld               = src_ld,
    .cols                 = cols,
    .rows                 = rows,
    .height_clip          = height_clip,
    .row                  = 0,
    .shape_row            = 0,
    .use_background       = background_mode != UI_RENDERER_BACKGROUND_MODE_TRANSPARENT,
    .opaque_background    = background_mode == UI_RENDERER_BACKGROUND_MODE_OPAQUE,
    .background_color     = background_color,
    .use_border           = (border_color != 0),
    .border_color         = border_color,
    .disk_mask            = disk_mask,
    .circle_mask          = circle_mask,
    .shape_mask_ld        = shape_mask_ld,
    .shape_mask_half_cols = shape_mask_half_cols,
    .shape_mask_half_size = shape_mask_half_size};

  switch (work_size) {
    case UI_RENDERER_WORK_SIZE_32BIT:
    default:
      _ui_renderer_render_rounded_box_row_kind_1(&args, UI_RENDERER_WORK_SIZE_32BIT);
      _ui_renderer_render_rounded_box_row_kind_2(&args, UI_RENDERER_WORK_SIZE_32BIT);
      _ui_renderer_render_rounded_box_row_kind_3(&args, UI_RENDERER_WORK_SIZE_32BIT);
      _ui_renderer_render_rounded_box_row_kind_2(&args, UI_RENDERER_WORK_SIZE_32BIT);
      _ui_renderer_render_rounded_box_row_kind_1(&args, UI_RENDERER_WORK_SIZE_32BIT);
      break;
    case UI_RENDERER_WORK_SIZE_128BIT:
      _ui_renderer_render_rounded_box_row_kind_1(&args, UI_RENDERER_WORK_SIZE_128BIT);
      _ui_renderer_render_rounded_box_row_kind_2(&args, UI_RENDERER_WORK_SIZE_128BIT);
      _ui_renderer_render_rounded_box_row_kind_3(&args, UI_RENDERER_WORK_SIZE_128BIT);
      _ui_renderer_render_rounded_box_row_kind_2(&args, UI_RENDERER_WORK_SIZE_128BIT);
      _ui_renderer_render_rounded_box_row_kind_1(&args, UI_RENDERER_WORK_SIZE_128BIT);
      break;
    case UI_RENDERER_WORK_SIZE_256BIT:
      _ui_renderer_render_rounded_box_row_kind_1(&args, UI_RENDERER_WORK_SIZE_256BIT);
      _ui_renderer_render_rounded_box_row_kind_2(&args, UI_RENDERER_WORK_SIZE_256BIT);
      _ui_renderer_render_rounded_box_row_kind_3(&args, UI_RENDERER_WORK_SIZE_256BIT);
      _ui_renderer_render_rounded_box_row_kind_2(&args, UI_RENDERER_WORK_SIZE_256BIT);
      _ui_renderer_render_rounded_box_row_kind_1(&args, UI_RENDERER_WORK_SIZE_256BIT);
      break;
  }
}

#endif /* MANDARIN_DUCK_UI_RENDERER_IMPL_H */
