#include "button.h"

#include "display.h"
#include "ui_renderer_utils.h"
#include "window.h"

static void _element_button_render_func(Element* button, Display* display) {
  ElementButtonData* data = (ElementButtonData*) &button->data;

  uint32_t color = (data->is_pressed) ? data->press_color : ((data->is_hovered) ? data->hover_color : data->color);

  Color256 color256        = color256_set_1(color);
  Color256 mask_low16      = color256_set_1(0x00FF00FF);
  Color256 mask_high16     = color256_set_1(0xFF00FF00);
  Color256 mask_add        = color256_set_1(0x00800080);
  Color256 mask_full_alpha = color256_set_1(0x000000FF);

  const uint32_t size = display->ui_renderer->shape_mask_size[data->shape_size_id];
  const uint8_t* src  = display->ui_renderer->disk_mask[data->shape_size_id];

  uint8_t* dst = display->buffer + 4 * button->x + button->y * display->ld;

  const uint32_t cols = size >> UI_RENDERER_STRIDE_LOG;
  const uint32_t rows = size;

  for (uint32_t row = 0; row < rows; row++) {
    for (uint32_t col = 0; col < cols; col++) {
      Color256 disk_mask = color256_load(src + col * 32);
      Color256 base      = color256_load(dst + col * 32);

      color256_store(dst + col * 32, color256_alpha_blend(color256, base, disk_mask, mask_low16, mask_high16, mask_add, mask_full_alpha));
    }

    src = src + 4 * size;
    dst = dst + display->ld;
  }
}

bool element_button(Window* window, Display* display, ElementButtonArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element button;

  button.type        = ELEMENT_TYPE_BUTTON;
  button.render_func = _element_button_render_func;

  ElementButtonData* data = (ElementButtonData*) &button.data;

  ElementMouseResult mouse_result;
  element_apply_context(&button, context, &args.size, display, &mouse_result);

  data->shape       = args.shape;
  data->color       = args.color;
  data->hover_color = args.hover_color;
  data->press_color = args.press_color;
  data->is_hovered  = mouse_result.is_hovered;
  data->is_pressed  = mouse_result.is_pressed;

  data->shape_size_id = 0;

  for (uint32_t size_id = 0; size_id < SHAPE_MASK_COUNT; size_id++) {
    const uint32_t size = display->ui_renderer->shape_mask_size[size_id];

    if (size <= button.width && size <= button.height) {
      data->shape_size_id = size_id;
    }
  }

  LUM_FAILURE_HANDLE(array_push(&window->element_queue, &button));

  return mouse_result.is_clicked;
}
