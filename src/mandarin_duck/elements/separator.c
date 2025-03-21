#include "separator.h"

#include "display.h"
#include "text_renderer.h"

static void _element_separator_render_func(Element* separator, Display* display) {
  ElementSeparatorData* data = (ElementSeparatorData*) &separator->data;

  const uint32_t padding_x = separator->width >> 1;
  const uint32_t padding_y = separator->height >> 1;

  uint32_t text_width;
  text_renderer_render(
    display->text_renderer, display, data->text, TEXT_RENDERER_FONT_REGULAR, MD_COLOR_WHITE, separator->x + padding_x,
    separator->y + padding_y, true, true, true, &text_width);

  const uint32_t line_length = (separator->width - text_width) >> 1;

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, line_length, 2, separator->x, separator->y + padding_y, 0, 0, 0xFF888888,
    UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, line_length, 2, separator->x + line_length + text_width, separator->y + padding_y, 0, 0, 0xFF888888,
    UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);
}

bool element_separator(Window* window, const MouseState* mouse_state, ElementSeparatorArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element text;

  text.type        = ELEMENT_TYPE_SEPARATOR;
  text.render_func = _element_separator_render_func;
  text.hash        = 0;

  ElementSeparatorData* data = (ElementSeparatorData*) &text.data;

  data->size = args.size;

  const size_t text_size = strlen(args.text);

  if (text_size > SEPARATOR_TEXT_SIZE - 1) {
    crash_message("Text is too long.");
  }

  memcpy(data->text, args.text, text_size);
  memset(data->text + text_size, 0, SEPARATOR_TEXT_SIZE - text_size);

  memcpy(window->separator_context_string, args.text, text_size);
  memset(window->separator_context_string + text_size, 0, SEPARATOR_TEXT_SIZE - text_size);

  ElementMouseResult mouse_result;
  element_apply_context(&text, context, &args.size, mouse_state, &mouse_result);

  window->status.received_hover |= mouse_result.is_hovered;
  window->status.received_mouse_action |= mouse_result.is_clicked;

  LUM_FAILURE_HANDLE(array_push(&window->element_queue, &text));

  return mouse_result.is_clicked;
}
