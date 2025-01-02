#include "text.h"

#include "display.h"
#include "text_renderer.h"

static void _element_text_render_func(Element* text, Display* display) {
  ElementTextData* data = (ElementTextData*) &text->data;

  if (data->highlighted) {
    ui_renderer_render_rounded_box(
      display->ui_renderer, display, text->width + 2 * data->highlight_padding, text->height, text->x - data->highlight_padding, text->y, 0,
      0, 0xFF998890, UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);
  }

  const uint32_t padding_x = data->center_x ? text->width >> 1 : 0;
  const uint32_t padding_y = data->center_y ? text->height >> 1 : 0;

  text_renderer_render(
    display->text_renderer, display, data->text, TEXT_RENDERER_FONT_REGULAR, text->x + padding_x, text->y + padding_y, data->center_x,
    data->center_y, data->cache_text, (uint32_t*) 0);
}

bool element_text(Window* window, Display* display, const MouseState* mouse_state, ElementTextArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element text;

  text.type        = ELEMENT_TYPE_TEXT;
  text.render_func = _element_text_render_func;
  text.hash        = 0;

  ElementTextData* data = (ElementTextData*) &text.data;

  data->color      = args.color;
  data->size       = args.size;
  data->center_x   = args.center_x;
  data->center_y   = args.center_y;
  data->cache_text = args.cache_text;

  const size_t text_size = strlen(args.text);

  if (text_size > 256) {
    crash_message("Text is too long.");
  }

  memcpy(data->text, args.text, text_size);
  memset(data->text + text_size, 0, 256 - text_size);

  if (args.auto_size) {
    text_renderer_compute_size(
      display->text_renderer, data->text, TEXT_RENDERER_FONT_REGULAR, data->cache_text, &args.size.width, &args.size.height);
  }

  ElementMouseResult mouse_result;
  element_apply_context(&text, context, &args.size, mouse_state, &mouse_result);

  data->highlighted       = args.highlighting && mouse_result.is_hovered;
  data->highlight_padding = context->padding;

  LUM_FAILURE_HANDLE(array_push(&window->element_queue, &text));

  return mouse_result.is_clicked;
}
