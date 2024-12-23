#include "slider.h"

#include <stdio.h>
#include <string.h>

#include "display.h"

static void _element_slider_render_float(Element* slider, Display* display) {
  ElementSliderData* data = (ElementSliderData*) &slider->data;

  ui_renderer_render_rounded_box(display->ui_renderer, display, slider->width, slider->height, slider->x, slider->y, 0);

  char text[256];
  sprintf(text, "%.2f", data->data_float);

  SDL_Surface* surface;
  text_renderer_render(display->text_renderer, text, TEXT_RENDERER_FONT_REGULAR, &surface);

  if (surface->h > (int32_t) slider->height) {
    crash_message("Text is taller than the element.");
  }

  const uint32_t padding_x = data->center_x ? (slider->width - surface->w) >> 1 : 0;
  const uint32_t padding_y = data->center_y ? (slider->height - surface->h) >> 1 : 0;

  SDL_Rect src_rect;
  src_rect.x = 0;
  src_rect.y = 0;
  src_rect.w = surface->w;
  src_rect.h = surface->h;

  SDL_Rect dst_rect;
  dst_rect.x = slider->x + padding_x + data->component_padding;
  dst_rect.y = slider->y + padding_y;
  dst_rect.w = surface->w;
  dst_rect.h = surface->h;

  SDL_BlitSurface(surface, &src_rect, display->sdl_surface, &dst_rect);

  SDL_DestroySurface(surface);
}

static void _element_slider_render_uint(Element* slider, Display* display) {
  ElementSliderData* data = (ElementSliderData*) &slider->data;

  ui_renderer_render_rounded_box(display->ui_renderer, display, slider->width, slider->height, slider->x, slider->y, 0);

  char text[256];
  sprintf(text, "%u", data->data_uint);

  SDL_Surface* surface;
  text_renderer_render(display->text_renderer, text, TEXT_RENDERER_FONT_REGULAR, &surface);

  if (surface->h > (int32_t) slider->height) {
    crash_message("Text is taller than the element.");
  }

  const uint32_t padding_x = data->center_x ? (slider->width - surface->w) >> 1 : data->component_padding;
  const uint32_t padding_y = data->center_y ? (slider->height - surface->h) >> 1 : 0;

  SDL_Rect src_rect;
  src_rect.x = 0;
  src_rect.y = 0;
  src_rect.w = surface->w;
  src_rect.h = surface->h;

  SDL_Rect dst_rect;
  dst_rect.x = slider->x + padding_x;
  dst_rect.y = slider->y + padding_y;
  dst_rect.w = surface->w;
  dst_rect.h = surface->h;

  SDL_BlitSurface(surface, &src_rect, display->sdl_surface, &dst_rect);

  SDL_DestroySurface(surface);
}

static void _element_slider_render_vector(Element* slider, Display* display) {
  ElementSliderData* data = (ElementSliderData*) &slider->data;

  float* vec_data = (float*) &data->data_vec3;

  uint32_t x_offset                    = slider->x;
  const uint32_t component_size        = (slider->width - 6 * data->component_padding) / 3;
  const uint32_t component_size_padded = component_size + 2 * data->component_padding;

  for (uint32_t component = 0; component < 3; component++) {
    ui_renderer_render_rounded_box(display->ui_renderer, display, component_size_padded, slider->height, x_offset, slider->y, 0);

    char text[256];
    sprintf(text, "%.2f", vec_data[component]);

    SDL_Surface* surface;
    text_renderer_render(display->text_renderer, text, TEXT_RENDERER_FONT_REGULAR, &surface);

    if (surface->h > (int32_t) slider->height || surface->w > (int32_t) component_size) {
      crash_message("Text is too larger.");
    }

    const uint32_t padding_x = data->center_x ? (component_size_padded - surface->w) >> 1 : data->component_padding;
    const uint32_t padding_y = data->center_y ? (slider->height - surface->h) >> 1 : 0;

    SDL_Rect src_rect;
    src_rect.x = 0;
    src_rect.y = 0;
    src_rect.w = surface->w;
    src_rect.h = surface->h;

    SDL_Rect dst_rect;
    dst_rect.x = x_offset + padding_x;
    dst_rect.y = slider->y + padding_y;
    dst_rect.w = surface->w;
    dst_rect.h = surface->h;

    SDL_BlitSurface(surface, &src_rect, display->sdl_surface, &dst_rect);

    x_offset += component_size_padded;

    SDL_DestroySurface(surface);
  }
}

static void _element_slider_render_func(Element* slider, Display* display) {
  ElementSliderData* data = (ElementSliderData*) &slider->data;

  switch (data->type) {
    case ELEMENT_SLIDER_DATA_TYPE_FLOAT:
      _element_slider_render_float(slider, display);
      break;
    case ELEMENT_SLIDER_DATA_TYPE_UINT:
      _element_slider_render_uint(slider, display);
      break;
    case ELEMENT_SLIDER_DATA_TYPE_VECTOR:
      _element_slider_render_vector(slider, display);
      break;
    case ELEMENT_SLIDER_DATA_TYPE_COLOR_RGB:
      break;
    case ELEMENT_SLIDER_DATA_TYPE_COLOR_RGBA:
      break;
  }
}

bool element_slider(Window* window, Display* display, ElementSliderArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element text;

  text.type        = ELEMENT_TYPE_SLIDER;
  text.render_func = _element_slider_render_func;

  ElementSliderData* data = (ElementSliderData*) &text.data;

  data->type              = args.type;
  data->color             = args.color;
  data->size              = args.size;
  data->component_padding = args.component_padding;
  data->center_x          = args.center_x;
  data->center_y          = args.center_y;

  switch (args.type) {
    case ELEMENT_SLIDER_DATA_TYPE_FLOAT:
      data->data_float = *(float*) args.data_binding;
      break;
    case ELEMENT_SLIDER_DATA_TYPE_UINT:
      data->data_uint = *(uint32_t*) args.data_binding;
      break;
    case ELEMENT_SLIDER_DATA_TYPE_VECTOR:
      data->data_vec3 = *(LuminaryVec3*) args.data_binding;
      break;
    case ELEMENT_SLIDER_DATA_TYPE_COLOR_RGB:
      data->data_RGB = *(LuminaryRGBF*) args.data_binding;
      break;
    case ELEMENT_SLIDER_DATA_TYPE_COLOR_RGBA:
      data->data_RGBAF = *(LuminaryRGBAF*) args.data_binding;
      break;
  }

  ElementMouseResult mouse_result;
  element_apply_context(&text, context, &args.size, display, &mouse_result);

  LUM_FAILURE_HANDLE(array_push(&window->element_queue, &text));

  return mouse_result.is_clicked;
}
