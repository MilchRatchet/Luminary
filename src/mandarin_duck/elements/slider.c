#include "slider.h"

#include <float.h>
#include <stdio.h>
#include <string.h>

#include "display.h"

static void _element_slider_render_float(Element* slider, Display* display) {
  ElementSliderData* data = (ElementSliderData*) &slider->data;

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, slider->width, slider->height, slider->x, slider->y, 0, 0xFF111111, 0xFF000000,
    UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);

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

  ui_renderer_render_rounded_box(
    display->ui_renderer, display, slider->width, slider->height, slider->x, slider->y, 0, 0xFF111111, 0xFF000000,
    UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);

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

  if (slider->width < 6 * data->component_padding) {
    return;
  }

  float* vec_data = (float*) &data->data_vec3;

  uint32_t x_offset = slider->x;

  const uint32_t component_size_padded = (slider->width - 2 * data->margins) / 3;
  const uint32_t component_size        = component_size_padded - 2 * data->component_padding;

  // Recompute margins so that we actually fill out the whole element's width.
  const uint32_t margins = (slider->width - component_size_padded * 3) >> 1;

  for (uint32_t component = 0; component < 3; component++) {
    ui_renderer_render_rounded_box(
      display->ui_renderer, display, component_size_padded, slider->height, x_offset, slider->y, 0, 0xFF111111, 0xFF000000,
      UI_RENDERER_BACKGROUND_MODE_SEMITRANSPARENT);

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

    x_offset += component_size_padded + margins;

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
    case ELEMENT_SLIDER_DATA_TYPE_RGB:
      _element_slider_render_vector(slider, display);
      break;
  }
}

static uint32_t _element_slider_get_subelement_index(Window* window, Display* display, Element* slider) {
  if ((slider->type != ELEMENT_SLIDER_DATA_TYPE_VECTOR) && (slider->type != ELEMENT_SLIDER_DATA_TYPE_RGB)) {
    return 0;
  }

  ElementSliderData* data = (ElementSliderData*) &slider->data;

  // Copy from the render functions
  const uint32_t component_size_padded = (slider->width - 2 * data->margins) / 3;
  const uint32_t margins               = (slider->width - component_size_padded * 3) >> 1;

  // We only execute this if we have pressed the element, so we know the mouse is inside the element.
  const uint32_t x = display->mouse_state->x - slider->x;

  if (x < component_size_padded) {
    return 0;
  }
  else if (x < 2 * component_size_padded + margins) {
    return 1;
  }
  else {
    return 2;
  }
}

bool element_slider(Window* window, Display* display, ElementSliderArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element slider;

  slider.type        = ELEMENT_TYPE_SLIDER;
  slider.render_func = _element_slider_render_func;
  slider.hash        = element_compute_hash(args.identifier);

  ElementSliderData* data = (ElementSliderData*) &slider.data;

  data->type              = args.type;
  data->color             = args.color;
  data->size              = args.size;
  data->component_padding = args.component_padding;
  data->margins           = args.margins;
  data->center_x          = args.center_x;
  data->center_y          = args.center_y;

  ElementMouseResult mouse_result;
  element_apply_context(&slider, context, &args.size, display, &mouse_result);

  bool updated_data = false;

  const bool use_slider = (window->state_data.state == WINDOW_INTERACTION_STATE_SLIDER) && (window->state_data.element_hash == slider.hash)
                          && (display->mouse_state->down);

  float mouse_change_rate = args.change_rate;
  if (display->keyboard_state->keys[SDL_SCANCODE_LCTRL].down) {
    mouse_change_rate = 0.1f * mouse_change_rate;
  }

  switch (args.type) {
    case ELEMENT_SLIDER_DATA_TYPE_FLOAT:
      data->data_float = *(float*) args.data_binding;

      if (use_slider) {
        mouse_change_rate *= (1.0f + sqrtf(fabsf(data->data_float)));

        data->data_float += display->mouse_state->x_motion * mouse_change_rate * 0.001f;
        data->data_float            = fminf(args.max, fmaxf(args.min, data->data_float));
        *(float*) args.data_binding = data->data_float;

        updated_data = true;
      }
      break;
    case ELEMENT_SLIDER_DATA_TYPE_UINT:
      data->data_uint = *(uint32_t*) args.data_binding;
      break;
    case ELEMENT_SLIDER_DATA_TYPE_VECTOR:
    case ELEMENT_SLIDER_DATA_TYPE_RGB:
      data->data_vec3 = *(LuminaryVec3*) args.data_binding;

      if (use_slider) {
        float* value = ((float*) &data->data_vec3) + window->state_data.subelement_index;
        *value += display->mouse_state->x_motion * mouse_change_rate * 0.001f;
        *value = fminf(args.max, fmaxf(args.min, *value));

        ((float*) args.data_binding)[window->state_data.subelement_index] = *value;

        updated_data = true;
      }
      break;
  }

  if (mouse_result.is_hovered) {
    window->element_has_hover = true;
  }

  if (mouse_result.is_pressed && window->state_data.state == WINDOW_INTERACTION_STATE_NONE) {
    window->state_data.state            = WINDOW_INTERACTION_STATE_SLIDER;
    window->state_data.element_hash     = slider.hash;
    window->state_data.subelement_index = _element_slider_get_subelement_index(window, display, &slider);

    display_set_mouse_visible(display, false);
  }

  window_push_element(window, &slider);

  return updated_data;
}
