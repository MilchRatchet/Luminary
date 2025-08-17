#include "button.h"

#include "display.h"
#include "ui_renderer_utils.h"
#include "window.h"
#include "windows/subwindow_tooltip.h"

static void _element_button_render_circle(Element* button, Display* display) {
  ElementButtonData* data = (ElementButtonData*) &button->data;

  uint32_t color = (data->is_down) ? data->press_color : ((data->is_hovered) ? data->hover_color : data->color);

  Color256 color256        = color256_set_1(color);
  Color256 mask_low16      = color256_set_1(0x00FF00FF);
  Color256 mask_high16     = color256_set_1(0xFF00FF00);
  Color256 mask_add        = color256_set_1(0x00800080);
  Color256 mask_full_alpha = color256_set_1(0x000000FF);

  const uint32_t size = display->ui_renderer->shape_mask_size[data->shape_size_id];
  const uint8_t* src  = display->ui_renderer->disk_mask[data->shape_size_id];

  uint8_t* dst = display->buffer + 4 * button->x + button->y * display->pitch;

  const uint32_t cols = size >> UI_RENDERER_STRIDE_LOG;
  const uint32_t rows = size;

  for (uint32_t row = 0; row < rows; row++) {
    for (uint32_t col = 0; col < cols; col++) {
      Color256 disk_mask = color256_load(src + col * 32);
      Color256 base      = color256_load(dst + col * 32);

      color256_store(dst + col * 32, color256_alpha_blend(color256, base, disk_mask, mask_low16, mask_high16, mask_add, mask_full_alpha));
    }

    src = src + 4 * size;
    dst = dst + display->pitch;
  }
}

static const char* _button_image_string[ELEMENT_BUTTON_IMAGE_COUNT] = {
  [ELEMENT_BUTTON_IMAGE_CHECK] = "\ue5ca",    [ELEMENT_BUTTON_IMAGE_SETTINGS] = "\ue8b8",      [ELEMENT_BUTTON_IMAGE_CAMERA] = "\ue412",
  [ELEMENT_BUTTON_IMAGE_WAVES] = "\ue176",    [ELEMENT_BUTTON_IMAGE_SUN] = "\uf157",           [ELEMENT_BUTTON_IMAGE_CLOUD] = "\ue2bd",
  [ELEMENT_BUTTON_IMAGE_MIST] = "\ue188",     [ELEMENT_BUTTON_IMAGE_PRECIPITATION] = "\ue810", [ELEMENT_BUTTON_IMAGE_MATERIAL] = "\uef8f",
  [ELEMENT_BUTTON_IMAGE_INSTANCE] = "\uead3", [ELEMENT_BUTTON_IMAGE_MOVE] = "\ue89f",          [ELEMENT_BUTTON_IMAGE_SELECT] = "\uf706",
  [ELEMENT_BUTTON_IMAGE_FOCUS] = "\ue3b4",    [ELEMENT_BUTTON_IMAGE_SYNC] = "\ue627",          [ELEMENT_BUTTON_IMAGE_REGION] = "\ue5d0",
  [ELEMENT_BUTTON_IMAGE_ERROR] = "\ue000",    [ELEMENT_BUTTON_IMAGE_STAR] = "\ue838"};

static void _element_button_render_image(Element* button, Display* display) {
  ElementButtonData* data = (ElementButtonData*) &button->data;

  uint32_t color = (data->is_down) ? data->press_color : ((data->is_hovered) ? data->hover_color : data->color);

  const uint32_t padding_x = button->width >> 1;
  const uint32_t padding_y = button->height >> 1;

  text_renderer_render(
    display->text_renderer, display, _button_image_string[data->image], TEXT_RENDERER_FONT_MATERIAL, color, button->x + padding_x,
    button->y + padding_y, true, true, true, (uint32_t*) 0);
}

static void _element_button_render_func(Element* button, Display* display) {
  ElementButtonData* data = (ElementButtonData*) &button->data;

  switch (data->shape) {
    case ELEMENT_BUTTON_SHAPE_CIRCLE:
      _element_button_render_circle(button, display);
      break;
    case ELEMENT_BUTTON_SHAPE_IMAGE:
      _element_button_render_image(button, display);
      break;
    default:
      break;
  }
}

bool element_button(Window* window, Display* display, const MouseState* mouse_state, ElementButtonArgs args) {
  WindowContext* context = window->context_stack + window->context_stack_ptr;

  Element button;

  button.type        = ELEMENT_TYPE_BUTTON;
  button.render_func = _element_button_render_func;
  button.hash        = 0;

  ElementButtonData* data = (ElementButtonData*) &button.data;

  ElementMouseResult mouse_result;
  element_apply_context(&button, context, &args.size, mouse_state, &mouse_result);

  data->shape       = args.shape;
  data->image       = args.image;
  data->color       = args.color;
  data->hover_color = args.hover_color;
  data->press_color = args.press_color;
  data->is_hovered  = mouse_result.is_hovered && (args.is_not_interactive == false);
  data->is_down     = mouse_result.is_down && (args.is_not_interactive == false);

  data->shape_size_id = 0;

  if (args.shape == ELEMENT_BUTTON_SHAPE_CIRCLE) {
    for (uint32_t size_id = 0; size_id < SHAPE_MASK_COUNT; size_id++) {
      const uint32_t size = display->ui_renderer->shape_mask_size[size_id];

      if (size <= button.width && size <= button.height) {
        data->shape_size_id = size_id;
      }
    }
  }

  if (mouse_result.is_hovered && (args.is_not_interactive == false)) {
    window->element_has_hover = true;

    display_set_cursor(display, SDL_SYSTEM_CURSOR_POINTER);

    window->status.received_hover |= true;
  }

  if (mouse_result.is_hovered) {
    const bool external_clicked_window_is_present = (window->state_data.state == WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_CLICKED);

    if (args.tooltip_text && window->external_subwindow && !external_clicked_window_is_present) {
      subwindow_tooltip_create(window->external_subwindow, args.tooltip_text, mouse_state->x + 16.0f, mouse_state->y + 16.0f);

      window->state_data =
        (WindowInteractionStateData) {.state = WINDOW_INTERACTION_STATE_EXTERNAL_WINDOW_HOVER, .element_hash = button.hash};
    }
  }

  window->status.received_mouse_action |= mouse_result.is_clicked;

  window_push_element(window, &button);

  return mouse_result.is_clicked;
}
