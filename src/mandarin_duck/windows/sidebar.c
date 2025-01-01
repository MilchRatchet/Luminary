#include "sidebar.h"

#include "display.h"
#include "elements/button.h"
#include "user_interface.h"

struct WindowSidebarData {
  uint32_t window_ids[WINDOW_ENTITY_PROPERTIES_TYPE_COUNT];
} typedef WindowSidebarData;
static_assert(sizeof(WindowSidebarData) <= WINDOW_DATA_SECTION_SIZE, "Window data exceeds allocated size.");

static bool _window_sidebar_action(Window* window, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowSidebarData* data = (WindowSidebarData*) window->data;

  if (element_button(
        window, display,
        (ElementButtonArgs){
          .shape       = ELEMENT_BUTTON_SHAPE_IMAGE,
          .size        = (ElementSize){.width = 32, .height = 32},
          .color       = 0xFF888888,
          .hover_color = 0xFFFFFFFF,
          .press_color = 0xFFFFFFFF})) {
    bool is_visible;
    user_interface_get_window_visible(display->ui, data->window_ids[WINDOW_ENTITY_PROPERTIES_TYPE_CAMERA], &is_visible);
    user_interface_set_window_visible(display->ui, data->window_ids[WINDOW_ENTITY_PROPERTIES_TYPE_CAMERA], !is_visible);
  }

  window_margin(window, 4);

  element_button(
    window, display,
    (ElementButtonArgs){
      .shape       = ELEMENT_BUTTON_SHAPE_IMAGE,
      .size        = (ElementSize){.width = 32, .height = 32},
      .color       = 0xFF888888,
      .hover_color = 0xFFFFFFFF,
      .press_color = 0xFFFFFFFF});

  window_margin(window, 4);

  element_button(
    window, display,
    (ElementButtonArgs){
      .shape       = ELEMENT_BUTTON_SHAPE_IMAGE,
      .size        = (ElementSize){.width = 32, .height = 32},
      .color       = 0xFF888888,
      .hover_color = 0xFFFFFFFF,
      .press_color = 0xFFFFFFFF});

  window_margin(window, 4);

  element_button(
    window, display,
    (ElementButtonArgs){
      .shape       = ELEMENT_BUTTON_SHAPE_IMAGE,
      .size        = (ElementSize){.width = 32, .height = 32},
      .color       = 0xFF888888,
      .hover_color = 0xFFFFFFFF,
      .press_color = 0xFFFFFFFF});

  window_margin(window, 4);

  element_button(
    window, display,
    (ElementButtonArgs){
      .shape       = ELEMENT_BUTTON_SHAPE_IMAGE,
      .size        = (ElementSize){.width = 32, .height = 32},
      .color       = 0xFF888888,
      .hover_color = 0xFFFFFFFF,
      .press_color = 0xFFFFFFFF});

  return false;
}

void window_sidebar_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->x             = 0;
  (*window)->y             = 0;
  (*window)->width         = 48;
  (*window)->height        = 128;
  (*window)->padding       = 8;
  (*window)->is_horizontal = false;
  (*window)->is_visible    = true;
  (*window)->is_movable    = false;
  (*window)->background    = true;
  (*window)->auto_size     = true;
  (*window)->auto_align    = true;
  (*window)->margins =
    (WindowMargins){.margin_top = 128, .margin_left = 32, .margin_right = WINDOW_MARGIN_INVALID, .margin_bottom = WINDOW_MARGIN_INVALID};
  (*window)->action_func = _window_sidebar_action;

  window_allocate_memory(*window);
}

void window_sidebar_register_window_id(Window* window, WindowEntityPropertiesType type, uint32_t window_id) {
  WindowSidebarData* data = (WindowSidebarData*) window->data;

  if (type >= WINDOW_ENTITY_PROPERTIES_TYPE_COUNT) {
    crash_message("Invalid entity properties window type was passed.");
  }

  data->window_ids[type] = window_id;
}
