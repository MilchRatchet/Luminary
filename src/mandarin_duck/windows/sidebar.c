#include "sidebar.h"

#include "display.h"
#include "elements/button.h"
#include "user_interface.h"

struct WindowSidebarData {
  uint32_t window_ids[WINDOW_ENTITY_PROPERTIES_TYPE_COUNT];
} typedef WindowSidebarData;
static_assert(sizeof(WindowSidebarData) <= WINDOW_DATA_SECTION_SIZE, "Window data exceeds allocated size.");

static const char* _window_entity_properties_type_tooltip_string[WINDOW_ENTITY_PROPERTIES_TYPE_COUNT] = {
  [WINDOW_ENTITY_PROPERTIES_TYPE_SETTINGS]  = "Renderer Settings",
  [WINDOW_ENTITY_PROPERTIES_TYPE_CAMERA]    = "Camera",
  [WINDOW_ENTITY_PROPERTIES_TYPE_OCEAN]     = "Ocean",
  [WINDOW_ENTITY_PROPERTIES_TYPE_SKY]       = "Sky",
  [WINDOW_ENTITY_PROPERTIES_TYPE_CLOUD]     = "Cloud",
  [WINDOW_ENTITY_PROPERTIES_TYPE_FOG]       = "Fog",
  [WINDOW_ENTITY_PROPERTIES_TYPE_PARTICLES] = "Particles",
  [WINDOW_ENTITY_PROPERTIES_TYPE_MATERIAL]  = "Material"};

static bool _window_sidebar_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowSidebarData* data = (WindowSidebarData*) window->data;

  for (uint32_t entity_properties_id = 0; entity_properties_id < WINDOW_ENTITY_PROPERTIES_TYPE_COUNT; entity_properties_id++) {
    if (entity_properties_id != 0) {
      window_margin(window, 4);
    }

    if (element_button(
          window, display, mouse_state,
          (ElementButtonArgs){
            .shape        = ELEMENT_BUTTON_SHAPE_IMAGE,
            .size         = (ElementSize){.width = 32, .height = 32},
            .color        = 0xFF888888,
            .hover_color  = 0xFFFFFFFF,
            .press_color  = 0xFFFFFFFF,
            .tooltip_text = _window_entity_properties_type_tooltip_string[entity_properties_id]})) {
      bool is_visible;
      user_interface_get_window_visible(display->ui, data->window_ids[entity_properties_id], &is_visible);
      user_interface_set_window_visible(display->ui, data->window_ids[entity_properties_id], !is_visible);
    }
  }

  return false;
}

void window_sidebar_create(Window** window) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->type          = WINDOW_TYPE_SIDEBAR;
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
  (*window)->fixed_depth = true;

  window_create_subwindow(*window);

  window_allocate_memory(*window);
}

void window_sidebar_register_window_id(Window* window, WindowEntityPropertiesType type, uint32_t window_id) {
  WindowSidebarData* data = (WindowSidebarData*) window->data;

  if (type >= WINDOW_ENTITY_PROPERTIES_TYPE_COUNT) {
    crash_message("Invalid entity properties window type was passed.");
  }

  data->window_ids[type] = window_id;
}
