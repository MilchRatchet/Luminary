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
  [WINDOW_ENTITY_PROPERTIES_TYPE_MATERIAL]  = "Material",
  [WINDOW_ENTITY_PROPERTIES_TYPE_INSTANCE]  = "Instance"};

static const ElementButtonImage _window_entity_properties_type_button_images[WINDOW_ENTITY_PROPERTIES_TYPE_COUNT] = {
  [WINDOW_ENTITY_PROPERTIES_TYPE_SETTINGS]  = ELEMENT_BUTTON_IMAGE_SETTINGS,
  [WINDOW_ENTITY_PROPERTIES_TYPE_CAMERA]    = ELEMENT_BUTTON_IMAGE_CAMERA,
  [WINDOW_ENTITY_PROPERTIES_TYPE_OCEAN]     = ELEMENT_BUTTON_IMAGE_WAVES,
  [WINDOW_ENTITY_PROPERTIES_TYPE_SKY]       = ELEMENT_BUTTON_IMAGE_SUN,
  [WINDOW_ENTITY_PROPERTIES_TYPE_CLOUD]     = ELEMENT_BUTTON_IMAGE_CLOUD,
  [WINDOW_ENTITY_PROPERTIES_TYPE_FOG]       = ELEMENT_BUTTON_IMAGE_MIST,
  [WINDOW_ENTITY_PROPERTIES_TYPE_PARTICLES] = ELEMENT_BUTTON_IMAGE_PRECIPITATION,
  [WINDOW_ENTITY_PROPERTIES_TYPE_MATERIAL]  = ELEMENT_BUTTON_IMAGE_MATERIAL,
  [WINDOW_ENTITY_PROPERTIES_TYPE_INSTANCE]  = ELEMENT_BUTTON_IMAGE_INSTANCE};

static void _window_sidebar_entity_properties_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  WindowSidebarData* data = (WindowSidebarData*) window->data;

  for (uint32_t entity_properties_id = 0; entity_properties_id < WINDOW_ENTITY_PROPERTIES_TYPE_COUNT; entity_properties_id++) {
    if (entity_properties_id != 0) {
      window_margin(window, 4);
    }

    bool is_visible;
    user_interface_get_window_visible(display->ui, data->window_ids[entity_properties_id], &is_visible);

    if (element_button(
          window, display, mouse_state,
          (ElementButtonArgs) {.shape        = ELEMENT_BUTTON_SHAPE_IMAGE,
                               .image        = _window_entity_properties_type_button_images[entity_properties_id],
                               .size         = (ElementSize) {.width = 32, .height = 32},
                               .color        = (is_visible) ? MD_COLOR_ACCENT_LIGHT_2 : MD_COLOR_GRAY,
                               .hover_color  = MD_COLOR_WHITE,
                               .press_color  = MD_COLOR_WHITE,
                               .tooltip_text = _window_entity_properties_type_tooltip_string[entity_properties_id]})) {
      user_interface_set_window_visible(display->ui, data->window_ids[entity_properties_id], !is_visible);
    }
  }
}

static const char* _window_mouse_modes_tooltip_string[DISPLAY_MOUSE_MODE_COUNT] = {
  [DISPLAY_MOUSE_MODE_DEFAULT]       = "Move",
  [DISPLAY_MOUSE_MODE_SELECT]        = "Select",
  [DISPLAY_MOUSE_MODE_FOCUS]         = "Focus Camera",
  [DISPLAY_MOUSE_MODE_RENDER_REGION] = "Render Region"};

static const ElementButtonImage _window_mouse_modes_button_images[DISPLAY_MOUSE_MODE_COUNT] = {
  [DISPLAY_MOUSE_MODE_DEFAULT]       = ELEMENT_BUTTON_IMAGE_MOVE,
  [DISPLAY_MOUSE_MODE_SELECT]        = ELEMENT_BUTTON_IMAGE_SELECT,
  [DISPLAY_MOUSE_MODE_FOCUS]         = ELEMENT_BUTTON_IMAGE_FOCUS,
  [DISPLAY_MOUSE_MODE_RENDER_REGION] = ELEMENT_BUTTON_IMAGE_REGION};

static void _window_sidebar_mouse_modes_action(Window* window, Display* display, LuminaryHost* host, const MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(window);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  for (uint32_t mouse_mode = 0; mouse_mode < DISPLAY_MOUSE_MODE_COUNT; mouse_mode++) {
    if (mouse_mode != 0) {
      window_margin(window, 4);
    }

    if (element_button(
          window, display, mouse_state,
          (ElementButtonArgs) {.shape = ELEMENT_BUTTON_SHAPE_IMAGE,
                               .image = _window_mouse_modes_button_images[mouse_mode],
                               .size  = (ElementSize) {.width = 32, .height = 32},
                               .color = (display->mouse_mode == (DisplayMouseMode) mouse_mode) ? MD_COLOR_ACCENT_LIGHT_2 : MD_COLOR_GRAY,
                               .hover_color  = MD_COLOR_WHITE,
                               .press_color  = MD_COLOR_WHITE,
                               .tooltip_text = _window_mouse_modes_tooltip_string[mouse_mode]})) {
      display_set_mouse_mode(display, mouse_mode);
    }
  }
}

void window_sidebar_create(Window** window, WindowSidebarType type) {
  MD_CHECK_NULL_ARGUMENT(window);

  window_create(window);

  (*window)->type            = WINDOW_TYPE_SIDEBAR;
  (*window)->visibility_mask = WINDOW_VISIBILITY_UTILITIES;
  (*window)->x               = 0;
  (*window)->y               = 0;
  (*window)->width           = 48;
  (*window)->height          = 128;
  (*window)->padding         = 8;
  (*window)->is_horizontal   = false;
  (*window)->is_visible      = true;
  (*window)->is_movable      = false;
  (*window)->background      = true;
  (*window)->auto_size       = true;
  (*window)->auto_align      = true;

  (*window)->fixed_depth = true;

  switch (type) {
    case WINDOW_SIDEBAR_TYPE_ENTITY_PROPERTIES:
      (*window)->margins = (WindowMargins) {
        .margin_top = 128, .margin_left = 32, .margin_right = WINDOW_MARGIN_INVALID, .margin_bottom = WINDOW_MARGIN_INVALID};
      (*window)->action_func = _window_sidebar_entity_properties_action;
      break;
    case WINDOW_SIDEBAR_TYPE_MOUSE_MODES:
      (*window)->margins = (WindowMargins) {
        .margin_top = 128, .margin_right = 32, .margin_left = WINDOW_MARGIN_INVALID, .margin_bottom = WINDOW_MARGIN_INVALID};
      (*window)->action_func = _window_sidebar_mouse_modes_action;
      break;
    default:
      break;
  }

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
