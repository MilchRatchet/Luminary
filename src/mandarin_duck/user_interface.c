#include "user_interface.h"

#include "display.h"
#include "window.h"
#include "windows/about.h"
#include "windows/caption_controls.h"
#include "windows/entity_properties.h"
#include "windows/frametime.h"
#include "windows/renderer_status.h"
#include "windows/sidebar.h"

////////////////////////////////////////////////////////////////////
// Internal functions
////////////////////////////////////////////////////////////////////

static void _user_interface_setup(UserInterface* ui) {
  MD_CHECK_NULL_ARGUMENT(ui);

  uint32_t entity_property_window_ids[WINDOW_ENTITY_PROPERTIES_TYPE_COUNT];

  for (uint32_t entity_property_type = 0; entity_property_type < WINDOW_ENTITY_PROPERTIES_TYPE_COUNT; entity_property_type++) {
    Window* window_entity_properties;
    window_entity_properties_create(&window_entity_properties, (WindowEntityPropertiesType) entity_property_type);

    LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &entity_property_window_ids[entity_property_type]));
    LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_entity_properties));
  }

  Window* window_caption_controls;
  window_caption_controls_create(&window_caption_controls);

  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_caption_controls));

  Window* window_about;
  window_about_create(&window_about);

  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &ui->about_window_id));
  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_about));

  Window* window_frametime;
  window_frametime_create(&window_frametime);

  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_frametime));

  Window* window_renderer_status;
  window_renderer_status_create(&window_renderer_status);

  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_renderer_status));

  Window* window_sidebar_entities;
  window_sidebar_create(&window_sidebar_entities, WINDOW_SIDEBAR_TYPE_ENTITY_PROPERTIES);

  for (uint32_t entity_property_type = 0; entity_property_type < WINDOW_ENTITY_PROPERTIES_TYPE_COUNT; entity_property_type++) {
    window_sidebar_register_window_id(window_sidebar_entities, entity_property_type, entity_property_window_ids[entity_property_type]);
  }

  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_sidebar_entities));

  Window* window_sidebar_mouse_modes;
  window_sidebar_create(&window_sidebar_mouse_modes, WINDOW_SIDEBAR_TYPE_MOUSE_MODES);

  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_sidebar_mouse_modes));

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &num_windows));

  LUM_FAILURE_HANDLE(array_resize(&ui->window_ids_sorted, num_windows));
}

static void _user_interface_sort_windows_by_depth(UserInterface* ui) {
  MD_CHECK_NULL_ARGUMENT(ui);

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &num_windows));

  for (uint32_t window_id = 0; window_id < num_windows; window_id++) {
    ui->window_ids_sorted[window_id] = window_id;
  }

  uint32_t i = 1;

  while (i < num_windows) {
    uint32_t j = i;

    while ((j > 0) && (ui->windows[ui->window_ids_sorted[j - 1]]->depth < ui->windows[ui->window_ids_sorted[j]]->depth)) {
      uint32_t temp                = ui->window_ids_sorted[j - 1];
      ui->window_ids_sorted[j - 1] = ui->window_ids_sorted[j];
      ui->window_ids_sorted[j]     = temp;

      j = j - 1;
    }

    i++;
  }
}

////////////////////////////////////////////////////////////////////
// API functions
////////////////////////////////////////////////////////////////////

UserInterfaceStatus user_interface_status_default() {
  UserInterfaceStatus status = {.received_hover = false, .received_keyboard_action = false, .received_mouse_action = false};

  return status;
}

UserInterfaceStatus user_interface_status_merge(UserInterfaceStatus a, UserInterfaceStatus b) {
  UserInterfaceStatus status = {
    .received_hover           = a.received_hover || b.received_hover,
    .received_keyboard_action = a.received_keyboard_action || b.received_keyboard_action,
    .received_mouse_action    = a.received_mouse_action || b.received_mouse_action};

  return status;
}

void user_interface_create(UserInterface** ui) {
  MD_CHECK_NULL_ARGUMENT(ui);

  LUM_FAILURE_HANDLE(host_malloc(ui, sizeof(UserInterface)));
  memset(*ui, 0, sizeof(UserInterface));

  mouse_state_create(&(*ui)->mouse_state);

  LUM_FAILURE_HANDLE(array_create(&(*ui)->windows, sizeof(Window*), 16));
  LUM_FAILURE_HANDLE(array_create(&(*ui)->window_ids_sorted, sizeof(uint32_t), 16));

  _user_interface_setup(*ui);
  _user_interface_sort_windows_by_depth(*ui);
}

void user_interface_mouse_hovers_background(UserInterface* ui, Display* display, bool* mouse_hovers_background) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(mouse_hovers_background);

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &num_windows));

  bool window_handled_mouse = false;

  for (uint32_t window_id = 0; window_id < num_windows; window_id++) {
    Window* window = ui->windows[window_id];

    if (window->is_visible == false)
      continue;

    window_handled_mouse |= window_is_mouse_hover(window, display, display->mouse_state);
  }

  *mouse_hovers_background = !window_handled_mouse;
}

UserInterfaceStatus user_interface_handle_inputs(
  UserInterface* ui, Display* display, LuminaryHost* host, WindowVisibilityMask visibility_mask) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  // Toggle the "About" window with F1.
  if (display->keyboard_state->keys[SDL_SCANCODE_F1].phase == KEY_PHASE_PRESSED) {
    bool visible;
    user_interface_get_window_visible(ui, ui->about_window_id, &visible);
    user_interface_set_window_visible(ui, ui->about_window_id, !visible);
  }

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &num_windows));

  mouse_state_copy(ui->mouse_state, display->mouse_state);

  if (display->mouse_visible == false) {
    mouse_state_invalidate_position(ui->mouse_state);
  }

  UserInterfaceStatus status = user_interface_status_default();

  for (uint32_t window_id = 0; window_id < num_windows; window_id++) {
    Window* window = ui->windows[ui->window_ids_sorted[window_id]];

    if (window->is_visible == false || ((window->visibility_mask & visibility_mask) == 0))
      continue;

    UserInterfaceStatus window_status = window_handle_input(window, display, host, ui->mouse_state);

    status = user_interface_status_merge(status, window_status);
  }

  _user_interface_sort_windows_by_depth(ui);

  return status;
}

void user_interface_render(UserInterface* ui, Display* display, WindowVisibilityMask visibility_mask) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(display);

  if (display->camera_handler->mode == CAMERA_MODE_ORBIT || display->camera_handler->mode == CAMERA_MODE_ZOOM) {
    ui_renderer_render_crosshair(display->ui_renderer, display, display->reference_x, display->reference_y);
  }

  if (display->mouse_mode == DISPLAY_MOUSE_MODE_RENDER_REGION) {
    render_region_render(display->region, display, display->ui_renderer);
  }

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &num_windows));

  for (int32_t window_id = num_windows - 1; window_id >= 0; window_id--) {
    Window* window = ui->windows[ui->window_ids_sorted[window_id]];

    if (window->is_visible == false || ((window->visibility_mask & visibility_mask) == 0))
      continue;

    window_render(window, display);
  }

  // TODO: Turn into option
  // ui_renderer_render_display_corners(display->ui_renderer, display);
}

void user_interface_destroy(UserInterface** ui) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(*ui);

  mouse_state_destroy(&(*ui)->mouse_state);

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements((*ui)->windows, &num_windows));

  for (uint32_t window_id = 0; window_id < num_windows; window_id++) {
    window_destroy(&(*ui)->windows[window_id]);
  }

  LUM_FAILURE_HANDLE(array_destroy(&(*ui)->windows));
  LUM_FAILURE_HANDLE(array_destroy(&(*ui)->window_ids_sorted));

  LUM_FAILURE_HANDLE(host_free(ui));
}

void user_interface_get_window_visible(UserInterface* ui, uint32_t window_id, bool* visible) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(visible);

  *visible = ui->windows[window_id]->is_visible;
}

void user_interface_set_window_visible(UserInterface* ui, uint32_t window_id, bool visible) {
  MD_CHECK_NULL_ARGUMENT(ui);

  Window* window = ui->windows[window_id];

  window->is_visible = visible;

  if (visible) {
    window_set_focus(window);
  }
}
