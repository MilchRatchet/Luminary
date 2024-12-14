#include "user_interface.h"

#include "windows/caption_controls.h"

////////////////////////////////////////////////////////////////////
// Internal functions
////////////////////////////////////////////////////////////////////

static void _user_interface_setup(UserInterface* ui) {
  MD_CHECK_NULL_ARGUMENT(ui);

  Window* window_caption_controls;
  window_caption_controls_create(&window_caption_controls);

  LUM_FAILURE_HANDLE(array_push(&ui->windows, &window_caption_controls));
}

////////////////////////////////////////////////////////////////////
// API functions
////////////////////////////////////////////////////////////////////

void user_interface_create(UserInterface** ui) {
  MD_CHECK_NULL_ARGUMENT(ui);

  LUM_FAILURE_HANDLE(host_malloc(ui, sizeof(UserInterface)));

  LUM_FAILURE_HANDLE(array_create(&(*ui)->windows, sizeof(Window*), 16));

  _user_interface_setup(*ui);
}

void user_interface_handle_inputs(UserInterface* ui, Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(ui);

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &num_windows));

  for (uint32_t window_id = 0; window_id < num_windows; window_id++) {
    Window* window = ui->windows[window_id];

    window_handle_input(window, display, host);
  }
}

void user_interface_render(UserInterface* ui, Display* display) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(display);

  uint32_t num_windows;
  LUM_FAILURE_HANDLE(array_get_num_elements(ui->windows, &num_windows));

  for (uint32_t window_id = 0; window_id < num_windows; window_id++) {
    Window* window = ui->windows[window_id];

    window_render(window, display);
  }
}

void user_interface_destroy(UserInterface** ui) {
  MD_CHECK_NULL_ARGUMENT(ui);
  MD_CHECK_NULL_ARGUMENT(*ui);

  LUM_FAILURE_HANDLE(array_destroy(&(*ui)->windows));

  LUM_FAILURE_HANDLE(host_free(ui));
}
