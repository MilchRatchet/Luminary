#ifndef MANDARIN_DUCK_USER_INTERFACE_H
#define MANDARIN_DUCK_USER_INTERFACE_H

#include "utils.h"
#include "window.h"

struct UserInterface {
  Window** windows;
  uint32_t* window_ids_sorted;
  uint32_t about_window_id;
  MouseState* mouse_state;
} typedef UserInterface;

void user_interface_create(UserInterface** ui);
void user_interface_mouse_hovers_background(UserInterface* ui, Display* display, bool* mouse_hovers_background);
void user_interface_handle_inputs(UserInterface* ui, Display* display, LuminaryHost* host);
void user_interface_render(UserInterface* ui, Display* display);
void user_interface_destroy(UserInterface** ui);

void user_interface_get_window_visible(UserInterface* ui, uint32_t window_id, bool* visible);
void user_interface_set_window_visible(UserInterface* ui, uint32_t window_id, bool visible);

#endif /* MANDARIN_DUCK_USER_INTERFACE_H */
