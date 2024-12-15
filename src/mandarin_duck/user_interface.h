#ifndef MANDARIN_DUCK_USER_INTERFACE_H
#define MANDARIN_DUCK_USER_INTERFACE_H

#include "utils.h"
#include "window.h"

struct UserInterface {
  Window** windows;
} typedef UserInterface;

void user_interface_create(UserInterface** ui);
void user_interface_mouse_hovers_background(UserInterface* ui, Display* display, bool* mouse_hovers_background);
void user_interface_handle_inputs(UserInterface* ui, Display* display, LuminaryHost* host);
void user_interface_render(UserInterface* ui, Display* display);
void user_interface_destroy(UserInterface** ui);

#endif /* MANDARIN_DUCK_USER_INTERFACE_H */
