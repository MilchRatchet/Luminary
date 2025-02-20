#ifndef MANDARIN_DUCK_USER_INTERFACE_H
#define MANDARIN_DUCK_USER_INTERFACE_H

#include "utils.h"

struct Display typedef Display;
struct Window typedef Window;
struct MouseState typedef MouseState;

typedef uint32_t WindowVisibilityMask;

struct UserInterfaceStatus {
  bool received_hover;
  bool received_mouse_action;
  bool received_keyboard_action;
} typedef UserInterfaceStatus;

UserInterfaceStatus user_interface_status_default();
UserInterfaceStatus user_interface_status_merge(UserInterfaceStatus a, UserInterfaceStatus b);

struct UserInterface {
  Window** windows;
  uint32_t* window_ids_sorted;
  uint32_t about_window_id;
  MouseState* mouse_state;
} typedef UserInterface;

void user_interface_create(UserInterface** ui);
void user_interface_mouse_hovers_background(UserInterface* ui, Display* display, bool* mouse_hovers_background);
UserInterfaceStatus user_interface_handle_inputs(
  UserInterface* ui, Display* display, LuminaryHost* host, WindowVisibilityMask visibility_mask);
void user_interface_render(UserInterface* ui, Display* display, WindowVisibilityMask visibility_mask);
void user_interface_destroy(UserInterface** ui);

void user_interface_get_window_visible(UserInterface* ui, uint32_t window_id, bool* visible);
void user_interface_set_window_visible(UserInterface* ui, uint32_t window_id, bool visible);

#endif /* MANDARIN_DUCK_USER_INTERFACE_H */
