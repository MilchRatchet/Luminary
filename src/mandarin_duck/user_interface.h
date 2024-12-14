#ifndef MANDARIN_DUCK_USER_INTERFACE_H
#define MANDARIN_DUCK_USER_INTERFACE_H

#include "utils.h"
#include "window.h"

struct UserInterface {
  Window* windows;
} typedef UserInterface;

void user_interface_create(UserInterface** ui);
void user_interface_destroy(UserInterface** ui);

#endif /* MANDARIN_DUCK_USER_INTERFACE_H */
