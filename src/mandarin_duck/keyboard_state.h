#ifndef MANDARIN_DUCK_KEYBOARD_STATE_H
#define MANDARIN_DUCK_KEYBOARD_STATE_H

#include <SDL3/SDL.h>

#include "utils.h"

struct KeyState {
  bool down;
  bool repeat;
} typedef KeyState;

struct KeyboardState {
  KeyState keys[SDL_SCANCODE_COUNT];
} typedef KeyboardState;

void keyboard_state_create(KeyboardState** keyboard_state);
void keyboard_state_update(KeyboardState* keyboard_state, SDL_KeyboardEvent sdl_event);
void keyboard_state_destroy(KeyboardState** keyboard_state);

#endif /* MANDARIN_DUCK_KEYBOARD_STATE_H */
