#ifndef MANDARIN_DUCK_KEYBOARD_STATE_H
#define MANDARIN_DUCK_KEYBOARD_STATE_H

#include <SDL3/SDL.h>

#include "utils.h"

enum KeyPhase { KEY_PHASE_STABLE = 0, KEY_PHASE_PRESSED = 1, KEY_PHASE_RELEASED = 2 } typedef KeyPhase;

struct KeyState {
  KeyPhase phase;
  uint64_t last_transition;
  bool down;
  bool repeat;
} typedef KeyState;

struct KeyboardState {
  KeyState keys[SDL_SCANCODE_COUNT];
  SDL_Scancode unstable_keys[SDL_SCANCODE_COUNT];
  uint32_t num_unstable_keys;
} typedef KeyboardState;

void keyboard_state_create(KeyboardState** keyboard_state);
void keyboard_state_reset_phases(KeyboardState* keyboard_state);
void keyboard_state_update(KeyboardState* keyboard_state, SDL_KeyboardEvent sdl_event);
void keyboard_state_destroy(KeyboardState** keyboard_state);

#endif /* MANDARIN_DUCK_KEYBOARD_STATE_H */
