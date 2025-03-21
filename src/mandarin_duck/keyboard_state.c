#include "keyboard_state.h"

void keyboard_state_create(KeyboardState** keyboard_state) {
  MD_CHECK_NULL_ARGUMENT(keyboard_state);

  LUM_FAILURE_HANDLE(host_malloc(keyboard_state, sizeof(KeyboardState)));
  memset((*keyboard_state), 0, sizeof(KeyboardState));
}

void keyboard_state_reset_phases(KeyboardState* keyboard_state) {
  MD_CHECK_NULL_ARGUMENT(keyboard_state);

  for (uint32_t key = 0; key < keyboard_state->num_unstable_keys; key++) {
    SDL_Scancode scancode                = keyboard_state->unstable_keys[key];
    keyboard_state->keys[scancode].phase = KEY_PHASE_STABLE;
  }

  keyboard_state->num_unstable_keys = 0;
}

void keyboard_state_update(KeyboardState* keyboard_state, SDL_KeyboardEvent sdl_event) {
  MD_CHECK_NULL_ARGUMENT(keyboard_state);

  KeyState* key_state = keyboard_state->keys + sdl_event.scancode;

  KeyPhase phase = (sdl_event.down) ? KEY_PHASE_PRESSED : KEY_PHASE_RELEASED;

  key_state->phase           = phase;
  key_state->last_transition = sdl_event.timestamp;
  key_state->down            = sdl_event.down;
  key_state->repeat          = sdl_event.repeat;

  keyboard_state->unstable_keys[keyboard_state->num_unstable_keys++] = sdl_event.scancode;
}

void keyboard_state_destroy(KeyboardState** keyboard_state) {
  MD_CHECK_NULL_ARGUMENT(keyboard_state);
  MD_CHECK_NULL_ARGUMENT(*keyboard_state);

  LUM_FAILURE_HANDLE(host_free(keyboard_state));
}
