#include "keyboard_state.h"

void keyboard_state_create(KeyboardState** keyboard_state) {
  MD_CHECK_NULL_ARGUMENT(keyboard_state);

  LUM_FAILURE_HANDLE(host_malloc(keyboard_state, sizeof(KeyboardState)));
  memset((*keyboard_state), 0, sizeof(KeyboardState));
}

void keyboard_state_update(KeyboardState* keyboard_state, SDL_KeyboardEvent sdl_event) {
  MD_CHECK_NULL_ARGUMENT(keyboard_state);

  KeyState* key_state = keyboard_state->keys + sdl_event.scancode;

  key_state->down   = sdl_event.down;
  key_state->repeat = sdl_event.repeat;
}

void keyboard_state_destroy(KeyboardState** keyboard_state) {
  MD_CHECK_NULL_ARGUMENT(keyboard_state);
  MD_CHECK_NULL_ARGUMENT(*keyboard_state);

  LUM_FAILURE_HANDLE(host_free(keyboard_state));
}
