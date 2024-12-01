#include "mouse_state.h"

void mouse_state_create(MouseState** mouse_state) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  LUM_FAILURE_HANDLE(host_malloc(mouse_state, sizeof(MouseState)));
  memset(*mouse_state, 0, sizeof(MouseState));
}

void mouse_state_reset_motion(MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  mouse_state->x_motion = 0.0f;
  mouse_state->y_motion = 0.0f;
}

void mouse_state_update_motion(MouseState* mouse_state, SDL_MouseMotionEvent sdl_event) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  mouse_state->x        = sdl_event.x;
  mouse_state->y        = sdl_event.y;
  mouse_state->x_motion = sdl_event.xrel;
  mouse_state->y_motion = sdl_event.yrel;
}

void mouse_state_update_button(MouseState* mouse_state, SDL_MouseButtonEvent sdl_event) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);
}

void mouse_state_update_wheel(MouseState* mouse_state, SDL_MouseWheelEvent sdl_event) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);
}

void mouse_state_destroy(MouseState** mouse_state) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);
  MD_CHECK_NULL_ARGUMENT(*mouse_state);

  LUM_FAILURE_HANDLE(host_free(mouse_state));
}
