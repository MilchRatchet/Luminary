#include "mouse_state.h"

#include <float.h>

void mouse_state_create(MouseState** mouse_state) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  LUM_FAILURE_HANDLE(host_malloc(mouse_state, sizeof(MouseState)));
  memset(*mouse_state, 0, sizeof(MouseState));
}

void mouse_state_copy(MouseState* dst, MouseState* src) {
  MD_CHECK_NULL_ARGUMENT(dst);
  MD_CHECK_NULL_ARGUMENT(src);

  memcpy(dst, src, sizeof(MouseState));
}

void mouse_state_reset_motion(MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  mouse_state->x_motion = 0.0f;
  mouse_state->y_motion = 0.0f;
}

void mouse_state_step_phase(MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  mouse_state->phase       = MOUSE_PHASE_STABLE;
  mouse_state->right_phase = MOUSE_PHASE_STABLE;
}

void mouse_state_invalidate(MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  mouse_state->x           = FLT_MAX;
  mouse_state->y           = FLT_MAX;
  mouse_state->down        = false;
  mouse_state->right_down  = false;
  mouse_state->phase       = MOUSE_PHASE_STABLE;
  mouse_state->right_phase = MOUSE_PHASE_STABLE;
}

void mouse_state_invalidate_position(MouseState* mouse_state) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  mouse_state->x = FLT_MAX;
  mouse_state->y = FLT_MAX;
}

void mouse_state_update_motion(MouseState* mouse_state, SDL_MouseMotionEvent sdl_event) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  mouse_state->x = sdl_event.x;
  mouse_state->y = sdl_event.y;
  mouse_state->x_motion += sdl_event.xrel;
  mouse_state->y_motion += sdl_event.yrel;
}

void mouse_state_update_button(MouseState* mouse_state, SDL_MouseButtonEvent sdl_event) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  switch (sdl_event.button) {
    case 1:
      mouse_state->down  = sdl_event.down;
      mouse_state->phase = (sdl_event.down) ? MOUSE_PHASE_PRESSED : MOUSE_PHASE_RELEASED;
      break;
    case 3:
      mouse_state->right_down  = sdl_event.down;
      mouse_state->right_phase = (sdl_event.down) ? MOUSE_PHASE_PRESSED : MOUSE_PHASE_RELEASED;
      break;
    default:
      break;
  }
}

void mouse_state_update_wheel(MouseState* mouse_state, SDL_MouseWheelEvent sdl_event) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);
}

void mouse_state_destroy(MouseState** mouse_state) {
  MD_CHECK_NULL_ARGUMENT(mouse_state);
  MD_CHECK_NULL_ARGUMENT(*mouse_state);

  LUM_FAILURE_HANDLE(host_free(mouse_state));
}
