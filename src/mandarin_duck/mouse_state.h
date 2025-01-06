#ifndef MANDARIN_DUCK_MOUSE_STATE_H
#define MANDARIN_DUCK_MOUSE_STATE_H

#include <SDL3/SDL.h>

#include "utils.h"

enum MousePhase { MOUSE_PHASE_STABLE = 0, MOUSE_PHASE_PRESSED = 1, MOUSE_PHASE_RELEASED = 2 } typedef MousePhase;

struct MouseState {
  float x;
  float y;
  float x_motion;
  float y_motion;
  MousePhase phase;
  bool down;
  MousePhase right_phase;
  bool right_down;
} typedef MouseState;

void mouse_state_create(MouseState** mouse_state);
void mouse_state_copy(MouseState* dst, MouseState* src);
void mouse_state_reset_motion(MouseState* mouse_state);
void mouse_state_step_phase(MouseState* mouse_state);
void mouse_state_invalidate(MouseState* mouse_state);
void mouse_state_invalidate_position(MouseState* mouse_state);
void mouse_state_update_motion(MouseState* mouse_state, SDL_MouseMotionEvent sdl_event);
void mouse_state_update_button(MouseState* mouse_state, SDL_MouseButtonEvent sdl_event);
void mouse_state_update_wheel(MouseState* mouse_state, SDL_MouseWheelEvent sdl_event);
void mouse_state_destroy(MouseState** mouse_state);

#endif /* MANDARIN_DUCK_MOUSE_STATE_H */
