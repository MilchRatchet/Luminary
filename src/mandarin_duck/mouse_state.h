#ifndef MANDARIN_DUCK_MOUSE_STATE_H
#define MANDARIN_DUCK_MOUSE_STATE_H

#include <SDL3/SDL.h>

#include "utils.h"

struct MouseState {
  float x;
  float y;
  float x_motion;
  float y_motion;
} typedef MouseState;

void mouse_state_create(MouseState** mouse_state);
void mouse_state_reset_motion(MouseState* mouse_state);
void mouse_state_update_motion(MouseState* mouse_state, SDL_MouseMotionEvent sdl_event);
void mouse_state_update_button(MouseState* mouse_state, SDL_MouseButtonEvent sdl_event);
void mouse_state_update_wheel(MouseState* mouse_state, SDL_MouseWheelEvent sdl_event);
void mouse_state_destroy(MouseState** mouse_state);

#endif /* MANDARIN_DUCK_MOUSE_STATE_H */
