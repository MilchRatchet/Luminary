#ifndef MANDARIN_DUCK_CAMERA_HANDLER_H
#define MANDARIN_DUCK_CAMERA_HANDLER_H

#include "keyboard_state.h"
#include "mouse_state.h"
#include "utils.h"

struct CameraHandler {
  float camera_speed;
} typedef CameraHandler;

void camera_handler_create(CameraHandler** camera_handler);
void camera_handler_update(
  CameraHandler* camera_handler, LuminaryHost* host, KeyboardState* keyboard_state, MouseState* mouse_state, float time_step);
void camera_handler_destroy(CameraHandler** camera_handler);

#endif /* MANDARIN_DUCK_CAMERA_HANDLER_H */
