#ifndef MANDARIN_DUCK_CAMERA_HANDLER_H
#define MANDARIN_DUCK_CAMERA_HANDLER_H

#include "keyboard_state.h"
#include "mouse_state.h"
#include "utils.h"

enum CameraMode { CAMERA_MODE_FLY, CAMERA_MODE_ORBIT, CAMERA_MODE_ZOOM, CAMERA_MODE_COUNT } typedef CameraMode;

struct CameraHandler {
  float camera_speed;
  CameraMode mode;
  LuminaryVec3 reference_pos;
  float dist;
} typedef CameraHandler;

void camera_handler_create(CameraHandler** camera_handler);
void camera_handler_set_mode(CameraHandler* camera_handler, CameraMode mode);
void camera_handler_set_reference_pos(CameraHandler* camera_handler, LuminaryVec3 ref_pos);
void camera_handler_update(
  CameraHandler* camera_handler, LuminaryHost* host, KeyboardState* keyboard_state, MouseState* mouse_state, float time_step);
void camera_handler_center_instance(CameraHandler* camera_handler, LuminaryHost* host, LuminaryInstance* instance);
void camera_handler_destroy(CameraHandler** camera_handler);

#endif /* MANDARIN_DUCK_CAMERA_HANDLER_H */
