#include "camera_handler.h"

void camera_handler_create(CameraHandler** camera_handler) {
  MD_CHECK_NULL_ARGUMENT(camera_handler);

  LUM_FAILURE_HANDLE(host_malloc(camera_handler, sizeof(CameraHandler)));

  (*camera_handler)->camera_speed = 10.0f;
}

struct Quaternion {
  float x;
  float y;
  float z;
  float w;
} typedef Quaternion;

static Quaternion _camera_handler_rotation_to_quaternion(LuminaryVec3 rotation) {
  const float cy = cosf(rotation.z * 0.5f);
  const float sy = sinf(rotation.z * 0.5f);
  const float cp = cosf(rotation.y * 0.5f);
  const float sp = sinf(rotation.y * 0.5f);
  const float cr = cosf(rotation.x * 0.5f);
  const float sr = sinf(rotation.x * 0.5f);

  Quaternion q;
  q.x = sr * cp * cy - cr * sp * sy;
  q.y = cr * sp * cy + sr * cp * sy;
  q.z = cr * cp * sy - sr * sp * cy;
  q.w = cr * cp * cy + sr * sp * sy;

  return q;
}

static LuminaryVec3 _camera_handler_rotate_vector_by_quaternion(const LuminaryVec3 v, const Quaternion q) {
  const LuminaryVec3 u = {.x = q.x, .y = q.y, .z = q.z};

  const float s = q.w;

  const float dot_uv = u.x * v.x + u.y * v.y + u.z * v.z;
  const float dot_uu = u.x * u.x + u.y * u.y + u.z * u.z;

  const LuminaryVec3 cross = {.x = u.y * v.z - u.z * v.y, .y = u.z * v.x - u.x * v.z, .z = u.x * v.y - u.y * v.x};

  const LuminaryVec3 result = {
    .x = 2.0f * dot_uv * u.x + ((s * s) - dot_uu) * v.x + 2.0f * s * cross.x,
    .y = 2.0f * dot_uv * u.y + ((s * s) - dot_uu) * v.y + 2.0f * s * cross.y,
    .z = 2.0f * dot_uv * u.z + ((s * s) - dot_uu) * v.z + 2.0f * s * cross.z};

  return result;
}

void camera_handler_update(
  CameraHandler* camera_handler, LuminaryHost* host, KeyboardState* keyboard_state, MouseState* mouse_state, float time_step) {
  MD_CHECK_NULL_ARGUMENT(camera_handler);
  MD_CHECK_NULL_ARGUMENT(host);
  MD_CHECK_NULL_ARGUMENT(keyboard_state);
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  LuminaryCamera camera;
  LUM_FAILURE_HANDLE(luminary_host_get_camera(host, &camera));

  camera.rotation.x -= mouse_state->y_motion * 0.001f;
  camera.rotation.y -= mouse_state->x_motion * 0.001f;

  Quaternion q = _camera_handler_rotation_to_quaternion(camera.rotation);

  LuminaryVec3 camera_movement = {.x = 0.0f, .y = 0.0f, .z = 0.0f};

  float movement_scale = camera_handler->camera_speed * time_step;

  if (keyboard_state->keys[SDL_SCANCODE_W].down) {
    camera_movement.z -= 1.0f;
  }

  if (keyboard_state->keys[SDL_SCANCODE_A].down) {
    camera_movement.x -= 1.0f;
  }

  if (keyboard_state->keys[SDL_SCANCODE_S].down) {
    camera_movement.z += 1.0f;
  }

  if (keyboard_state->keys[SDL_SCANCODE_D].down) {
    camera_movement.x += 1.0f;
  }

  if (keyboard_state->keys[SDL_SCANCODE_SPACE].down) {
    camera_movement.y += 1.0f;
  }

  if (keyboard_state->keys[SDL_SCANCODE_LCTRL].down) {
    camera_movement.y -= 1.0f;
  }

  if (keyboard_state->keys[SDL_SCANCODE_LSHIFT].down) {
    movement_scale *= 2.0f;
  }

  camera_movement = _camera_handler_rotate_vector_by_quaternion(camera_movement, q);

  camera.pos.x += movement_scale * camera_movement.x;
  camera.pos.y += movement_scale * camera_movement.y;
  camera.pos.z += movement_scale * camera_movement.z;

  LUM_FAILURE_HANDLE(luminary_host_set_camera(host, &camera));
}

void camera_handler_destroy(CameraHandler** camera_handler) {
  MD_CHECK_NULL_ARGUMENT(camera_handler);
  MD_CHECK_NULL_ARGUMENT(*camera_handler);

  LUM_FAILURE_HANDLE(host_free(camera_handler));
}
