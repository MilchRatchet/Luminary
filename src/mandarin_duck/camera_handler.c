#include "camera_handler.h"

void camera_handler_create(CameraHandler** camera_handler) {
  MD_CHECK_NULL_ARGUMENT(camera_handler);

  LUM_FAILURE_HANDLE(host_malloc(camera_handler, sizeof(CameraHandler)));
  memset(*camera_handler, 0, sizeof(CameraHandler));

  (*camera_handler)->camera_speed = 10.0f;
  (*camera_handler)->mode         = CAMERA_MODE_FLY;
}

void camera_handler_set_mode(CameraHandler* camera_handler, CameraMode mode) {
  MD_CHECK_NULL_ARGUMENT(camera_handler);

  camera_handler->mode = mode;
}

void camera_handler_set_reference_pos(CameraHandler* camera_handler, LuminaryVec3 ref_pos) {
  MD_CHECK_NULL_ARGUMENT(camera_handler);

  camera_handler->reference_pos = ref_pos;
  camera_handler->dist          = sqrtf(ref_pos.x * ref_pos.x + ref_pos.y * ref_pos.y + ref_pos.z * ref_pos.z);
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

static void _camera_handler_update_fly(
  CameraHandler* camera_handler, LuminaryHost* host, KeyboardState* keyboard_state, MouseState* mouse_state, float time_step) {
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

static void _camera_handler_update_orbit(CameraHandler* camera_handler, LuminaryHost* host, MouseState* mouse_state) {
  LuminaryCamera camera;
  LUM_FAILURE_HANDLE(luminary_host_get_camera(host, &camera));

  Quaternion q0 = _camera_handler_rotation_to_quaternion(camera.rotation);
  q0.x          = -q0.x;
  q0.y          = -q0.y;
  q0.z          = -q0.z;

  camera.rotation.x -= mouse_state->y_motion * 0.005f;
  camera.rotation.y -= mouse_state->x_motion * 0.005f;

  Quaternion q1 = _camera_handler_rotation_to_quaternion(camera.rotation);

  const LuminaryVec3 ref_pos = camera_handler->reference_pos;

  const LuminaryVec3 center = (LuminaryVec3) {.x = camera.pos.x + ref_pos.x, .y = camera.pos.y + ref_pos.y, .z = camera.pos.z + ref_pos.z};

  const LuminaryVec3 v0 = _camera_handler_rotate_vector_by_quaternion(ref_pos, q0);
  const LuminaryVec3 v1 = _camera_handler_rotate_vector_by_quaternion(v0, q1);

  camera.pos.x = center.x - v1.x;
  camera.pos.y = center.y - v1.y;
  camera.pos.z = center.z - v1.z;

  camera_handler->reference_pos.x = v1.x;
  camera_handler->reference_pos.y = v1.y;
  camera_handler->reference_pos.z = v1.z;

  LUM_FAILURE_HANDLE(luminary_host_set_camera(host, &camera));
}

static void _camera_handler_update_zoom(CameraHandler* camera_handler, LuminaryHost* host, MouseState* mouse_state) {
  LuminaryCamera camera;
  LUM_FAILURE_HANDLE(luminary_host_get_camera(host, &camera));

  const LuminaryVec3 ref_pos = camera_handler->reference_pos;

  const float scale     = sqrtf(ref_pos.x * ref_pos.x + ref_pos.y * ref_pos.y + ref_pos.z * ref_pos.z);
  const float inv_scale = 1.0f / fmaxf(1.0f, scale);

  const float movement_scale = camera_handler->dist * inv_scale * (mouse_state->x_motion - mouse_state->y_motion) * 0.0025f;

  const LuminaryVec3 movement =
    (LuminaryVec3) {.x = ref_pos.x * movement_scale, .y = ref_pos.y * movement_scale, .z = ref_pos.z * movement_scale};

  camera.pos.x = camera.pos.x + movement.x;
  camera.pos.y = camera.pos.y + movement.y;
  camera.pos.z = camera.pos.z + movement.z;

  camera_handler->reference_pos.x = camera_handler->reference_pos.x - movement.x;
  camera_handler->reference_pos.y = camera_handler->reference_pos.y - movement.y;
  camera_handler->reference_pos.z = camera_handler->reference_pos.z - movement.z;

  LUM_FAILURE_HANDLE(luminary_host_set_camera(host, &camera));
}

void camera_handler_update(
  CameraHandler* camera_handler, LuminaryHost* host, KeyboardState* keyboard_state, MouseState* mouse_state, float time_step) {
  MD_CHECK_NULL_ARGUMENT(camera_handler);
  MD_CHECK_NULL_ARGUMENT(host);
  MD_CHECK_NULL_ARGUMENT(keyboard_state);
  MD_CHECK_NULL_ARGUMENT(mouse_state);

  switch (camera_handler->mode) {
    case CAMERA_MODE_FLY:
      _camera_handler_update_fly(camera_handler, host, keyboard_state, mouse_state, time_step);
      break;
    case CAMERA_MODE_ORBIT:
      _camera_handler_update_orbit(camera_handler, host, mouse_state);
      break;
    case CAMERA_MODE_ZOOM:
      _camera_handler_update_zoom(camera_handler, host, mouse_state);
      break;
    default:
      break;
  }
}

void camera_handler_center_instance(CameraHandler* camera_handler, LuminaryHost* host, LuminaryInstance* instance) {
  MD_CHECK_NULL_ARGUMENT(camera_handler);
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryCamera camera;
  LUM_FAILURE_HANDLE(luminary_host_get_camera(host, &camera));

  LuminaryVec3 forward = {.x = 0.0f, .y = 0.0f, .z = -1.0f};
  Quaternion q         = _camera_handler_rotation_to_quaternion(camera.rotation);
  forward              = _camera_handler_rotate_vector_by_quaternion(forward, q);

  // TODO: Position object based on size of object.
  const float dist = 10.0f;

  instance->position.x = camera.pos.x + dist * forward.x;
  instance->position.y = camera.pos.y + dist * forward.y;
  instance->position.z = camera.pos.z + dist * forward.z;
}

void camera_handler_destroy(CameraHandler** camera_handler) {
  MD_CHECK_NULL_ARGUMENT(camera_handler);
  MD_CHECK_NULL_ARGUMENT(*camera_handler);

  LUM_FAILURE_HANDLE(host_free(camera_handler));
}
