#include "mandarin_duck.h"

static void _mandarin_duck_update_host_output_props(LuminaryHost* host, uint32_t width, uint32_t height) {
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryOutputProperties properties;
  properties.enabled = true;
  properties.width   = width;
  properties.height  = height;

  LUM_FAILURE_HANDLE(luminary_host_set_output_properties(host, properties));
}

static void _mandarin_duck_handle_inputs(LuminaryHost* host, KeyboardState* state) {
  MD_CHECK_NULL_ARGUMENT(host);
  MD_CHECK_NULL_ARGUMENT(state);

  LuminaryCamera camera;
  LUM_FAILURE_HANDLE(luminary_host_get_camera(host, &camera));

  if (state->keys[SDL_SCANCODE_W].down) {
    camera.pos.z -= 0.1f;
  }

  if (state->keys[SDL_SCANCODE_A].down) {
    camera.pos.x -= 0.1f;
  }

  if (state->keys[SDL_SCANCODE_S].down) {
    camera.pos.z += 0.1f;
  }

  if (state->keys[SDL_SCANCODE_D].down) {
    camera.pos.x += 0.1f;
  }

  if (state->keys[SDL_SCANCODE_SPACE].down) {
    camera.pos.y += 0.1f;
  }

  if (state->keys[SDL_SCANCODE_LCTRL].down) {
    camera.pos.y -= 0.1f;
  }

  LUM_FAILURE_HANDLE(luminary_host_set_camera(host, &camera));
}

void mandarin_duck_create(MandarinDuck** _duck, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(_duck);
  MD_CHECK_NULL_ARGUMENT(host);

  MandarinDuck* duck;
  LUM_FAILURE_HANDLE(host_malloc(&duck, sizeof(MandarinDuck)));

  duck->host = host;

  LuminaryRendererSettings renderer_settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(duck->host, &renderer_settings));

  display_create(&duck->display, renderer_settings.width, renderer_settings.height);

  _mandarin_duck_update_host_output_props(duck->host, duck->display->width, duck->display->height);

  *_duck = duck;
}

void mandarin_duck_run(MandarinDuck* duck) {
  bool exit_requested = false;

  LuminaryWallTime* md_timer;
  LUM_FAILURE_HANDLE(wall_time_create(&md_timer));

  while (!exit_requested) {
    LUM_FAILURE_HANDLE(wall_time_start(md_timer));

    bool display_dirty = false;
    display_query_events(duck->display, &exit_requested, &display_dirty);

    if (display_dirty) {
      _mandarin_duck_update_host_output_props(duck->host, duck->display->width, duck->display->height);
    }

    _mandarin_duck_handle_inputs(duck->host, duck->display->keyboard_state);

    display_render(duck->display, duck->host);

    display_update(duck->display);

    LUM_FAILURE_HANDLE(wall_time_stop(md_timer));

    double time;
    LUM_FAILURE_HANDLE(wall_time_get_time(md_timer, &time));
  }

  LUM_FAILURE_HANDLE(wall_time_destroy(&md_timer));
}

void mandarin_duck_destroy(MandarinDuck** duck) {
  MD_CHECK_NULL_ARGUMENT(duck);

  display_destroy(&(*duck)->display);

  LUM_FAILURE_HANDLE(host_free(duck));
}
