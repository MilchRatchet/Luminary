#include "mandarin_duck.h"

static void _mandarin_duck_update_host_output_props(LuminaryHost* host, uint32_t width, uint32_t height) {
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryOutputProperties properties;
  properties.enabled = true;
  properties.width   = width;
  properties.height  = height;

  LUM_FAILURE_HANDLE(luminary_host_set_output_properties(host, properties));
}

static void _mandarin_duck_handle_file_drop(LuminaryHost* host, DisplayFileDrop* file_drop_array) {
  uint32_t num_file_drops;
  LUM_FAILURE_HANDLE(array_get_num_elements(file_drop_array, &num_file_drops));

  for (uint32_t file_drop_index = 0; file_drop_index < num_file_drops; file_drop_index++) {
    DisplayFileDrop file_drop = file_drop_array[file_drop_index];

    LuminaryPath* lum_path;
    LUM_FAILURE_HANDLE(luminary_path_create(&lum_path));
    LUM_FAILURE_HANDLE(luminary_path_set_from_string(lum_path, file_drop.file_path));

    LUM_FAILURE_HANDLE(luminary_host_load_obj_file(host, lum_path));

    LUM_FAILURE_HANDLE(luminary_path_destroy(&lum_path));
  }

  LUM_FAILURE_HANDLE(array_clear(file_drop_array));
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

  LUM_FAILURE_HANDLE(wall_time_start(md_timer));

  DisplayFileDrop* file_drop_array;
  LUM_FAILURE_HANDLE(array_create(&file_drop_array, sizeof(DisplayFileDrop), 4));

  while (!exit_requested) {
    bool display_dirty = false;

    // Measure the time between event queries
    LUM_FAILURE_HANDLE(wall_time_stop(md_timer));
    display_query_events(duck->display, &file_drop_array, &exit_requested, &display_dirty);

    double time_step;
    LUM_FAILURE_HANDLE(wall_time_get_time(md_timer, &time_step));

    LUM_FAILURE_HANDLE(wall_time_start(md_timer));

    if (display_dirty) {
      _mandarin_duck_update_host_output_props(duck->host, duck->display->width, duck->display->height);
    }

    _mandarin_duck_handle_file_drop(duck->host, file_drop_array);

    display_handle_inputs(duck->display, duck->host, time_step);

    display_render(duck->display, duck->host);

    display_update(duck->display);
  }

  LUM_FAILURE_HANDLE(array_destroy(&file_drop_array));

  LUM_FAILURE_HANDLE(wall_time_destroy(&md_timer));
}

void mandarin_duck_destroy(MandarinDuck** duck) {
  MD_CHECK_NULL_ARGUMENT(duck);

  display_destroy(&(*duck)->display);

  LUM_FAILURE_HANDLE(host_free(duck));
}
