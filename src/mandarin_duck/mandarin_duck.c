#include "mandarin_duck.h"

#include <stdio.h>
#include <string.h>

static void _mandarin_duck_update_host_output_props(LuminaryHost* host, uint32_t width, uint32_t height) {
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryOutputProperties properties;
  properties.enabled = true;
  properties.width   = width;
  properties.height  = height;

  LUM_FAILURE_HANDLE(luminary_host_set_output_properties(host, properties));
}

static void _mandarin_duck_handle_file_drop(MandarinDuck* duck, LuminaryHost* host, DisplayFileDrop* file_drop_array) {
  uint32_t num_file_drops;
  LUM_FAILURE_HANDLE(array_get_num_elements(file_drop_array, &num_file_drops));

  for (uint32_t file_drop_index = 0; file_drop_index < num_file_drops; file_drop_index++) {
    DisplayFileDrop file_drop = file_drop_array[file_drop_index];

    uint32_t mesh_id;
    LUM_FAILURE_HANDLE(luminary_host_get_num_meshes(host, &mesh_id));

    LuminaryPath* lum_path;
    LUM_FAILURE_HANDLE(luminary_path_create(&lum_path));
    LUM_FAILURE_HANDLE(luminary_path_set_from_string(lum_path, file_drop.file_path));

    LUM_FAILURE_HANDLE(luminary_host_load_obj_file(host, lum_path));

    LUM_FAILURE_HANDLE(luminary_path_destroy(&lum_path));

    LuminaryInstance instance;
    LUM_FAILURE_HANDLE(luminary_host_new_instance(host, &instance));

    instance.mesh_id  = mesh_id;
    instance.position = (LuminaryVec3) {.x = 0.0f, .y = 0.0f, .z = 0.0f};
    instance.rotation = (LuminaryVec3) {.x = 0.0f, .y = 0.0f, .z = 0.0f};
    instance.scale    = (LuminaryVec3) {.x = 1.0f, .y = 1.0f, .z = 1.0f};

    camera_handler_center_instance(duck->display->camera_handler, host, &instance);

    LUM_FAILURE_HANDLE(luminary_host_set_instance(host, &instance));
  }

  LUM_FAILURE_HANDLE(array_clear(file_drop_array));
}

void mandarin_duck_create(MandarinDuck** _duck, MandarinDuckCreateArgs args) {
  MD_CHECK_NULL_ARGUMENT(_duck);
  MD_CHECK_NULL_ARGUMENT(args.host);

  MandarinDuck* duck;
  LUM_FAILURE_HANDLE(host_malloc(&duck, sizeof(MandarinDuck)));
  memset(duck, 0, sizeof(MandarinDuck));

  duck->mode             = args.mode;
  duck->host             = args.host;
  duck->output_directory = args.output_directory;

  switch (duck->mode) {
    case MANDARIN_DUCK_MODE_DEFAULT: {
      LuminaryRendererSettings renderer_settings;
      LUM_FAILURE_HANDLE(luminary_host_get_settings(duck->host, &renderer_settings));

      display_create(&duck->display, renderer_settings.width, renderer_settings.height);

      _mandarin_duck_update_host_output_props(duck->host, duck->display->width, duck->display->height);
    } break;
    case MANDARIN_DUCK_MODE_BENCHMARK: {
      LuminaryRendererSettings renderer_settings;
      LUM_FAILURE_HANDLE(luminary_host_get_settings(duck->host, &renderer_settings));

      duck->benchmark_name = args.benchmark_name;

      LUM_FAILURE_HANDLE(array_create(&duck->benchmark_output_promises, sizeof(LuminaryOutputPromiseHandle), args.num_benchmark_outputs));

      for (uint32_t output_id = 0; output_id <= args.num_benchmark_outputs; output_id++) {
        LuminaryOutputRequestProperties properties;
        properties.sample_count = 1 << output_id;
        properties.width        = renderer_settings.width;
        properties.height       = renderer_settings.height;

        LuminaryOutputPromiseHandle handle;
        LUM_FAILURE_HANDLE(luminary_host_request_output(duck->host, properties, &handle));

        LUM_FAILURE_HANDLE(array_push(&duck->benchmark_output_promises, &handle));
      }
    } break;
    default:
      break;
  }

  *_duck = duck;
}

static void _mandarin_duck_run_mode_default(MandarinDuck* duck) {
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

    _mandarin_duck_handle_file_drop(duck, duck->host, file_drop_array);

    display_handle_inputs(duck->display, duck->host, time_step);
    display_handle_outputs(duck->display, duck->host, duck->output_directory);

    display_render(duck->display, duck->host);

    display_update(duck->display);
  }

  LUM_FAILURE_HANDLE(array_destroy(&file_drop_array));

  LUM_FAILURE_HANDLE(wall_time_destroy(&md_timer));
}

void _mandarin_duck_run_mode_benchmark(MandarinDuck* duck) {
  char render_times_file_path[4096];
  sprintf(render_times_file_path, "%s/LuminaryBenchmarkResults.txt", duck->output_directory);

  FILE* render_times_file = fopen(render_times_file_path, "wb");

  uint32_t num_benchmark_outputs;
  LUM_FAILURE_HANDLE(array_get_num_elements(duck->benchmark_output_promises, &num_benchmark_outputs));

  uint32_t obtained_outputs = 0;

  while (obtained_outputs != num_benchmark_outputs) {
    for (uint32_t output_id = 0; output_id < num_benchmark_outputs; output_id++) {
      LuminaryOutputPromiseHandle promise_handle = duck->benchmark_output_promises[output_id];

      if (promise_handle == LUMINARY_OUTPUT_HANDLE_INVALID)
        continue;

      LuminaryOutputHandle output_handle;
      luminary_host_try_await_output(duck->host, promise_handle, &output_handle);

      if (output_handle == LUMINARY_OUTPUT_HANDLE_INVALID)
        continue;

      LuminaryImage output_image;
      LUM_FAILURE_HANDLE(luminary_host_get_image(duck->host, output_handle, &output_image))

      info_message("[%07.1fs] %05u Samples", output_image.meta_data.time, output_image.meta_data.sample_count);
      obtained_outputs++;

      LuminaryPath* image_path;
      LUM_FAILURE_HANDLE(luminary_path_create(&image_path));

      char string[4096];
      sprintf(string, "%s/Bench-%05u-%s.png", duck->output_directory, output_image.meta_data.sample_count, duck->benchmark_name);

      fprintf(render_times_file, "%u, %f\n", output_image.meta_data.sample_count, output_image.meta_data.time);

      LUM_FAILURE_HANDLE(luminary_path_set_from_string(image_path, string));

      LUM_FAILURE_HANDLE(luminary_host_save_png(duck->host, output_handle, image_path));

      LUM_FAILURE_HANDLE(luminary_path_destroy(&image_path));

      LUM_FAILURE_HANDLE(luminary_host_release_output(duck->host, output_handle));

      duck->benchmark_output_promises[output_id] = LUMINARY_OUTPUT_HANDLE_INVALID;
    }
  }

  fclose(render_times_file);
}

void mandarin_duck_run(MandarinDuck* duck) {
  switch (duck->mode) {
    case MANDARIN_DUCK_MODE_DEFAULT:
      _mandarin_duck_run_mode_default(duck);
      break;
    case MANDARIN_DUCK_MODE_BENCHMARK:
      _mandarin_duck_run_mode_benchmark(duck);
      break;
    default:
      break;
  }
}

void mandarin_duck_destroy(MandarinDuck** duck) {
  MD_CHECK_NULL_ARGUMENT(duck);

  if ((*duck)->display) {
    display_destroy(&(*duck)->display);
  }

  if ((*duck)->benchmark_output_promises) {
    LUM_FAILURE_HANDLE(array_destroy(&(*duck)->benchmark_output_promises));
  }

  LUM_FAILURE_HANDLE(host_free(duck));
}
