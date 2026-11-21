#include "mandarin_duck.h"

#include <stdio.h>
#include <string.h>

#include "worker.h"

static void _mandarin_duck_update_host_output_props(LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(host);

  LuminaryRendererSettings settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

  LuminaryOutputProperties properties;
  memset(&properties, 0, sizeof(LuminaryOutputProperties));

  properties.enabled = true;
  properties.width   = settings.width;
  properties.height  = settings.height;

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

    uint32_t instance_id;
    LUM_FAILURE_HANDLE(luminary_host_get_instance_from_name(host, file_drop.file_path, &instance_id));

    LuminaryInstance instance;
    LUM_FAILURE_HANDLE(luminary_host_get_instance(host, instance_id, &instance));

    instance.mesh_id  = mesh_id;
    instance.position = (LuminaryVec3) {.x = 0.0f, .y = 0.0f, .z = 0.0f};
    instance.rotation = (LuminaryVec3) {.x = 0.0f, .y = 0.0f, .z = 0.0f};
    instance.scale    = (LuminaryVec3) {.x = 1.0f, .y = 1.0f, .z = 1.0f};

    camera_handler_center_instance(duck->display->camera_handler, host, &instance);

    LUM_FAILURE_HANDLE(luminary_host_set_instance(host, instance_id, &instance));
  }

  LUM_FAILURE_HANDLE(array_clear(file_drop_array));
}

static void _mandarin_duck_queue_benchmark_output(MandarinDuck* duck, LuminaryOutputRequestProperties properties) {
  LuminaryOutputPromiseHandle handle;
  LUM_FAILURE_HANDLE(luminary_host_request_output(duck->host, properties, &handle));
  LUM_FAILURE_HANDLE(array_push(&duck->benchmark_output_promises, &handle));
}

static void _mandarin_duck_queue_benchmark_outputs(MandarinDuck* duck, uint32_t num_outputs) {
  LuminaryRendererSettings renderer_settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(duck->host, &renderer_settings));

  LUM_FAILURE_HANDLE(array_create(&duck->benchmark_output_promises, sizeof(LuminaryOutputPromiseHandle), num_outputs));

  const uint32_t max_sample_count            = 1u << num_outputs;
  const uint32_t num_exponential_outputs     = (num_outputs < 5) ? num_outputs : 5;
  LuminaryOutputRequestProperties properties = {.width = renderer_settings.width, .height = renderer_settings.height};

  for (uint32_t output_id = 0; output_id <= num_exponential_outputs; output_id++) {
    properties.sample_count = 1u << output_id;
    _mandarin_duck_queue_benchmark_output(duck, properties);

    if (output_id >= 2) {
      properties.sample_count = (1u << (output_id - 1)) + (1u << (output_id - 2));
      _mandarin_duck_queue_benchmark_output(duck, properties);
    }
  }

  for (uint32_t sample_count = 64; sample_count <= max_sample_count; sample_count += 32) {
    properties.sample_count = sample_count;
    _mandarin_duck_queue_benchmark_output(duck, properties);
  }
}

void mandarin_duck_create(MandarinDuck** _duck, MandarinDuckCreateInfo info) {
  MD_CHECK_NULL_ARGUMENT(_duck);
  MD_CHECK_NULL_ARGUMENT(info.host);

  MandarinDuck* duck;
  LUM_FAILURE_HANDLE(host_malloc(&duck, sizeof(MandarinDuck)));
  memset(duck, 0, sizeof(MandarinDuck));

  duck->mode             = info.mode;
  duck->host             = info.host;
  duck->output_directory = info.output_directory;

  switch (duck->mode) {
    case MANDARIN_DUCK_MODE_DEFAULT: {
      display_create(&duck->display, info.initial_window_width, info.initial_window_height);
      _mandarin_duck_update_host_output_props(duck->host);
    } break;
    case MANDARIN_DUCK_MODE_CLI: {
      duck->cli_samples = info.cli_samples;
      duck->cli_format  = info.cli_format;

      if (info.initial_window_width > 0 || info.initial_window_height > 0) {
        LuminaryRendererSettings settings;
        LUM_FAILURE_HANDLE(luminary_host_get_settings(duck->host, &settings));
        if (info.initial_window_width > 0)
          settings.width = info.initial_window_width;
        if (info.initial_window_height > 0)
          settings.height = info.initial_window_height;
        LUM_FAILURE_HANDLE(luminary_host_set_settings(duck->host, &settings));
      }

      const LuminaryOutputProperties properties = {0};
      LUM_FAILURE_HANDLE(luminary_host_set_output_properties(duck->host, properties));

      if (info.num_benchmark_outputs > 0) {
        duck->benchmark_name = info.benchmark_name;
        _mandarin_duck_queue_benchmark_outputs(duck, info.num_benchmark_outputs);
      }
    } break;
    default:
      break;
  }

  *_duck = duck;
}

static void _mandarin_duck_run_mode_default(MandarinDuck* duck) {
  bool exit_requested = false;

  LuminaryThreadStatus* ui_thread;
  LUM_FAILURE_HANDLE(thread_status_create(&ui_thread));

  LUM_FAILURE_HANDLE(thread_status_set_worker_name(ui_thread, "MandarinDuck"));
  LUM_FAILURE_HANDLE(thread_status_start(ui_thread, "Process Frame"));

  DisplayFileDrop* file_drop_array;
  LUM_FAILURE_HANDLE(array_create(&file_drop_array, sizeof(DisplayFileDrop), 4));

  LUM_FAILURE_HANDLE(luminary_host_start_new_render(duck->host));

  while (!exit_requested) {
    // Measure the time between event queries
    LUM_FAILURE_HANDLE(thread_status_stop(ui_thread));
    display_query_events(duck->display, &file_drop_array, &exit_requested);

    double time_step;
    LUM_FAILURE_HANDLE(thread_status_get_time(ui_thread, &time_step));

    LUM_FAILURE_HANDLE(luminary_host_get_last_error(duck->host));

    LUM_FAILURE_HANDLE(thread_status_start(ui_thread, "Process Frame"));

    _mandarin_duck_handle_file_drop(duck, duck->host, file_drop_array);

    display_process_tasks(duck->display);

    display_handle_inputs(duck->display, duck->host, time_step);
    display_handle_outputs(duck->display, duck->host, duck->output_directory);

    display_render(duck->display, duck->host);

    display_update(duck->display);
  }

  LUM_FAILURE_HANDLE(array_destroy(&file_drop_array));

  LUM_FAILURE_HANDLE(thread_status_destroy(&ui_thread));
}

struct BenchmarkSaveImageTaskArgs {
  LuminaryImageSaveArgs save_args;
  LuminaryHost* host;
  LuminaryOutputHandle output_handle;
} typedef BenchmarkSaveImageTaskArgs;

static void _benchmark_save_image_task_continuation(void* args) {
  BenchmarkSaveImageTaskArgs* task_args = (BenchmarkSaveImageTaskArgs*) args;

  luminary_host_release_output(task_args->host, task_args->output_handle);
}

static void _benchmark_save_image_task(void* args, MDPromise* promise) {
  BenchmarkSaveImageTaskArgs* task_args = (BenchmarkSaveImageTaskArgs*) args;

  LUM_FAILURE_HANDLE(luminary_image_save(&task_args->save_args));
  LUM_FAILURE_HANDLE(luminary_path_destroy(&task_args->save_args.file_path));

  SDL_LockMutex(promise->mutex);
  promise->continuation      = _benchmark_save_image_task_continuation;
  promise->continuation_args = task_args;
  SDL_UnlockMutex(promise->mutex);
}

static void _mandarin_duck_save_benchmark_outputs(MandarinDuck* duck) {
  const LuminaryImageFileFormat format = duck->cli_format;
  const char* output_directory         = duck->output_directory ? duck->output_directory : ".";
  char render_times_file_path[4096];
  const int times_path_length =
    snprintf(render_times_file_path, sizeof(render_times_file_path), "%s/BenchResults-%s.txt", output_directory, duck->benchmark_name);
  if (times_path_length < 0 || (size_t) times_path_length >= sizeof(render_times_file_path))
    crash_message("Benchmark results path is too long.");

  FILE* render_times_file = fopen(render_times_file_path, "wb");

  if (render_times_file == NULL)
    crash_message("Failed to open file %s", render_times_file_path);

  uint32_t num_benchmark_outputs;
  LUM_FAILURE_HANDLE(array_get_num_elements(duck->benchmark_output_promises, &num_benchmark_outputs));

  uint32_t obtained_outputs = 0;

  MDThreadPool* thread_pool;
  md_thread_pool_create(&thread_pool);

  MDTaskNode* active_promises = NULL;

  while (obtained_outputs != num_benchmark_outputs || active_promises != NULL) {
    LUM_FAILURE_HANDLE(luminary_host_get_last_error(duck->host));
    md_task_list_process(&active_promises);

    if (obtained_outputs == num_benchmark_outputs) {
      // Just wait a little if we are only burning out the remaining promises
      SDL_Delay(5);
      continue;
    }

    for (uint32_t output_id = 0; output_id < num_benchmark_outputs; output_id++) {
      LuminaryOutputPromiseHandle promise_handle = duck->benchmark_output_promises[output_id];

      if (promise_handle == LUMINARY_OUTPUT_HANDLE_INVALID)
        continue;

      LuminaryOutputHandle output_handle;
      LUM_FAILURE_HANDLE(luminary_host_try_await_output(duck->host, promise_handle, &output_handle));

      if (output_handle == LUMINARY_OUTPUT_HANDLE_INVALID)
        continue;

      LuminaryImage output_image;
      LUM_FAILURE_HANDLE(luminary_host_get_image(duck->host, output_handle, &output_image))

      info_message("[%07.1fs] %05u Samples", output_image.meta_data.time, output_image.meta_data.sample_count);
      obtained_outputs++;

      LuminaryPath* image_path;
      LUM_FAILURE_HANDLE(luminary_path_create(&image_path));

      char string[4096];
      const int image_path_length = snprintf(
        string, sizeof(string), "%s/Bench-%05u-%s.%s", output_directory, output_image.meta_data.sample_count, duck->benchmark_name,
        luminary_strings_image_file_format[format]);
      if (image_path_length < 0 || (size_t) image_path_length >= sizeof(string))
        crash_message("Benchmark image path is too long.");

      fprintf(render_times_file, "%u, %f\n", output_image.meta_data.sample_count, output_image.meta_data.time);

      LUM_FAILURE_HANDLE(luminary_path_set_from_string(image_path, string));

      BenchmarkSaveImageTaskArgs* task_args;
      LUM_FAILURE_HANDLE(host_malloc(&task_args, sizeof(BenchmarkSaveImageTaskArgs)));

      task_args->save_args.image        = output_image;
      task_args->save_args.file_path    = image_path;
      task_args->save_args.format       = format;
      task_args->save_args.jpeg_quality = 100;
      task_args->host                   = duck->host;
      task_args->output_handle          = output_handle;

      MDPromise* promise = md_thread_pool_enqueue(thread_pool, _benchmark_save_image_task, task_args);

      MDTaskNode* node;
      LUM_FAILURE_HANDLE(host_malloc(&node, sizeof(MDTaskNode)));

      node->task.promise = promise;
      node->task.args    = task_args;
      node->next         = active_promises;
      active_promises    = node;

      duck->benchmark_output_promises[output_id] = LUMINARY_OUTPUT_HANDLE_INVALID;
    }
  }

  md_thread_pool_destroy(&thread_pool);

  fclose(render_times_file);
}

static void _mandarin_duck_run_mode_cli(MandarinDuck* duck) {
  if (duck->benchmark_output_promises != NULL) {
    LUM_FAILURE_HANDLE(luminary_host_start_new_render(duck->host));
    _mandarin_duck_save_benchmark_outputs(duck);
    return;
  }

  static const char* const output_names[LUMINARY_IMAGE_FILE_FORMAT_COUNT] = {
    [LUMINARY_IMAGE_FILE_FORMAT_PNG] = "render.png",
    [LUMINARY_IMAGE_FILE_FORMAT_JPG] = "render.jpg",
    [LUMINARY_IMAGE_FILE_FORMAT_BMP] = "render.bmp",
    [LUMINARY_IMAGE_FILE_FORMAT_TGA] = "render.tga",
  };

  LuminaryRendererSettings settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(duck->host, &settings));

  const LuminaryOutputRequestProperties properties = {
    .sample_count = duck->cli_samples,
    .width        = settings.width,
    .height       = settings.height,
  };
  LuminaryOutputPromiseHandle promise_handle;
  LUM_FAILURE_HANDLE(luminary_host_request_output(duck->host, properties, &promise_handle));
  LUM_FAILURE_HANDLE(luminary_host_start_new_render(duck->host));

  LuminaryOutputHandle output_handle;
  for (;;) {
    LUM_FAILURE_HANDLE(luminary_host_get_last_error(duck->host));
    LUM_FAILURE_HANDLE(luminary_host_try_await_output(duck->host, promise_handle, &output_handle));

    if (output_handle != LUMINARY_OUTPUT_HANDLE_INVALID)
      break;

    SDL_Delay(5);
  }

  const char* output_directory = duck->output_directory ? duck->output_directory : ".";
  char output_path[4096];
  const int path_length = snprintf(output_path, sizeof(output_path), "%s/%s", output_directory, output_names[duck->cli_format]);
  if (path_length < 0 || (size_t) path_length >= sizeof(output_path))
    crash_message("Render output path is too long.");

  LuminaryImageSaveArgs save_args = {.format = duck->cli_format, .jpeg_quality = 100};
  LUM_FAILURE_HANDLE(luminary_host_get_image(duck->host, output_handle, &save_args.image));
  LUM_FAILURE_HANDLE(luminary_path_create(&save_args.file_path));
  LUM_FAILURE_HANDLE(luminary_path_set_from_string(save_args.file_path, output_path));
  LUM_FAILURE_HANDLE(luminary_image_save(&save_args));
  LUM_FAILURE_HANDLE(luminary_path_destroy(&save_args.file_path));
  LUM_FAILURE_HANDLE(luminary_host_release_output(duck->host, output_handle));

  info_message("Saved %s (%u samples).", output_path, save_args.image.meta_data.sample_count);
}

void mandarin_duck_run(MandarinDuck* duck) {
  switch (duck->mode) {
    case MANDARIN_DUCK_MODE_DEFAULT:
      _mandarin_duck_run_mode_default(duck);
      break;
    case MANDARIN_DUCK_MODE_CLI:
      _mandarin_duck_run_mode_cli(duck);
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
