#include <stdio.h>
#include <time.h>

#include "display_internal.h"

void display_screenshot_init(Display* display) {
  display->output_promise_handle = LUMINARY_OUTPUT_HANDLE_INVALID;
  display->image_file_format     = LUMINARY_IMAGE_FILE_FORMAT_JPG;
  display->active_promises       = NULL;
  md_thread_pool_create(&display->thread_pool);
}

void display_screenshot_destroy(Display* display) {
  while (display->active_promises != NULL) {
    display_process_tasks(display);
  }

  if (display->thread_pool != NULL) {
    md_thread_pool_destroy(&display->thread_pool);
  }
}

static void _display_generate_screenshot_name(
  const char* output_directory, char* string, LuminaryImage image, LuminaryImageFileFormat format) {
  time_t rawtime;
  struct tm timeinfo;

  char time_string[2048];

  time(&rawtime);
  timeinfo = *localtime(&rawtime);
  strftime(time_string, 2048, "%Y-%m-%d-%H-%M-%S", &timeinfo);

  const char* dir = (output_directory != (const char*) 0) ? output_directory : ".";

  sprintf(
    string, "%s/Snap-%s-%05u-%07.1fs.%s", dir, time_string, image.meta_data.sample_count, image.meta_data.time,
    luminary_strings_image_file_format[format]);
}

void display_screenshot_request(Display* display, LuminaryHost* host) {
  if (display->output_promise_handle != LUMINARY_OUTPUT_HANDLE_INVALID)
    return;

  LuminaryRendererSettings settings;
  LUM_FAILURE_HANDLE(luminary_host_get_settings(host, &settings));

  LuminaryOutputRequestProperties properties;
  properties.sample_count = 0;
  properties.width        = settings.width;
  properties.height       = settings.height;

  char status_message[32];
  display->pending_screenshot_format = display->image_file_format;
  snprintf(status_message, sizeof(status_message), "Save %s", luminary_strings_image_file_format[display->pending_screenshot_format]);
  display->pending_screenshot_status_id = display_add_status_message(display, "Display", status_message);

  LUM_FAILURE_HANDLE(luminary_host_request_output(host, properties, &display->output_promise_handle));
}

struct SaveImageTaskArgs {
  LuminaryImageSaveArgs save_args;
  LuminaryHost* host;
  LuminaryOutputHandle output_handle;
  Display* display;
  uint32_t status_message_id;
} typedef SaveImageTaskArgs;

static void _save_image_task_continuation(void* args) {
  SaveImageTaskArgs* task_args = (SaveImageTaskArgs*) args;

  LUM_FAILURE_HANDLE(luminary_host_release_output(task_args->host, task_args->output_handle));

  display_remove_status_message(task_args->display, task_args->status_message_id);
}

static void _save_image_task(void* args, MDPromise* promise) {
  SaveImageTaskArgs* task_args = (SaveImageTaskArgs*) args;

  LUM_FAILURE_HANDLE(luminary_image_save(&task_args->save_args));
  LUM_FAILURE_HANDLE(luminary_path_destroy(&task_args->save_args.file_path));

  SDL_LockMutex(promise->mutex);
  promise->continuation      = _save_image_task_continuation;
  promise->continuation_args = task_args;
  SDL_UnlockMutex(promise->mutex);
}

void display_process_tasks(Display* display) {
  md_task_list_process(&display->active_promises);
}

void display_handle_outputs(Display* display, LuminaryHost* host, const char* output_directory) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  if (display->output_promise_handle != LUMINARY_OUTPUT_HANDLE_INVALID) {
    LuminaryOutputHandle output_handle;
    LUM_FAILURE_HANDLE(luminary_host_try_await_output(host, display->output_promise_handle, &output_handle));

    if (output_handle == LUMINARY_OUTPUT_HANDLE_INVALID)
      return;

    LuminaryImage output_image;
    LUM_FAILURE_HANDLE(luminary_host_get_image(host, output_handle, &output_image));

    LuminaryPath* image_path;
    LUM_FAILURE_HANDLE(luminary_path_create(&image_path));

    char string[4096];
    _display_generate_screenshot_name(output_directory, string, output_image, display->pending_screenshot_format);

    LUM_FAILURE_HANDLE(luminary_path_set_from_string(image_path, string));

    SaveImageTaskArgs* task_args;
    LUM_FAILURE_HANDLE(host_malloc(&task_args, sizeof(SaveImageTaskArgs)));

    task_args->save_args.image        = output_image;
    task_args->save_args.file_path    = image_path;
    task_args->save_args.format       = display->pending_screenshot_format;
    task_args->save_args.jpeg_quality = 100;
    task_args->host                   = host;
    task_args->output_handle          = output_handle;
    task_args->display                = display;
    task_args->status_message_id      = display->pending_screenshot_status_id;

    MDPromise* promise = md_thread_pool_enqueue(display->thread_pool, _save_image_task, task_args);

    MDTaskNode* node;
    LUM_FAILURE_HANDLE(host_malloc(&node, sizeof(MDTaskNode)));

    node->task.promise       = promise;
    node->task.args          = task_args;
    node->next               = display->active_promises;
    display->active_promises = node;

    display->output_promise_handle = LUMINARY_OUTPUT_HANDLE_INVALID;
  }
}
