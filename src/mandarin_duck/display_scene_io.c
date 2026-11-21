#include "display_internal.h"

void display_scene_handle_save(Display* display, LuminaryHost* host) {
  MD_CHECK_NULL_ARGUMENT(display);
  MD_CHECK_NULL_ARGUMENT(host);

  if (display->queued_save_scene == false || display->scene_file_path->dialog_open)
    return;

  LuminaryPath* path;
  file_dialog_handler_try_acquire_path(display->scene_file_path, &path);

  if (path == (LuminaryPath*) 0)
    return;

  bool is_empty;
  LUM_FAILURE_HANDLE(luminary_path_get_is_empty(path, &is_empty));

  if (is_empty == false) {
    LUM_FAILURE_HANDLE(luminary_host_save_as_lumV5(host, path));
  }

  file_dialog_handler_release_path(display->scene_file_path);
  display->queued_save_scene = false;
}

void display_scene_request_save(Display* display) {
  MD_CHECK_NULL_ARGUMENT(display);

  display->queued_save_scene = true;

  const bool shift_down = display->keyboard_state->keys[SDL_SCANCODE_LSHIFT].down;

  if (shift_down == false) {
    LuminaryPath* path;
    file_dialog_handler_try_acquire_path(display->scene_file_path, &path);

    if (path == (LuminaryPath*) 0)
      return;

    bool is_empty;
    LUM_FAILURE_HANDLE(luminary_path_get_is_empty(path, &is_empty));

    file_dialog_handler_release_path(display->scene_file_path);

    if (is_empty == false)
      return;
  }

  FileDialogHandlerOpenArgs open_args;
  open_args.sdl_window   = display->sdl_window;
  open_args.dialog_title = "Select scene file path";
  open_args.dialog_type  = SDL_FILEDIALOG_SAVEFILE;
  open_args.filter_type  = FILE_DIALOG_HANDLER_FILTER_TYPE_SCENE;

  file_dialog_handler_open_dialog(display->scene_file_path, &open_args);
}
