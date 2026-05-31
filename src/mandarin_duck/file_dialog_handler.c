#include "file_dialog_handler.h"

#include <SDL3/SDL_dialog.h>

void file_dialog_handler_create(FileDialogHandler** handler) {
  MD_CHECK_NULL_ARGUMENT(handler);

  LUM_FAILURE_HANDLE(host_malloc(handler, sizeof(FileDialogHandler)));
  memset(*handler, 0, sizeof(FileDialogHandler));

  LUM_FAILURE_HANDLE(luminary_path_create(&(*handler)->path));
  LUM_FAILURE_HANDLE(mutex_create(&(*handler)->mutex));
}

void file_dialog_handler_acquire_path(FileDialogHandler* handler, LuminaryPath** path) {
  MD_CHECK_NULL_ARGUMENT(handler);
  MD_CHECK_NULL_ARGUMENT(path);

  LUM_FAILURE_HANDLE(mutex_lock(handler->mutex));

  *path = handler->path;
}

void file_dialog_handler_try_acquire_path(FileDialogHandler* handler, LuminaryPath** path) {
  MD_CHECK_NULL_ARGUMENT(handler);
  MD_CHECK_NULL_ARGUMENT(path);

  bool success;
  LUM_FAILURE_HANDLE(mutex_try_lock(handler->mutex, &success));

  *path = (success) ? handler->path : (LuminaryPath*) 0;
}

void file_dialog_handler_release_path(FileDialogHandler* handler) {
  MD_CHECK_NULL_ARGUMENT(handler);

  LUM_FAILURE_HANDLE(mutex_unlock(handler->mutex));
}

static void _file_dialog_handler_sdl_callback(void* userdata, const char* const* filelist, int filter) {
  MD_CHECK_NULL_ARGUMENT(userdata);

  MD_UNUSED(filter);

  // Error
  if (filelist == (const char* const*) 0)
    return;

  const char* file = filelist[0];

  FileDialogHandler* handler = (FileDialogHandler*) userdata;

  LUM_FAILURE_HANDLE(mutex_lock(handler->mutex));

  handler->dialog_open = false;

  // No file was selected
  if (file == (const char*) 0) {
    LUM_FAILURE_HANDLE(luminary_path_clear(handler->path));

    LUM_FAILURE_HANDLE(mutex_unlock(handler->mutex));
    return;
  }

  LUM_FAILURE_HANDLE(luminary_path_set_from_string(handler->path, file));

  LUM_FAILURE_HANDLE(mutex_unlock(handler->mutex));
}

static const SDL_DialogFileFilter _sdl_filters_scene[] = {
  [0] = {.name = "Luminary Scene File (.lum)", .pattern = "lum"},
};

static const SDL_DialogFileFilter _sdl_filters_texture[] = {
  [0] = {.name = "Image File", .pattern = "png;bmp;jpg;jpeg;tga"},
  [1] = {.name = "3D lookup table (.cube)", .pattern = "cube"},
};

static const SDL_DialogFileFilter* _sdl_filters[FILE_DIALOG_HANDLER_FILTER_TYPE_COUNT] = {
  [FILE_DIALOG_HANDLER_FILTER_TYPE_SCENE]   = _sdl_filters_scene,
  [FILE_DIALOG_HANDLER_FILTER_TYPE_TEXTURE] = _sdl_filters_texture,
};

static Sint64 _sdl_filters_count[FILE_DIALOG_HANDLER_FILTER_TYPE_COUNT] = {
  [FILE_DIALOG_HANDLER_FILTER_TYPE_SCENE]   = sizeof(_sdl_filters_scene) / sizeof(SDL_DialogFileFilter),
  [FILE_DIALOG_HANDLER_FILTER_TYPE_TEXTURE] = sizeof(_sdl_filters_texture) / sizeof(SDL_DialogFileFilter),
};

bool file_dialog_handler_open_dialog(FileDialogHandler* handler, const FileDialogHandlerOpenArgs* args) {
  MD_CHECK_NULL_ARGUMENT(handler);
  MD_CHECK_NULL_ARGUMENT(args);

  if (handler->dialog_open)
    return false;

  LUM_FAILURE_HANDLE(mutex_lock(handler->mutex));

  handler->dialog_open = true;

  LUM_FAILURE_HANDLE(mutex_unlock(handler->mutex));

  SDL_PropertiesID sdl_properties = SDL_CreateProperties();
  SDL_SetStringProperty(sdl_properties, SDL_PROP_FILE_DIALOG_TITLE_STRING, args->dialog_title);
  SDL_SetPointerProperty(sdl_properties, SDL_PROP_FILE_DIALOG_WINDOW_POINTER, args->sdl_window);
  SDL_SetBooleanProperty(sdl_properties, SDL_PROP_FILE_DIALOG_MANY_BOOLEAN, false);
  SDL_SetPointerProperty(sdl_properties, SDL_PROP_FILE_DIALOG_FILTERS_POINTER, (void*) _sdl_filters[args->filter_type]);
  SDL_SetNumberProperty(sdl_properties, SDL_PROP_FILE_DIALOG_NFILTERS_NUMBER, _sdl_filters_count[args->filter_type]);

  SDL_ShowFileDialogWithProperties(
    args->dialog_type, (SDL_DialogFileCallback) _file_dialog_handler_sdl_callback, (void*) handler, sdl_properties);

  SDL_DestroyProperties(sdl_properties);

  return true;
}

void file_dialog_handler_destroy(FileDialogHandler** handler) {
  MD_CHECK_NULL_ARGUMENT(handler);
  MD_CHECK_NULL_ARGUMENT(*handler);

  LUM_FAILURE_HANDLE(luminary_path_destroy(&(*handler)->path));
  LUM_FAILURE_HANDLE(mutex_destroy(&(*handler)->mutex));

  LUM_FAILURE_HANDLE(host_free(handler));
}
