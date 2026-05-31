#ifndef MANDARIN_DUCK_FILE_DIALOG_HANDLER_H
#define MANDARIN_DUCK_FILE_DIALOG_HANDLER_H

#include <SDL3/SDL.h>

#include "utils.h"

enum FileDialogHandlerFilterType {
  FILE_DIALOG_HANDLER_FILTER_TYPE_SCENE,
  FILE_DIALOG_HANDLER_FILTER_TYPE_TEXTURE,
  FILE_DIALOG_HANDLER_FILTER_TYPE_COUNT,
} typedef FileDialogHandlerFilterType;

struct FileDialogHandler {
  LuminaryMutex* mutex;
  LuminaryPath* path;
  bool dialog_open;
} typedef FileDialogHandler;

struct FileDialogHandlerOpenArgs {
  SDL_Window* sdl_window;
  const char* dialog_title;
  SDL_FileDialogType dialog_type;
  FileDialogHandlerFilterType filter_type;
} typedef FileDialogHandlerOpenArgs;

void file_dialog_handler_create(FileDialogHandler** handler);
void file_dialog_handler_acquire_path(FileDialogHandler* handler, LuminaryPath** path);
void file_dialog_handler_try_acquire_path(FileDialogHandler* handler, LuminaryPath** path);
void file_dialog_handler_release_path(FileDialogHandler* handler);
bool file_dialog_handler_open_dialog(FileDialogHandler* handler, const FileDialogHandlerOpenArgs* args);
void file_dialog_handler_destroy(FileDialogHandler** handler);

#endif /* MANDARIN_DUCK_FILE_DIALOG_HANDLER_H */
