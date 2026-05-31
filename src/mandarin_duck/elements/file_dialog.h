#ifndef MANDARIN_DUCK_ELEMENTS_FILE_DIALOG_H
#define MANDARIN_DUCK_ELEMENTS_FILE_DIALOG_H

#include "elements_common.h"
#include "file_dialog_handler.h"
#include "utils.h"

struct ElementFileDialogData {
  FileDialogHandler* handler;
  bool is_hovered;
} typedef ElementFileDialogData;
static_assert(sizeof(ElementFileDialogData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

struct ElementFileDialogArgs {
  FileDialogHandler* file_dialog_handler;
  const char* window_title;
  ElementSize size;
  SDL_FileDialogType dialog_type;
} typedef ElementFileDialogArgs;

bool element_file_dialog(Window* window, Display* display, const MouseState* mouse_state, ElementFileDialogArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_FILE_DIALOG_H */
