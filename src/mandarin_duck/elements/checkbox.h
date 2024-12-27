#ifndef MANDARIN_DUCK_ELEMENTS_CHECKBOX_H
#define MANDARIN_DUCK_ELEMENTS_CHECKBOX_H

#include <assert.h>

#include "elements_common.h"
#include "utils.h"

struct ElementCheckBoxData {
  bool data;
  ElementSize size;
} typedef ElementCheckBoxData;
static_assert(sizeof(ElementCheckBoxData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

struct ElementCheckBoxArgs {
  void* data_binding;
  ElementSize size;
} typedef ElementCheckBoxArgs;

bool element_checkbox(Window* window, Display* display, ElementCheckBoxArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_CHECKBOX_H */
