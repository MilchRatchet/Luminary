#ifndef MANDARIN_DUCK_ELEMENTS_SEPARATOR_H
#define MANDARIN_DUCK_ELEMENTS_SEPARATOR_H

#include <assert.h>

#include "elements_common.h"
#include "utils.h"

struct ElementSeparatorData {
  char text[256];
  ElementSize size;
} typedef ElementSeparatorData;
static_assert(sizeof(ElementSeparatorData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

struct ElementSeparatorArgs {
  const char* text;
  ElementSize size;
} typedef ElementSeparatorArgs;

bool element_separator(Window* window, const MouseState* mouse_state, ElementSeparatorArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_SEPARATOR_H */
