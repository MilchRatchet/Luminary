#ifndef MANDARIN_DUCK_ELEMENTS_DROPDOWN_H
#define MANDARIN_DUCK_ELEMENTS_DROPDOWN_H

#include <assert.h>

#include "elements_common.h"
#include "utils.h"

struct ElementDropdownData {
  char text[512];
} typedef ElementDropdownData;
static_assert(sizeof(ElementDropdownData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

struct ElementDropdownArgs {
  ElementSize size;
} typedef ElementDropdownArgs;

bool element_dropdown(Window* window, Display* display, ElementDropdownArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_DROPDOWN_H */
