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
  const char* identifier;
  ElementSize size;
  uint32_t* selected_index;
  uint32_t num_strings;
  char** strings;
} typedef ElementDropdownArgs;

bool element_dropdown(Window* window, const MouseState* mouse_state, ElementDropdownArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_DROPDOWN_H */
