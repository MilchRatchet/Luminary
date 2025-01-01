#ifndef MANDARIN_DUCK_ELEMENTS_COLOR_H
#define MANDARIN_DUCK_ELEMENTS_COLOR_H

#include <assert.h>

#include "elements_common.h"
#include "utils.h"

struct ElementColorData {
  uint32_t color;
} typedef ElementColorData;
static_assert(sizeof(ElementColorData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

struct ElementColorArgs {
  ElementSize size;
  uint32_t color;
} typedef ElementColorArgs;

bool element_color(Window* window, const MouseState* mouse_state, ElementColorArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_COLOR_H */
