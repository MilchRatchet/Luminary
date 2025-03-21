#ifndef MANDARIN_DUCK_ELEMENTS_TEXT_H
#define MANDARIN_DUCK_ELEMENTS_TEXT_H

#include <assert.h>

#include "elements_common.h"
#include "utils.h"

struct ElementTextData {
  char text[256];
  ElementSize size;
  uint32_t color;
  bool center_x;
  bool center_y;
  bool highlighted;
  uint32_t highlight_padding;
  bool cache_text;
  bool is_clickable;
} typedef ElementTextData;
static_assert(sizeof(ElementTextData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

struct ElementTextArgs {
  const char* text;
  ElementSize size;
  uint32_t color;
  bool center_x;
  bool center_y;
  bool highlighting;
  bool cache_text;
  bool auto_size;
  bool is_clickable;
} typedef ElementTextArgs;

bool element_text(Window* window, Display* display, const MouseState* mouse_state, ElementTextArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_TEXT_H */
