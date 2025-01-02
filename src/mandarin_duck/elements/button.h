#ifndef MANDARIN_DUCK_ELEMENTS_BUTTON_H
#define MANDARIN_DUCK_ELEMENTS_BUTTON_H

#include <assert.h>

#include "elements_common.h"
#include "utils.h"

enum ElementButtonShape { ELEMENT_BUTTON_SHAPE_CIRCLE = 0, ELEMENT_BUTTON_SHAPE_IMAGE = 1 } typedef ElementButtonShape;

struct ElementButtonData {
  ElementButtonShape shape;
  uint32_t shape_size_id;
  uint32_t color;
  uint32_t hover_color;
  uint32_t press_color;
  bool is_hovered;
  bool is_down;
} typedef ElementButtonData;
static_assert(sizeof(ElementButtonData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

struct ElementButtonArgs {
  ElementSize size;
  ElementButtonShape shape;
  uint32_t color;
  uint32_t hover_color;
  uint32_t press_color;
  const char* tooltip_text;
} typedef ElementButtonArgs;

bool element_button(Window* window, Display* display, const MouseState* mouse_state, ElementButtonArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_BUTTON_H */
