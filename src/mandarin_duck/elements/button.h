#ifndef MANDARIN_DUCK_ELEMENTS_BUTTON_H
#define MANDARIN_DUCK_ELEMENTS_BUTTON_H

#include <assert.h>

#include "elements_common.h"
#include "utils.h"

enum ElementButtonShape { ELEMENT_BUTTON_SHAPE_CIRCLE = 0 } typedef ElementButtonShape;

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
} typedef ElementButtonArgs;

bool element_button(Window* window, Display* display, ElementButtonArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_BUTTON_H */
