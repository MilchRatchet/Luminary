#ifndef MANDARIN_DUCK_ELEMENTS_SLIDER_H
#define MANDARIN_DUCK_ELEMENTS_SLIDER_H

#include <assert.h>

#include "elements_common.h"
#include "utils.h"

struct ElementSliderData {
  float data_float;
  uint32_t data_uint;
  bool is_integer;
  ElementSize size;
  uint32_t color;
  bool center_x;
  bool center_y;
} typedef ElementSliderData;
static_assert(sizeof(ElementSliderData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

struct ElementSliderArgs {
  void* data_binding;
  bool is_integer;
  float max;
  float min;
  ElementSize size;
  uint32_t color;
  bool center_x;
  bool center_y;
} typedef ElementSliderArgs;

bool element_slider(Window* window, Display* display, ElementSliderArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_SLIDER_H */
